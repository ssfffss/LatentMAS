import time
from typing import Dict, List, Optional, Tuple

from util.infrastructure_monitor import InfrastructureMonitor
from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
from vllm import SamplingParams
import pdb

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps # the number of latent steps to do
        self.judger_max_new_tokens = judger_max_new_tokens # judger
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task

        args.enable_power_monitoring = getattr(args, 'enable_power_monitoring', True)
        self.monitor = InfrastructureMonitor(args, method_name=self.method_name)

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        ## initialize infrastructure monitor
        self.monitor.start_monitoring()
        experiment_start_time = time.time()
        tokens_processed_total = 0  # 用于功耗计算

        try:
            for list_idx, agent in enumerate(self.agents):
                step_start_time = time.time()
                step_input_data = None
                step_output_data = None
                step_latent_vectors = None
                step_kv_cache_size = 0.0
                step_tokens_processed = 0.0

                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]

                prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )

                step_tokens_processed = attention_mask.sum().item()

                if agent.role != "judger":
                    prev_past_len = _past_length(past_kv)

                    if self.args.think:
                            wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                    else: 
                        wrapped_prompts = prompts

                    wrapped_encoded = self.model.tokenizer(
                        wrapped_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                    wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                    wrapped_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                    torch.cuda.reset_peak_memory_stats()
                    step_inference_start = time.time()

                    past_kv = self.model.generate_latent_batch(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )

                    step_inference_time = time.time() - step_inference_start
                    
                    # 计算额外处理的token数
                    if hasattr(self.model, 'last_tokens_processed'):
                        step_tokens_processed += self.model.last_tokens_processed
                    
                    tokens_processed_total += step_tokens_processed

                    if self.sequential_info_only or self.latent_only:
                        new_past_len = _past_length(past_kv)
                        tokens_added = new_past_len - prev_past_len
                        tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                        past_kv = self._truncate_past(past_kv, tokens_to_keep)

                    for idx in range(batch_size):
                        mask = wrapped_mask[idx].bool()
                        trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": wrapped_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": wrapped_tokens_batch[idx],
                                "latent_steps": self.latent_steps,
                                "output": "",
                            }
                        )
                    
                    # 记录latent vectors用于功耗分析
                    if hasattr(self.model, 'last_latent_vectors'):
                        step_latent_vectors = self.model.last_latent_vectors
                    
                    # 获取KV缓存大小
                    if past_kv is not None:
                        step_kv_cache_size = sum(
                            sum(p.numel() * p.element_size() for p in layer) 
                            for layer in past_kv
                        ) / (1024**3)  # GB
                    
                    # 记录Agent通信指标（包含功耗）
                    self.monitor.record_agent_communication(
                        agent_name=agent.name,
                        role=agent.role,
                        step_idx=list_idx,
                        batch_size=batch_size,
                        input_data=batch_messages,
                        output_data="latent_thoughts",  # latent输出
                        latent_vectors=step_latent_vectors,
                        kv_cache_size=step_kv_cache_size,
                        inference_time=step_inference_time,
                        tokens_processed=step_tokens_processed,
                        samples_processed=batch_size
                    )
                else:
                    past_for_decoding = past_kv if self.latent_steps > 0 else None

                    if self.args.think:
                            judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                    else: 
                        judger_prompts = prompts
                    
                    judger_encoded = self.model.tokenizer(
                        judger_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    judger_ids = judger_encoded["input_ids"].to(self.model.device)
                    judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                    judger_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(judger_ids, judger_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                    # 记录推理前的状态
                    torch.cuda.reset_peak_memory_stats()
                    step_inference_start = time.time()
                    generated_batch, _ = self.model.generate_text_batch(
                        judger_ids,
                        judger_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_for_decoding,
                    )

                    step_inference_time = time.time() - step_inference_start
                
                    # 计算生成的token数
                    generated_tokens = sum(len(text.split()) for text in generated_batch)
                    step_tokens_processed += generated_tokens
                    tokens_processed_total += step_tokens_processed

                    for idx in range(batch_size):
                        final_text = generated_batch[idx].strip()
                        final_texts[idx] = final_text
                        mask = judger_mask[idx].bool()
                        trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": judger_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": judger_tokens_batch[idx],
                                "output": final_text,
                            }
                        )
                    step_output_data = final_texts
                    step_input_data = judger_prompts

                    # 获取KV缓存大小
                    if past_kv is not None:
                        step_kv_cache_size = sum(
                            sum(p.numel() * p.element_size() for p in layer) 
                            for layer in past_kv
                        ) / (1024**3)  # GB

                    # 记录Judger通信指标（包含功耗）
                    self.monitor.record_agent_communication(
                        agent_name=agent.name,
                        role=agent.role,
                        step_idx=list_idx,
                        batch_size=batch_size,
                        input_data=judger_prompts,
                        output_data=generated_batch,
                        latent_vectors=None,  # Text输出
                        kv_cache_size=step_kv_cache_size,
                        inference_time=step_inference_time,
                        tokens_processed=step_tokens_processed,
                        samples_processed=batch_size
                    )
                
                # # compute the agent inference time
                # step_time = time.time() - step_start_time
                
                # # estimate the number of tokens processed
                # tokens_processed = 0
                # if hasattr(self.model, 'last_tokens_processed'):
                #     tokens_processed = self.model.last_tokens_processed
                
                # # 记录Agent通信指标
                # monitor.record_agent_communication(
                #     agent_name=agent.name,
                #     role=agent.role,
                #     step_idx=list_idx,
                #     batch_size=batch_size,
                #     input_data=step_input_data,
                #     output_data=step_output_data,
                #     latent_vectors=step_latent_vectors,
                #     kv_cache_size=step_kv_cache_size,
                #     inference_time=step_time,
                #     tokens_processed=tokens_processed
                # )

            results: List[Dict] = []
            for idx, item in enumerate(items):
                final_text = final_texts[idx]
                if self.task in ['mbppplus', 'humanevalplus']:
                    pred = extract_markdown_python_block(final_text)
                    gold = item.get("gold", "")

                    if pred is None:
                        ok = False
                        error_msg = "python error: No python code block found"
                    else:
                        python_code_to_exe = pred + "\n" + gold
                        ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                    
                    print(f'=========================================')
                    print(f'Question {idx}')
                    print(f'error_msg: {error_msg}')
                    # print(f'=========================================')

                elif self.task in ["aime2024", "aime2025"]:
                    pred = normalize_answer(extract_gsm8k_answer(final_text))
                    gold = str(item.get("gold", "")).strip()
                    try:
                        pred_int = int(pred)
                        gold_int = int(gold)
                        ok = (pred_int == gold_int)
                        error_msg = None
                    except ValueError:
                        ok = False
                        error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

                else:
                    pred = normalize_answer(extract_gsm8k_answer(final_text))
                    gold = item.get("gold", "")
                    ok = (pred == gold) if (pred and gold) else False
                    error_msg = None
                
                results.append(
                    {
                        "question": item["question"],
                        "gold": gold,
                        "solution": item["solution"],
                        "prediction": pred,
                        "raw_prediction": final_text,
                        "agents": agent_traces[idx],
                        "correct": ok,
                    }
                )
            
            # 计算整体实验时间
            total_time = time.time() - experiment_start_time
            
            # 停止监控并保存指标
            self.monitor.stop_monitoring()
            
            # 生成实验名称
            experiment_name = f"{self.method_name}_{self.args.model_name.split('/')[-1]}_{self.task}_{batch_size}"
            
            # 保存指标
            metrics_path = self.monitor.save_metrics(
                output_dir=getattr(self.args, 'metrics_output_dir', 'metrics'),
                experiment_name=experiment_name
            )
            
            # 打印摘要统计
            summary_stats = self.monitor.get_summary_statistics()
            print("\n" + "="*50)
            print(f"Infrastructure Metrics Summary for {self.method_name}")
            print("="*50)
            for dimension, stats in summary_stats.items():
                print(f"\n{dimension.upper()} METRICS:")
                for metric_name, values in stats.items():
                    if values['count'] > 0:
                        print(f"  {metric_name}:")
                        print(f"    mean: {values['mean']:.4f}")
                        print(f"    std:  {values['std']:.4f}")
                        print(f"    min:  {values['min']:.4f}")
                        print(f"    max:  {values['max']:.4f}")
            
            print(f"\nTotal experiment time: {total_time:.2f} seconds")
            print(f"Metrics saved to: {metrics_path}")
            print("="*50)

            # 打印功耗摘要
            power_summary = self.monitor.get_power_summary()
            if power_summary:
                print("\n" + "="*60)
                print(f"POWER CONSUMPTION SUMMARY for {self.method_name}")
                print("="*60)
                print(f"Total Energy Consumed: {power_summary['total_energy_consumed']:.2f} J")
                print(f"Average Power Draw: {power_summary['avg_power_draw']:.2f} W")
                print(f"Peak Power Draw: {power_summary['peak_power_draw']:.2f} W")
                print(f"GPU Energy Fraction: {power_summary['gpu_energy_fraction']*100:.1f}%")
                print(f"CPU Energy Fraction: {power_summary['cpu_energy_fraction']*100:.1f}%")
                print(f"DRAM Energy Fraction: {power_summary['dram_energy_fraction']*100:.1f}%")
                print(f"Tokens per Joule: {power_summary['avg_tokens_per_watt']:.2f}")
                print(f"Samples per Joule: {power_summary['avg_samples_per_watt']:.2f}")
                print(f"Total Tokens Processed: {tokens_processed_total}")
                print(f"Energy per Sample: {power_summary['total_energy_consumed']/len(items):.2f} J/sample")
                print("="*60)

            return results
        finally:
            self.monitor.stop_monitoring()
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # initialize and start the monitor 
        self.monitor.start_monitoring()
        experiment_start_time = time.time()
        tokens_processed_total = 0.0  # 用于功耗计算

        embedding_record = []

        try: 
            for agent_idx, agent in enumerate(self.agents):
                agent_start_time = time.time()
                agent_tokens_processed = 0.0
                agent_latent_vectors = None
                agent_kv_cache_size = 0.0
                agent_tokens_processed = 0.0

                if self.args.prompt == "sequential":
                    batch_messages = [
                        build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                elif self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                        for item in items
                    ]
                    
                prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )

                agent_tokens_processed = attention_mask.sum().item()

                if agent.role != "judger":
                    prev_past_len = _past_length(past_kv)

                    # to wrap all latent thoughts from previous agents
                    if self.args.think:
                            wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                    else: 
                        wrapped_prompts = prompts

                    wrapped_encoded = self.model.tokenizer(
                        wrapped_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                    wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                    wrapped_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                    
                    torch.cuda.reset_peak_memory_stats()
                    agent_inference_start = time.time()

                    past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )

                    agent_inference_time = time.time() - agent_inference_start

                    # 计算额外处理的token数
                    if hasattr(self.model, 'last_tokens_processed'):
                        agent_tokens_processed += self.model.last_tokens_processed
                    
                    tokens_processed_total += agent_tokens_processed

                    if self.sequential_info_only or self.latent_only:
                        new_past_len = _past_length(past_kv)
                        tokens_added = new_past_len - prev_past_len
                        tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                        past_kv = self._truncate_past(past_kv, tokens_to_keep)

                    if self.latent_only:
                        if self.latent_steps > 0:
                            previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                        else:
                            previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                    embedding_record.append(previous_hidden_embedding)

                    if self.sequential_info_only or self.latent_only:
                        embedding_record = embedding_record[-1:]
                    
                    for idx in range(batch_size):
                        mask = wrapped_mask[idx].bool()
                        trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": wrapped_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": wrapped_tokens_batch[idx],
                                "latent_steps": self.latent_steps,
                                "output": "",
                            }
                        )
                    
                    # 记录latent vectors用于功耗分析
                    if hasattr(self.model, 'last_latent_vectors'):
                        agent_latent_vectors = self.model.last_latent_vectors
                    
                    # 获取KV缓存大小
                    if past_kv is not None:
                        agent_kv_cache_size = sum(
                            sum(p.numel() * p.element_size() for p in layer) 
                            for layer in past_kv
                        ) / (1024**3)  # GB
                    
                    # 记录Agent通信指标（包含功耗）
                    self.monitor.record_agent_communication(
                        agent_name=agent.name,
                        role=agent.role,
                        step_idx=agent_idx,
                        batch_size=batch_size,
                        input_data=batch_messages,
                        output_data="latent_thoughts",  # latent输出
                        latent_vectors=agent_latent_vectors,
                        kv_cache_size=agent_kv_cache_size,
                        inference_time=agent_inference_time,
                        tokens_processed=agent_tokens_processed,
                        samples_processed=batch_size
                    )
                else:
                    # A stack of [B, L_i, H]
                    past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                    
                    if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                    else: 
                        judger_prompts = prompts
                    
                    judger_encoded = self.model.tokenizer(
                        judger_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ) 
                    judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                    # Get current prompt embedding
                    curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)
                    
                    # assert Qwen model
                    assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                    # handle latent embedding insertion position    
                    len_of_left = []
                    for p in judger_prompts:
                        idx = p.find("<|im_start|>user\n")
                        # Get the text up to and including "<|im_start|>user\n"
                        left = p[: idx + len("<|im_start|>user\n")]
                        len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                        
                    B, L, H = curr_prompt_emb.shape
                    _, Lp, H = past_embedding.shape  # assume shape consistency
                        
                    whole_prompt_emb_list = []
                    for i in range(B):
                        insert_idx = len_of_left[i]
                        left_emb = curr_prompt_emb[i, :insert_idx, :]
                        right_emb = curr_prompt_emb[i, insert_idx:, :]
                        combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                        whole_prompt_emb_list.append(combined)

                    # Pad back to max length if needed
                    max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                    whole_prompt_emb = torch.stack([
                        torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                        for x in whole_prompt_emb_list
                    ])

                    # else:
                        # Get full prompt embedding from cat with previous ones 
                        # B L H B L H
                        # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                    
                    # pdb.set_trace()              
                    
                    # Use vLLM 
                    prompt_embeds_list = [
                        {
                            "prompt_embeds": embeds
                        } for embeds in whole_prompt_emb 
                    ]
                    
                    # 记录推理前的状态
                    torch.cuda.reset_peak_memory_stats()
                    agent_inference_start = time.time()
                    
                    outputs = self.model.vllm_engine.generate(
                        prompt_embeds_list,
                        self.sampling_params,
                    )

                    agent_inference_time = time.time() - agent_inference_start

                    generated_texts = [out.outputs[0].text.strip() for out in outputs]

                    agent_inference_time = time.time() - agent_inference_start
                    
                    # 计算生成的token数
                    generated_tokens = sum(len(text.split()) for text in generated_texts)
                    agent_tokens_processed += generated_tokens
                    tokens_processed_total += agent_tokens_processed
                        
                    for idx in range(batch_size):
                        text_out = generated_texts[idx].strip()
                        final_texts[idx] = text_out
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": judger_prompts[idx],
                                "output": text_out,
                            }
                        )
                    
                    # 获取KV缓存大小
                    if past_kv is not None:
                        step_kv_cache_size = sum(
                            sum(p.numel() * p.element_size() for p in layer) 
                            for layer in past_kv
                        ) / (1024**3)  # GB

                    # 记录Judger通信指标（包含功耗）
                    self.monitor.record_agent_communication(
                        agent_name=agent.name,
                        role=agent.role,
                        step_idx=agent_idx,
                        batch_size=batch_size,
                        input_data=judger_prompts,
                        output_data=generated_texts,
                        latent_vectors=None,  # Text输出
                        kv_cache_size=step_kv_cache_size,
                        inference_time=agent_inference_time,
                        tokens_processed=agent_tokens_processed,
                        samples_processed=batch_size
                    )


            results: List[Dict] = []
            for idx, item in enumerate(items):
                final_text = final_texts[idx]
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item["gold"]
                ok = (pred == gold) if (pred and gold) else False
                results.append(
                    {
                        "question": item["question"],
                        "gold": gold,
                        "solution": item["solution"],
                        "prediction": pred,
                        "raw_prediction": final_text,
                        "agents": agent_traces[idx],
                        "correct": ok,
                    }
                )

            # 计算整体实验时间
            total_time = time.time() - experiment_start_time
            
            # 停止监控并保存指标
            self.monitor.stop_monitoring()
            
            # 生成实验名称
            experiment_name = f"{self.method_name}_{self.args.model_name.split('/')[-1]}_{self.task}_{batch_size}"
            
            # 保存指标
            metrics_path = self.monitor.save_metrics(
                output_dir=getattr(self.args, 'metrics_output_dir', 'metrics'),
                experiment_name=experiment_name
            )
            
            # 打印摘要统计
            summary_stats = self.monitor.get_summary_statistics()
            print("\n" + "="*50)
            print(f"Infrastructure Metrics Summary for {self.method_name}")
            print("="*50)
            for dimension, stats in summary_stats.items():
                print(f"\n{dimension.upper()} METRICS:")
                for metric_name, values in stats.items():
                    if values['count'] > 0:
                        print(f"  {metric_name}:")
                        print(f"    mean: {values['mean']:.4f}")
                        print(f"    std:  {values['std']:.4f}")
                        print(f"    min:  {values['min']:.4f}")
                        print(f"    max:  {values['max']:.4f}")
            
            print(f"\nTotal experiment time: {total_time:.2f} seconds")
            print(f"Metrics saved to: {metrics_path}")
            print("="*50)

            # 打印功耗摘要
            power_summary = self.monitor.get_power_summary()
            if power_summary:
                print("\n" + "="*60)
                print(f"POWER CONSUMPTION SUMMARY for {self.method_name}")
                print("="*60)
                print(f"Total Energy Consumed: {power_summary['total_energy_consumed']:.2f} J")
                print(f"Average Power Draw: {power_summary['avg_power_draw']:.2f} W")
                print(f"Peak Power Draw: {power_summary['peak_power_draw']:.2f} W")
                print(f"GPU Energy Fraction: {power_summary['gpu_energy_fraction']*100:.1f}%")
                print(f"Tokens per Joule: {power_summary['avg_tokens_per_joule']:.2f}")
                print(f"Samples per Joule: {power_summary['avg_samples_per_joule']:.2f}")
                print(f"Total Tokens Processed: {tokens_processed_total}")
                print(f"Energy per Sample: {power_summary['total_energy_consumed']/len(items):.2f} J/sample")
                print("="*60)

            return results
        finally:
            self.monitor.stop_monitoring()

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
