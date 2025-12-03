import time
from typing import Dict, List

import torch

from util.infrastructure_monitor import InfrastructureMonitor

from . import default_agents
from models import ModelWrapper
# from prompts import build_agent_messages, build_agent_messages_v6, build_agent_messages_v6_text_mas
from prompts import build_agent_messages_hierarchical_text_mas, build_agent_messages_sequential_text_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import argparse
import pdb

class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task

        # 初始化基础设施监控器
        self.monitor = InfrastructureMonitor(args, method_name=self.method_name)
        
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        # 启动基础设施监控
        self.monitor.start_monitoring()
        experiment_start_time = time.time()
        tokens_processed_total = 0  # 总处理token数

        try:
            for step_idx, agent in self.agents:
                step_start_time = time.time()
                step_input_data = None
                step_output_data = None
                step_kv_cache_size = 0.0
                if self.args.prompt == "hierarchical":
                    batch_messages = [
                        build_agent_messages_hierarchical_text_mas(
                            role=agent.role,
                            question=item["question"],
                            context=contexts[idx],
                            method=self.method_name,
                            args=self.args,
                        )
                        for idx, item in enumerate(items)
                    ]
                else:
                    batch_messages = [
                        build_agent_messages_sequential_text_mas(
                            role=agent.role,
                            question=item["question"],
                            context=contexts[idx],
                            method=self.method_name,
                            args=self.args,
                        )
                        for idx, item in enumerate(items)
                    ]
                
                 # 记录输入数据
                step_input_data = batch_messages

                prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )

                # 记录推理前的状态
                torch.cuda.reset_peak_memory_stats()
                step_inference_start = time.time()

                if self.model.use_vllm:
                    generated_texts = self.model.vllm_generate_text_batch(
                        prompts,
                        max_new_tokens=self.max_new_tokens_each,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                else:
                    generated_texts, _ = self.model.generate_text_batch(
                        input_ids,
                        attention_mask,
                        max_new_tokens=self.max_new_tokens_each,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )

                step_inference_time = time.time() - step_inference_start
                
                # 计算处理的token数
                input_tokens_count = sum(len(tokens) for tokens in tokens_batch)
                output_tokens_count = sum(len(self.model.tokenizer.encode(text)) for text in generated_texts)
                step_tokens_processed = input_tokens_count + output_tokens_count
                tokens_processed_total += step_tokens_processed
                
                # 获取KV缓存大小（如果可用）
                if hasattr(self.model, 'last_kv_cache') and self.model.last_kv_cache is not None:
                    step_kv_cache_size = sum(
                        sum(p.numel() * p.element_size() for p in layer) 
                        for layer in self.model.last_kv_cache
                    ) / (1024**3)  # GB

                agent_name_map_for_prompt_hierarchical = {
                    "Planner": "Math Agent",
                    "Critic": "Science Agent",
                    "Refiner": "Code Agent",
                    "Judger": "Task Summrizer",
                    "planner": "Math Agent",
                    "critic": "Science Agent",
                    "refiner": "Code Agent",
                    "judger": "Task Summrizer",
                }

                for idx in range(batch_size):

                    text_out = generated_texts[idx].strip()

                    if self.args.prompt == "hierarchical":
                        formatted_output = f"[{agent_name_map_for_prompt_hierarchical[agent.name]}]:\n{text_out}\n\n"
                    else:
                        formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                    if agent.role != "judger":

                        contexts[idx] = f"{contexts[idx]}{formatted_output}"
                        history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                    else:
                        final_texts[idx] = text_out
                    mask = attention_mask[idx].bool()
                    trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": tokens_batch[idx],
                            "output": text_out,
                        }
                    )

                    # 记录输出数据（对于分析很重要）
                    step_output_data = generated_texts
                # import pdb; pdb.set_trace()

                self.monitor.record_agent_communication(
                    agent_name=agent.name,
                    role=agent.role,
                    step_idx=step_idx,
                    batch_size=batch_size,
                    input_data=step_input_data,           # 输入提示
                    output_data=step_output_data,         # 生成的文本
                    latent_vectors=None,                  # TextMAS没有latent vectors
                    kv_cache_size=step_kv_cache_size,     # KV缓存大小
                    inference_time=step_inference_time,   # 推理时间
                    tokens_processed=step_tokens_processed,  # 处理的token数
                    samples_processed=batch_size          # 处理的样本数
                )

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
                        "context": history_contexts[idx],
                        "prediction": pred,
                        "raw_prediction": final_text,
                        "agents": agent_traces[idx],
                        "correct": ok,
                    }
                )
        
            # 实验完成，停止监控
            total_time = time.time() - experiment_start_time
            self.monitor.stop_monitoring()
            
            # 生成实验名称
            experiment_name = f"{self.method_name}_{self.args.model_name.split('/')[-1]}_{self.task}_{batch_size}"
            
            # 保存指标
            metrics_path = self.monitor.save_metrics(
                output_dir=getattr(self.args, 'metrics_output_dir', 'metrics'),
                experiment_name=experiment_name
            )
            
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
            # 确保监控器总是停止
            if self.monitor.is_monitoring:
                self.monitor.stop_monitoring()

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
