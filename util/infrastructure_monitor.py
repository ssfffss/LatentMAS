import time
import torch
import psutil
import numpy as np
import pynvml
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
from collections import defaultdict, deque
import copy

from util.infrastructure_analyzer import convert_numpy_types

@dataclass
class ComputeMetrics:
    gpu_util: float = 0.0
    gpu_mem_used: float = 0.0
    gpu_mem_total: float = 0.0
    cpu_util: float = 0.0
    cpu_mem_used: float = 0.0
    flops_estimate: float = 0.0  # GFLOPS
    inference_time: float = 0.0  # seconds
    total_time: float = 0.0 # seconds
    tokens_per_second: float = 0.0

@dataclass
class MemoryMetrics:
    vram_allocated: float = 0.0  # GB
    vram_reserved: float = 0.0   # GB
    vram_peak: float = 0.0       # GB
    ram_used: float = 0.0        # GB
    kv_cache_size: float = 0.0   # GB
    model_size: float = 0.0      # GB
    latent_vector_size: float = 0.0  # MB

@dataclass
class NetworkMetrics:
    comm_data_size: float = 0.0  # MB
    comm_latency: float = 0.0    # ms
    bandwidth_usage: float = 0.0 # MB/s
    packets_sent: int = 0
    packets_received: int = 0
    agent_comm_count: int = 0

@dataclass
class StorageMetrics:
    input_data_size: float = 0.0    # MB
    output_data_size: float = 0.0   # MB
    intermediate_data_size: float = 0.0  # MB
    log_data_size: float = 0.0      # MB
    checkpoint_size: float = 0.0    # MB
    io_throughput: float = 0.0      # MB/s

@dataclass
class PowerMetrics:
    total_energy_consumed: float = 0.0  # Joules
    avg_power_draw: float = 0.0         # joules/s = W
    peak_power_draw: float = 0.0        # joules/s = W
    gpu_energy_fraction: float = 0.0    # 0-1
    cpu_energy_fraction: float = 0.0    # 0-1
    dram_energy_fraction: float = 0.0 # 0-1
    avg_tokens_per_joule: float = 0.0   # tokens/J
    avg_samples_per_joule: float = 0.0  # samples/J
    measurement_duration: float = 0.0   # seconds

@dataclass
class AgentCommunicationMetrics:
    method: str = "unknown"  # "text_mas", "latent_mas", "baseline"
    agent_name: str = ""
    role: str = ""
    step_idx: int = 0
    batch_size: int = 0
    compute: ComputeMetrics = None
    memory: MemoryMetrics = None
    network: NetworkMetrics = None
    storage: StorageMetrics = None
    power: PowerMetrics = None
    duration: float = 0.0
    experiment_id: str = ""  # 新增实验ID字段
    
    def __post_init__(self):
        if self.compute is None:
            self.compute = ComputeMetrics()
        if self.memory is None:
            self.memory = MemoryMetrics()
        if self.network is None:
            self.network = NetworkMetrics()
        if self.storage is None:
            self.storage = StorageMetrics()
        if self.power is None:
            self.power = PowerMetrics()

class InfrastructureMonitor:
    def __init__(self, args, method_name: str = "unknown"):
        self.args = args
        self.cpu_count = os.cpu_count()
        self.method_name = method_name
        self.metrics_queue = queue.Queue()
        self.power_metrics_queue = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        self.power_monitor_thread = None
        self.start_time = 0.0
        self.last_metric_collection_time = 0.0
        
        # 重构：使用时间窗口缓存
        self.metrics_buffer = deque(maxlen=1000)  # 保留最近1000个指标点
        self.power_buffer = deque(maxlen=1000)
        
        # 重构：实验隔离机制
        self.current_experiment_id = None
        self.experiment_start_time = 0.0
        self.agent_metrics = defaultdict(lambda: defaultdict(list))
        self.overall_metrics = {
            'compute': [],
            'memory': [],
            'network': [],
            'storage': [],
            'power': []
        }
        
        # 模型参数估计
        self.model_params = 0
        self._estimate_model_flops()
        
        # GPU监控初始化
        self.gpu_handles = []
        self.gpu_names = []
        self.gpu_total_mems = []
        self._init_gpu_monitoring()
        
        # 功耗监控状态
        self.power_monitoring_enabled = hasattr(args, 'enable_power_monitoring') and args.enable_power_monitoring
        
        # 基准测量
        self._initialize_baselines()
    
    def _initialize_baselines(self):
        """初始化基准测量值"""
        # CPU基准
        self.cpu_base = self._get_cpu_metrics()
        
        # 网络基准
        self.network_baseline = self._get_network_stats()
        
        # 功耗基准
        if self.power_monitoring_enabled:
            self.power_base = self._get_power_metrics()
            print(f"Power baseline established: CPU={self.power_base['cpu_power']:.2f}W, GPU={self.power_base['gpu_power']:.2f}W, DRAM={self.power_base['dram_power']:.2f}W")
    
    def _init_gpu_monitoring(self):
        """初始化GPU监控 - 支持多GPU"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode()
                    gpu_total_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)  # GB
                    
                    self.gpu_handles.append(handle)
                    self.gpu_names.append(gpu_name)
                    self.gpu_total_mems.append(gpu_total_mem)
                    
                    print(f"GPU {i} initialized: {gpu_name}, Total Memory: {gpu_total_mem:.2f} GB")
                except Exception as e:
                    print(f"Warning: Failed to initialize GPU {i}: {e}")
            
            if not self.gpu_handles:
                print("Warning: No GPUs initialized successfully")
                
        except Exception as e:
            print(f"Warning: GPU monitoring initialization failed: {e}")
    
    def _get_gpu_metrics(self, gpu_selected_ids: list = [0,1]) -> Dict[str, float]:
        """获取GPU指标 - 支持多GPU聚合"""
        if gpu_selected_ids is None:
            gpu_selected_ids = list(range(len(self.gpu_handles)))
        
        if not self.gpu_handles or not gpu_selected_ids:
            return {
                'gpu_util': 0.0,
                'gpu_mem_used': 0.0,
                'gpu_mem_total': sum(self.gpu_total_mems) if self.gpu_total_mems else 0.0
            }
        
        try:
            total_util = 0
            total_mem_used = 0
            total_mem_total = 0
            
            for i in gpu_selected_ids:
                if i >= len(self.gpu_handles):
                    continue
                try:
                    handle = self.gpu_handles[i]
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    total_util += util.gpu
                    total_mem_used += mem.used / (1024**3)  # GB
                    total_mem_total += mem.total / (1024**3)  # GB
                except Exception as e:
                    print(f"Warning: Failed to get metrics for GPU {i}: {e}")
            
            avg_util = total_util # / len(gpu_selected_ids) if gpu_selected_ids else 0.0
            
            return {
                'gpu_util': avg_util,
                'gpu_mem_used': total_mem_used,
                'gpu_mem_total': total_mem_total
            }
        except Exception as e:
            print(f"Warning: Failed to get GPU metrics: {e}")
            return {
                'gpu_util': 0.0,
                'gpu_mem_used': 0.0,
                'gpu_mem_total': sum(self.gpu_total_mems) if self.gpu_total_mems else 0.0
            }
    
    def _estimate_model_flops(self):
        """估计模型FLOPS"""
        if hasattr(self.args, 'model_name'):
            model_name_lower = self.args.model_name.lower()
            if '7b' in model_name_lower:
                self.model_params = 7e9
            elif '14b' in model_name_lower:
                self.model_params = 14e9
            elif '32b' in model_name_lower:
                self.model_params = 32e9
            elif '70b' in model_name_lower:
                self.model_params = 70e9
            else:
                self.model_params = 7e9
    
    def _get_cpu_metrics(self) -> Dict[str, float]:
        """获取CPU和内存指标，自动处理基准"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        
        result = {
            'cpu_util': cpu_percent * self.cpu_count,
            'cpu_mem_used': mem.used / (1024**3),  # GB
            'cpu_mem_total': mem.total / (1024**3)  # GB
        }
        
        # 如果有基准，计算相对值
        if hasattr(self, 'cpu_base') and self.cpu_base is not None:
            result['cpu_util'] = max(result['cpu_util'] - self.cpu_base['cpu_util'], 0)
            result['cpu_mem_used'] = max(result['cpu_mem_used'] - self.cpu_base['cpu_mem_used'], 0)
        
        return result
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def _get_torch_memory_stats(self) -> Dict[str, float]:
        """获取PyTorch内存统计"""
        if not torch.cuda.is_available():
            return {'allocated': 0.0, 'reserved': 0.0, 'peak': 0.0}
        
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        peak = torch.cuda.max_memory_allocated() / (1024**3)   # GB
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'peak': peak
        }
    
    def _get_cpu_power(self, label="measurement"):
        """获取CPU功耗，使用pyRAPL（如果可用）"""
        try:
            import pyRAPL
            pyRAPL.setup()
            measure = pyRAPL.Measurement(label)
            measure.begin()
            time.sleep(0.1)  # 短暂测量
            measure.end()
            cpu_package_power = np.array(measure.result.pkg) / measure.result.duration  # 转换为W
            dram_package_power = np.array(measure.result.dram) / measure.result.duration # 转换为W
            return cpu_package_power.sum(), dram_package_power.sum()
        except ImportError:
            print("pyRAPL not available, using estimated CPU power")
            # 估算：每核心2W基础功耗 + 每10%利用率1W
            cpu_util = psutil.cpu_percent() / 100
            est_cpu_power = self.cpu_count * 2 + (self.cpu_count * 10 * cpu_util)
            est_dram_power = self.cpu_count * 0.5  # 估计DRAM功耗
            return est_cpu_power, est_dram_power
        except Exception as e:
            print(f"Error measuring CPU power: {e}")
            return 0.0, 0.0
    
    def _get_gpu_power_metrics(self, gpu_ids: list = [0, 1]) -> Dict[int, Dict]:
        """获取指定GPU的功耗指标"""
        if gpu_ids is None:
            gpu_ids = list(range(len(self.gpu_handles)))
        
        power_metrics = {}
        for i in gpu_ids:
            if i >= len(self.gpu_handles):
                continue
            try:
                handle = self.gpu_handles[i]
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                timestamp = time.time()
                
                power_metrics[i] = {
                    'gpu_power_draw': power_draw,
                    'timestamp': timestamp
                }
                
            except Exception as e:
                print(f"Warning: Failed to get power metrics for GPU {i}: {e}")
        
        return power_metrics
    
    def _get_power_metrics(self) -> Dict[str, float]:
        """获取系统功耗指标，处理基准校正"""
        timestamp = time.time()
        
        # 获取CPU和DRAM功耗
        cpu_power, dram_power = self._get_cpu_power("instant")
        
        # 获取GPU功耗
        gpu_power_metrics = self._get_gpu_power_metrics()
        gpu_power = sum(power['gpu_power_draw'] for power in gpu_power_metrics.values()) if gpu_power_metrics else 0.0
        
        # 应用基准校正
        if hasattr(self, 'power_base') and self.power_base is not None:
            cpu_power = max(cpu_power - self.power_base['cpu_power'], 0)
            dram_power = max(dram_power - self.power_base['dram_power'], 0)
            gpu_power = max(gpu_power - self.power_base['gpu_power'], 0)
        
        return {
            'cpu_power': cpu_power,
            'dram_power': dram_power,
            'gpu_power': gpu_power,
            'total_power': cpu_power + dram_power + gpu_power,
            'timestamp': timestamp
        }
    
    def start_monitoring(self, experiment_id: str = None):
        """开始监控，支持实验ID"""
        if self.is_monitoring:
            self.stop_monitoring()
        
        self.current_experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.is_monitoring = True
        self.start_time = time.time()
        self.last_metric_collection_time = self.start_time
        self.experiment_start_time = self.start_time
        
        # 重置缓冲区
        self.metrics_buffer.clear()
        self.power_buffer.clear()
        
        # 重置指标存储
        self.agent_metrics = defaultdict(lambda: defaultdict(list))
        self.overall_metrics = {
            'compute': [],
            'memory': [],
            'network': [],
            'storage': [],
            'power': []
        }
        
        # 重新初始化基准
        self._initialize_baselines()
        
        # 基础监控线程
        def monitoring_loop():
            last_collection = time.time()
            while self.is_monitoring:
                current_time = time.time()
                if current_time - last_collection >= 1.0:  # 1秒采样间隔
                    gpu_metrics = self._get_gpu_metrics()
                    cpu_metrics = self._get_cpu_metrics()
                    torch_mem = self._get_torch_memory_stats()
                    
                    metrics = {
                        'timestamp': current_time - self.start_time,
                        'absolute_timestamp': current_time,
                        'gpu_util': gpu_metrics['gpu_util'],
                        'gpu_mem_used': gpu_metrics['gpu_mem_used'],
                        'cpu_util': cpu_metrics['cpu_util'],
                        'cpu_mem_used': cpu_metrics['cpu_mem_used'],
                        'torch_allocated': torch_mem['allocated'],
                        'torch_reserved': torch_mem['reserved'],
                        'torch_peak': torch_mem['peak']
                    }
                    
                    self.metrics_queue.put(metrics)
                    self.metrics_buffer.append(metrics)
                    last_collection = current_time
                
                time.sleep(0.1)  # 避免CPU占用过高
        
        # 功耗监控线程
        def power_monitoring_loop():
            last_collection = time.time()
            while self.is_monitoring and self.power_monitoring_enabled:
                current_time = time.time()
                if current_time - last_collection >= 0.5:  # 0.5秒采样间隔
                    power_metrics = self._get_power_metrics()
                    self.power_metrics_queue.put(power_metrics)
                    self.power_buffer.append(power_metrics)
                    last_collection = current_time
                
                time.sleep(0.1)
        
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.power_monitoring_enabled:
            self.power_monitor_thread = threading.Thread(target=power_monitoring_loop, daemon=True)
            self.power_monitor_thread.start()
        
        print(f"Infrastructure monitoring started for experiment: {self.current_experiment_id}")
        return self.current_experiment_id
    
    def stop_monitoring(self):
        """停止监控并返回实验ID"""
        experiment_id = self.current_experiment_id
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        if self.power_monitor_thread:
            self.power_monitor_thread.join(timeout=1.0)
        
        # 清理GPU监控
        try:
            if hasattr(pynvml, 'nvmlShutdown') and pynvml.nvmlShutdown:
                pynvml.nvmlShutdown()
        except Exception as e:
            print(f"Warning during GPU cleanup: {e}")
        
        print(f"Infrastructure monitoring stopped for experiment: {experiment_id}")
        return experiment_id
    
    def _collect_metrics_for_window(self, start_time: float, end_time: float):
        """收集指定时间窗口内的指标"""
        window_metrics = {
            'compute': [],
            'memory': [],
            'power': []
        }
        
        # 收集常规指标
        while not self.metrics_queue.empty():
            metric = self.metrics_queue.get()
            if start_time <= metric['absolute_timestamp'] <= end_time:
                window_metrics['compute'].append({
                    'gpu_util': metric['gpu_util'],
                    'cpu_util': metric['cpu_util'],
                    'duration': metric['timestamp']
                })
                window_metrics['memory'].append({
                    'gpu_mem_used': metric['gpu_mem_used'],
                    'cpu_mem_used': metric['cpu_mem_used'],
                    'torch_allocated': metric['torch_allocated'],
                    'torch_reserved': metric['torch_reserved'],
                    'torch_peak': metric['torch_peak'],
                    'duration': metric['timestamp']
                })
        
        # 收集功耗指标
        if self.power_monitoring_enabled:
            while not self.power_metrics_queue.empty():
                power_metric = self.power_metrics_queue.get()
                if start_time <= power_metric['timestamp'] <= end_time:
                    window_metrics['power'].append(power_metric)
        
        return window_metrics
    
    def _aggregate_metrics(self, metrics_list: List[Dict], duration: float) -> Dict[str, float]:
        """聚合指标列表为单一值"""
        if not metrics_list or duration <= 0:
            return {}
        
        # 计算平均值
        aggregated = {}
        for key in metrics_list[0].keys():
            if key != 'duration' and isinstance(metrics_list[0][key], (int, float)):
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    aggregated[f"avg_{key}"] = sum(values) / len(values)
                    aggregated[f"peak_{key}"] = max(values)
                    aggregated[f"total_{key}"] = sum(values) * (duration / len(values)) if len(values) > 0 else 0
        
        # 添加统计信息
        aggregated['sample_count'] = len(metrics_list)
        aggregated['duration'] = duration
        
        return aggregated
    
    def _calculate_energy_consumption(self, power_metrics: List[Dict], duration: float, inference_time: float = 0.0):
        """计算能耗，区分推理时间和总时间"""
        if not power_metrics or duration <= 0:
            return PowerMetrics()
        
        # 计算平均功耗
        avg_cpu_power = sum(m['cpu_power'] for m in power_metrics) / len(power_metrics)
        avg_gpu_power = sum(m['gpu_power'] for m in power_metrics) / len(power_metrics)
        avg_dram_power = sum(m['dram_power'] for m in power_metrics) / len(power_metrics)
        
        # 计算能耗：功率 × 时间
        cpu_time = duration - inference_time if inference_time > 0 else duration
        gpu_time = inference_time if inference_time > 0 else duration
        
        cpu_energy = avg_cpu_power * cpu_time
        gpu_energy = avg_gpu_power * gpu_time
        dram_energy = avg_dram_power * duration
        
        total_energy = cpu_energy + gpu_energy + dram_energy
        avg_power = total_energy / duration if duration > 0 else 0
        
        # 计算能量占比
        gpu_fraction = gpu_energy / total_energy if total_energy > 0 else 0
        cpu_fraction = cpu_energy / total_energy if total_energy > 0 else 0
        dram_fraction = dram_energy / total_energy if total_energy > 0 else 0
        
        return {
            'total_energy': total_energy,
            'avg_power': avg_power,
            'peak_power': max(m['total_power'] for m in power_metrics) if power_metrics else 0,
            'gpu_energy': gpu_energy,
            'cpu_energy': cpu_energy,
            'dram_energy': dram_energy,
            'gpu_fraction': gpu_fraction,
            'cpu_fraction': cpu_fraction,
            'dram_fraction': dram_fraction
        }
    
    def record_agent_communication(
        self, 
        agent_name: str, 
        role: str, 
        step_idx: int,
        batch_size: int,
        input_data: Any = None,
        output_data: Any = None,
        latent_vectors: Optional[torch.Tensor] = None,
        kv_cache_size: float = 0.0,
        inference_time: float = 0.0,
        tokens_processed: int = 0,
        samples_processed: int = 1,
        total_time: float = None
    ):
        """记录Agent通信的基础设施指标 - 从queue中读取指标"""
        if not self.is_monitoring:
            print("Warning: Monitoring not started. Call start_monitoring() first.")
            return None
        
        # 确定时间窗口
        current_time = time.time()
        window_start = self.last_metric_collection_time
        window_end = current_time
        duration = window_end - window_start
        
        if total_time is None:
            total_time = duration
        
        # 1. 从队列中收集指标
        window_metrics = self._collect_metrics_for_window(window_start, window_end)
        
        # 2. 聚合常规指标
        compute_agg = self._aggregate_metrics(window_metrics['compute'], duration)
        memory_agg = self._aggregate_metrics(window_metrics['memory'], duration)
        
        # 3. 获取瞬时指标（网络、存储等）
        current_network = self._get_network_stats()
        torch_mem = self._get_torch_memory_stats()
        
        # 4. 计算网络指标
        network_delta = {
            'bytes_sent': current_network['bytes_sent'] - self.network_baseline['bytes_sent'],
            'bytes_recv': current_network['bytes_recv'] - self.network_baseline['bytes_recv'],
            'packets_sent': current_network['packets_sent'] - self.network_baseline['packets_sent'],
            'packets_recv': current_network['packets_recv'] - self.network_baseline['packets_recv']
        }
        
        # 5. 估计通信数据量
        comm_data_size = 0.0  # MB
        if self.method_name == "latent_mas" and latent_vectors is not None:
            latent_size = latent_vectors.numel() * latent_vectors.element_size() / (1024**2)  # MB
            comm_data_size = latent_size
        elif (self.method_name == "text_mas" or self.method_name == "baseline") and output_data:
            if isinstance(output_data, str):
                comm_data_size = len(output_data.encode('utf-8')) / (1024**2)  # MB
            elif isinstance(output_data, list) and output_data:
                text_size = sum(len(str(item).encode('utf-8')) for item in output_data) / (1024**2)
                comm_data_size = text_size
        
        # 6. 估计存储指标
        input_size = 0.0
        output_size = 0.0
        if input_data:
            input_size = self._estimate_data_size(input_data) / (1024**2)  # MB
        if output_data:
            output_size = self._estimate_data_size(output_data) / (1024**2)  # MB
        
        # 7. 估计FLOPS
        seq_length = tokens_processed // batch_size if batch_size > 0 else 0
        flops_estimate = (2 * self.model_params * seq_length * batch_size) / 1e9  # GFLOPS
        if inference_time > 0:
            flops_estimate = flops_estimate / inference_time
        
        # 8. 创建指标对象
        metrics = AgentCommunicationMetrics(
            method=self.method_name,
            agent_name=agent_name,
            role=role,
            step_idx=step_idx,
            batch_size=batch_size,
            duration=duration,
            experiment_id=self.current_experiment_id
        )
        
        # 9. 填充计算指标
        metrics.compute = ComputeMetrics(
            gpu_util=compute_agg.get('avg_gpu_util', 0.0),
            gpu_mem_used=memory_agg.get('avg_gpu_mem_used', 0.0),
            gpu_mem_total=sum(self.gpu_total_mems) if self.gpu_total_mems else 0.0,
            cpu_util=compute_agg.get('avg_cpu_util', 0.0),
            cpu_mem_used=memory_agg.get('avg_cpu_mem_used', 0.0),
            flops_estimate=flops_estimate,
            inference_time=inference_time,
            total_time=total_time,
            tokens_per_second=tokens_processed / inference_time if inference_time > 0 else 0
        )
        
        # 10. 填充内存指标
        metrics.memory = MemoryMetrics(
            vram_allocated=memory_agg.get('avg_torch_allocated', torch_mem['allocated']),
            vram_reserved=memory_agg.get('avg_torch_reserved', torch_mem['reserved']),
            vram_peak=memory_agg.get('peak_torch_peak', torch_mem['peak']),
            ram_used=memory_agg.get('avg_cpu_mem_used', 0.0),
            kv_cache_size=kv_cache_size,
            model_size=self.model_params * 4 / (1024**3),  # 假设FP32，4 bytes per param
            latent_vector_size=comm_data_size if self.method_name == "latent_mas" else 0
        )
        
        # 11. 填充网络指标
        metrics.network = NetworkMetrics(
            comm_data_size=comm_data_size,
            comm_latency=inference_time * 1000 if inference_time > 0 else 0,  # ms
            bandwidth_usage=comm_data_size / inference_time if inference_time > 0 else 0,
            packets_sent=network_delta['packets_sent'],
            packets_received=network_delta['packets_recv'],
            agent_comm_count=1
        )
        
        # 12. 填充存储指标
        metrics.storage = StorageMetrics(
            input_data_size=input_size,
            output_data_size=output_size,
            intermediate_data_size=comm_data_size,
            log_data_size=(input_size + output_size + comm_data_size) * 0.1,  # 估计日志大小
            checkpoint_size=0.0,
            io_throughput=(input_size + output_size) / inference_time if inference_time > 0 else 0
        )
        
        # 13. 填充功耗指标
        if self.power_monitoring_enabled and window_metrics['power']:
            energy_data = self._calculate_energy_consumption(window_metrics['power'], duration, inference_time)
            
            tokens_per_joule = tokens_processed / energy_data['total_energy'] if energy_data['total_energy'] > 0 else 0
            samples_per_joule = samples_processed / energy_data['total_energy'] if energy_data['total_energy'] > 0 else 0
            
            metrics.power = PowerMetrics(
                total_energy_consumed=energy_data['total_energy'],
                avg_power_draw=energy_data['avg_power'],
                peak_power_draw=energy_data['peak_power'],
                gpu_energy_fraction=energy_data['gpu_fraction'],
                cpu_energy_fraction=energy_data['cpu_fraction'],
                dram_energy_fraction=energy_data['dram_fraction'],
                avg_tokens_per_joule=tokens_per_joule,
                avg_samples_per_joule=samples_per_joule,
                measurement_duration=duration
            )
        
        # 14. 存储指标
        self.agent_metrics[agent_name][step_idx].append(metrics)
        self.overall_metrics['compute'].append(asdict(metrics.compute))
        self.overall_metrics['memory'].append(asdict(metrics.memory))
        self.overall_metrics['network'].append(asdict(metrics.network))
        self.overall_metrics['storage'].append(asdict(metrics.storage))
        if metrics.power:
            self.overall_metrics['power'].append(asdict(metrics.power))
        
        # 15. 更新最后收集时间
        self.last_metric_collection_time = current_time
        
        return metrics
    
    def _estimate_data_size(self, data: Any) -> float:
        """估算数据大小（bytes）"""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_data_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._estimate_data_size(k) + self._estimate_data_size(v) for k, v in data.items())
        elif data is None:
            return 0
        else:
            try:
                import pickle
                return len(pickle.dumps(data))
            except:
                return 1024  # 默认1KB
    
    def save_metrics(self, output_dir: str, experiment_name: str = None):
        """保存监控指标，自动重置状态"""
        if not self.current_experiment_id:
            print("Warning: No active experiment. Start monitoring first.")
            return None
        
        experiment_name = experiment_name or self.current_experiment_id
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存Agent级别指标
        agent_metrics_data = {}
        for agent_name, steps in self.agent_metrics.items():
            agent_metrics_data[agent_name] = {}
            for step_idx, metrics_list in steps.items():
                agent_metrics_data[agent_name][f"step_{step_idx}"] = [
                    asdict(metric) for metric in metrics_list
                ]
        
        # 保存整体指标
        overall_data = {
            'experiment_name': experiment_name,
            'experiment_id': self.current_experiment_id,
            'method': self.method_name,
            'model': getattr(self.args, 'model_name', 'unknown'),
            'timestamp': time.time(),
            'duration': time.time() - self.experiment_start_time,
            'agent_metrics': agent_metrics_data,
            'overall_metrics': copy.deepcopy(self.overall_metrics)  # 深拷贝避免引用问题
        }
        
        filename = f"{experiment_name}_{self.method_name}_{int(time.time())}.json"
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(convert_numpy_types(overall_data), f, indent=2)
            print(f"Infrastructure metrics saved to {filepath}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
            # 尝试保存到临时文件
            temp_filename = f"temp_{filename}"
            temp_filepath = os.path.join(output_dir, temp_filename)
            with open(temp_filepath, 'w') as f:
                json.dump(convert_numpy_types(overall_data), f, indent=2, default=str)
            print(f"Fallback: Metrics saved to {temp_filepath}")
            filepath = temp_filepath
        
        # 保存功耗指标
        if self.power_monitoring_enabled:
            power_summary = self.get_power_summary()
            power_data = {
                'experiment_name': experiment_name,
                'experiment_id': self.current_experiment_id,
                'method': self.method_name,
                'power_summary': power_summary,
                'raw_power_data': list(self.power_buffer)  # 保存原始功耗数据
            }
            
            power_filename = f"{experiment_name}_{self.method_name}_{int(time.time())}_power.json"
            power_filepath = os.path.join(output_dir, power_filename)
            
            try:
                with open(power_filepath, 'w') as f:
                    json.dump(convert_numpy_types(power_data), f, indent=2)
                print(f"Power metrics saved to {power_filepath}")
            except Exception as e:
                print(f"Error saving power metrics: {e}")
        
        # 重要：重置状态为下一次实验准备
        self._reset_for_next_experiment()
        
        return filepath
    
    def _reset_for_next_experiment(self):
        """重置监控器状态，为下一次实验准备"""
        self.agent_metrics = defaultdict(lambda: defaultdict(list))
        self.overall_metrics = {
            'compute': [],
            'memory': [],
            'network': [],
            'storage': [],
            'power': []
        }
        self.metrics_buffer.clear()
        self.power_buffer.clear()
        self.last_metric_collection_time = time.time()
        print("Monitor state reset for next experiment")
    
    def get_power_summary(self) -> Dict[str, float]:
        """获取功耗摘要信息"""
        if not self.power_monitoring_enabled or not self.overall_metrics['power']:
            return {}
        
        power_metrics = self.overall_metrics['power']
        total_energy = sum(pm['total_energy_consumed'] for pm in power_metrics)
        total_duration = sum(pm['measurement_duration'] for pm in power_metrics)
        
        if total_duration <= 0:
            return {}
        
        avg_power = total_energy / total_duration
        peak_power = max(pm['peak_power_draw'] for pm in power_metrics) if power_metrics else 0
        
        # 加权平均计算能量占比
        gpu_energy_total = sum(pm['total_energy_consumed'] * pm['gpu_energy_fraction'] for pm in power_metrics)
        cpu_energy_total = sum(pm['total_energy_consumed'] * pm['cpu_energy_fraction'] for pm in power_metrics)
        dram_energy_total = sum(pm['total_energy_consumed'] * pm['dram_energy_fraction'] for pm in power_metrics)
        
        gpu_fraction = gpu_energy_total / total_energy if total_energy > 0 else 0
        cpu_fraction = cpu_energy_total / total_energy if total_energy > 0 else 0
        dram_fraction = dram_energy_total / total_energy if total_energy > 0 else 0
        
        # 加权平均 tokens/samples per joule
        tokens_per_joule = sum(pm['avg_tokens_per_joule'] * pm['total_energy_consumed'] for pm in power_metrics) / total_energy if total_energy > 0 else 0
        samples_per_joule = sum(pm['avg_samples_per_joule'] * pm['total_energy_consumed'] for pm in power_metrics) / total_energy if total_energy > 0 else 0
        
        return {
            'total_energy_consumed': total_energy,
            'avg_power_draw': avg_power,
            'peak_power_draw': peak_power,
            'gpu_energy_fraction': gpu_fraction,
            'cpu_energy_fraction': cpu_fraction,
            'dram_energy_fraction': dram_fraction,
            'avg_tokens_per_joule': tokens_per_joule,
            'avg_samples_per_joule': samples_per_joule,
            'measurement_duration': total_duration
        }
    
    def get_summary_statistics(self) -> Dict[str, Dict]:
        """获取摘要统计信息"""
        summary = {
            'compute': {},
            'memory': {},
            'network': {},
            'storage': {},
            'power': {}
        }
        
        for dimension in ['compute', 'memory', 'network', 'storage', 'power']:
            if self.overall_metrics[dimension]:
                try:
                    import pandas as pd
                    df = pd.DataFrame(self.overall_metrics[dimension])
                    
                    summary_stats = {}
                    for column in df.columns:
                        if pd.api.types.is_numeric_dtype(df[column]):
                            summary_stats[column] = {
                                'mean': float(df[column].mean()),
                                'std': float(df[column].std()),
                                'min': float(df[column].min()),
                                'max': float(df[column].max()),
                                'count': int(len(df))
                            }
                    
                    summary[dimension] = summary_stats
                except ImportError:
                    print("pandas not available, skipping detailed statistics")
                    # 简单统计
                    for metric in self.overall_metrics[dimension]:
                        for key, value in metric.items():
                            if isinstance(value, (int, float)):
                                if key not in summary[dimension]:
                                    summary[dimension][key] = []
                                summary[dimension][key].append(value)
                    
                    # 计算简单平均值
                    for key, values in summary[dimension].items():
                        summary[dimension][key] = {
                            'mean': sum(values) / len(values) if values else 0,
                            'count': len(values)
                        }
        
        return convert_numpy_types(summary)