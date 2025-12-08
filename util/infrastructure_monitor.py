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
from collections import defaultdict

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
    avg_power_draw: float = 0.0         # Watts/s
    peak_power_draw: float = 0.0        # Watts
    gpu_energy_fraction: float = 0.0    # 0-1
    cpu_energy_fraction: float = 0.0    # 0-1
    dram_energy_fraction: float = 0.0 # 0-1
    avg_tokens_per_watt: float = 0.0   # tokens/J
    avg_samples_per_watt: float = 0.0  # samples/J
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
    timestamp: float = 0.0
    
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
        self.method_name = method_name
        self.metrics_queue = queue.Queue()
        self.power_metrics_queue = queue.Queue()  # 专门用于功耗监控
        self.is_monitoring = False
        self.monitor_thread = None
        self.power_monitor_thread = None
        self.start_time = 0.0
        self.network_baseline = self._get_network_stats()
        
        # 为每个agent和step存储指标
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
        self.gpu_handles = []  # 支持多GPU
        self.gpu_names = []
        self.gpu_total_mems = []
        self._init_gpu_monitoring()
        
        # 功耗监控状态
        self.power_monitoring_enabled = hasattr(args, 'enable_power_monitoring') and args.enable_power_monitoring
        self.total_energy_consumed = 0.0
        self.last_power_readings = {}
        self.power_start_time = 0.0
        self.power_base = self._get_power_metrics()
    
    def _init_gpu_monitoring(self):
        """初始化GPU监控 - 支持多GPU"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
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
    
    def _get_gpu_metrics(self, gpu_selected_ids: list = [0]) -> Dict[str, float]:
        """获取GPU指标 - 支持多GPU聚合"""
        if not self.gpu_handles:
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
                try:
                    handle = self.gpu_handles[i]
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    total_util += util.gpu
                    total_mem_used += mem.used / (1024**3)  # GB
                    total_mem_total += mem.total / (1024**3)  # GB
                except Exception as e:
                    print(f"Warning: Failed to get metrics for GPU {i}: {e}")
            
            avg_util = total_util / len(self.gpu_handles) if self.gpu_handles else 0.0
            
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
            # 从模型名称估计参数量
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
                self.model_params = 7e9  # 默认
    
    def _get_cpu_metrics(self) -> Dict[str, float]:
        """获取CPU和内存指标"""
        return {
            'cpu_util': psutil.cpu_percent(),
            'cpu_mem_used': psutil.virtual_memory().used / (1024**3),  # GB
            'cpu_mem_total': psutil.virtual_memory().total / (1024**3)  # GB
        }
    
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
    
    def _get_cpu_power(self, label):
        """获取CPU功耗基准值"""
        try:
            # 尝试使用pyRAPL如果可用
            import pyRAPL
            pyRAPL.setup()
            measure = pyRAPL.Measurement(label)
            measure.begin()
            print(f"power monitor: ", label)
            measure.end()
            cpu_package_power = np.array(measure.result.pkg) / measure.result.duration # the pkg measure the power of each CPU package in micro Jules within the called duration in microseconds;
            dram_package_power = np.array(measure.result.dram) / measure.result.duration
            return cpu_package_power, dram_package_power
        except ImportError:
            # 估算CPU功耗
            print("fail to monitor the CPU power consumption in watts")
    
    def _get_gpu_power_metrics(self, gpu_ids: list = [0]) -> Dict[int, Dict]:
        """获取所有GPU的功耗指标"""
        power_metrics = {}
        for i in gpu_ids:
            try:
                handle = self.gpu_handles[i]
                # 获取当前功耗
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                # 计算能耗（需要时间积分）
                current_time = time.time()
                if i not in self.last_power_readings:
                    self.last_power_readings[i] = {'power': power_draw, 'time': current_time}
                
                power_metrics[i] = {
                    'gpu_power_draw': power_draw,
                    'timestamp': current_time
                }
                
            except Exception as e:
                print(f"Warning: Failed to get power metrics for GPU {i}: {e}")
        
        return power_metrics
    
    def _get_power_metrics(self) -> Dict[str, float]:
        cpu_package_power, dram_package_power = self._get_cpu_power("running")
        cpu_power = cpu_package_power.sum()
        dram_power = dram_package_power.sum()
        gpu_power = 0.0
        gpu_nums = 0.0
        gpu_power_metrics = self._get_gpu_power_metrics()
        for _, power in gpu_power_metrics.items():
            gpu_nums += 1
            gpu_power += power
        if gpu_nums > 0:
            gpu_power /= gpu_nums
        if self.power_base is not None:
            cpu_power = max(cpu_power - self.power_base.cpu_power, 0)
            dram_power = max(dram_power - self.power_base.dram_power)
            gpu_power = max(gpu_power - self.power_base.gpu_power, 0)
        
        
        return {
            'cpu_power': cpu_power,
            'dram_power': dram_power,
            'gpu_power': gpu_power,
            'timestamp': time.time()
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.is_monitoring = True
        self.start_time = time.time()
        self.power_start_time = self.start_time
        self.network_baseline = self._get_network_stats()
        self.power_base = self._get_power_metrics()
        
        # 基础监控线程
        def monitoring_loop():
            while self.is_monitoring:
                gpu_metrics = self._get_gpu_metrics()
                cpu_metrics = self._get_cpu_metrics()
                torch_mem = self._get_torch_memory_stats()
                
                metrics = {
                    'timestamp': time.time() - self.start_time,
                    'gpu_util': gpu_metrics['gpu_util'],
                    'gpu_mem_used': gpu_metrics['gpu_mem_used'],
                    'cpu_util': cpu_metrics['cpu_util'],
                    'cpu_mem_used': cpu_metrics['cpu_mem_used'],
                    'torch_allocated': torch_mem['allocated'],
                    'torch_reserved': torch_mem['reserved'],
                    'torch_peak': torch_mem['peak']
                }
                
                self.metrics_queue.put(metrics)
                time.sleep(1.0)  # 1s采样间隔
        
        # 功耗监控线程（如果启用）
        def power_monitoring_loop():
            while self.is_monitoring and self.power_monitoring_enabled:
                power_metrics = self._get_power_metrics()
                self.power_metrics_queue.put(power_metrics)
                time.sleep(1.0)  # 1秒采样间隔，功耗变化较慢
        
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.power_monitoring_enabled:
            self.power_monitor_thread = threading.Thread(target=power_monitoring_loop, daemon=True)
            self.power_monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        if self.power_monitor_thread:
            self.power_monitor_thread.join(timeout=1.0)
        
        # 清理GPU监控
        try:
            if hasattr(pynvml, 'nvmlShutdown'):
                pynvml.nvmlShutdown()
        except:
            pass
    
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
        samples_processed: int = 1
    ):
        """记录Agent通信的基础设施指标 - 独立实现，不依赖父类"""
        timestamp = time.time() - self.start_time
        
        # 计算指标
        gpu_metrics = self._get_gpu_metrics()
        cpu_metrics = self._get_cpu_metrics()
        torch_mem = self._get_torch_memory_stats()
        current_network = self._get_network_stats()
        
        # 估计FLOPS
        seq_length = tokens_processed // batch_size if batch_size > 0 else 0
        flops_estimate = (2 * self.model_params * seq_length * batch_size) / 1e9  # GFLOPS
        if inference_time > 0:
            flops_estimate = flops_estimate / inference_time
        
        # 计算网络指标
        network_delta = {
            'bytes_sent': current_network['bytes_sent'] - self.network_baseline['bytes_sent'],
            'bytes_recv': current_network['bytes_recv'] - self.network_baseline['bytes_recv'],
            'packets_sent': current_network['packets_sent'] - self.network_baseline['packets_sent'],
            'packets_recv': current_network['packets_recv'] - self.network_baseline['packets_recv']
        }
        
        # 估计通信数据量
        comm_data_size = 0.0  # MB
        if self.method_name == "latent_mas" and latent_vectors is not None:
            # Latent space: 计算latent vector大小
            latent_size = latent_vectors.numel() * latent_vectors.element_size() / (1024**2)  # MB
            comm_data_size = latent_size
        elif (self.method_name == "text_mas" or self.method_name == "baseline") and output_data is not None:
            # Text space: 估算文本大小
            if isinstance(output_data, str):
                comm_data_size = len(output_data.encode('utf-8')) / (1024**2)  # MB
            elif isinstance(output_data, list) and output_data:
                text_size = sum(len(str(item).encode('utf-8')) for item in output_data) / (1024**2)
                comm_data_size = text_size
        
        # 估计存储指标
        input_size = 0.0
        output_size = 0.0
        if input_data is not None:
            input_size = self._estimate_data_size(input_data) / (1024**2)  # MB
        if output_data is not None:
            output_size = self._estimate_data_size(output_data) / (1024**2)  # MB
        
        # 创建指标对象
        metrics = AgentCommunicationMetrics(
            method=self.method_name,
            agent_name=agent_name,
            role=role,
            step_idx=step_idx,
            batch_size=batch_size,
            timestamp=timestamp
        )
        
        # 填充各维度指标
        metrics.compute = ComputeMetrics(
            gpu_util=gpu_metrics['gpu_util'],
            gpu_mem_used=gpu_metrics['gpu_mem_used'],
            gpu_mem_total=gpu_metrics['gpu_mem_total'],
            cpu_util=cpu_metrics['cpu_util'],
            cpu_mem_used=cpu_metrics['cpu_mem_used'],
            flops_estimate=flops_estimate,
            inference_time=inference_time,
            tokens_per_second=tokens_processed / inference_time if inference_time > 0 else 0
        )
        
        metrics.memory = MemoryMetrics(
            vram_allocated=torch_mem['allocated'],
            vram_reserved=torch_mem['reserved'],
            vram_peak=torch_mem['peak'],
            ram_used=cpu_metrics['cpu_mem_used'],
            kv_cache_size=kv_cache_size,
            model_size=self.model_params * 4 / (1024**3),  # 假设FP32，4 bytes per param
            latent_vector_size=comm_data_size if self.method_name == "latent_mas" else 0
        )
        
        metrics.network = NetworkMetrics(
            comm_data_size=comm_data_size,
            comm_latency=inference_time * 1000,  # ms
            bandwidth_usage=comm_data_size / inference_time if inference_time > 0 else 0,
            packets_sent=network_delta['packets_sent'],
            packets_received=network_delta['packets_recv'],
            agent_comm_count=1
        )
        
        metrics.storage = StorageMetrics(
            input_data_size=input_size,
            output_data_size=output_size,
            intermediate_data_size=comm_data_size,
            log_data_size=(input_size + output_size + comm_data_size) * 0.1,  # 估计日志大小
            checkpoint_size=0.0,
            io_throughput=(input_size + output_size) / inference_time if inference_time > 0 else 0
        )
        
        # 功耗指标
        if self.power_monitoring_enabled:
            # 获取最新的功耗数据
            power_metrics = []
            while not self.power_metrics_queue.empty():
                power_metrics.append(self.power_metrics_queue.get())
            
            if power_metrics:
                latest_power = power_metrics[-1]
                total_cpu_power = sum(power['cpu_power'] for power in power_metrics)
                total_gpu_power = sum(power['gpu_power'] for power in power_metrics)
                total_dram_power = sum(power['dram_power'] for power in power_metrics)
                total_energy = sum(power['cpu_power'] + power['dram_power'] + power['gpu_power'] for power in power_metrics)
                avg_power = total_energy / len(power_metrics)
                
                tokens_per_watt = tokens_processed / total_energy if total_energy > 0 else 0
                samples_per_watt = samples_processed / total_energy if total_energy > 0 else 0
                
                metrics.power = PowerMetrics(
                    total_energy_consumed=total_energy,
                    avg_power_draw=avg_power,
                    peak_power_draw=max(power['cpu_power'] + power['dram_power'] + power['gpu_power'] for power in power_metrics) if power_metrics else 0,
                    gpu_energy_fraction=total_gpu_power / total_energy,
                    cpu_energy_fraction=total_cpu_power / total_energy,
                    dram_energy_fraction=total_dram_power / total_energy,
                    avg_tokens_per_watt=tokens_per_watt,
                    avg_samples_per_watt=samples_per_watt,
                    measurement_duration=len(power_metrics)
                )
        
        # 存储指标
        self.agent_metrics[agent_name][step_idx].append(metrics)
        self.overall_metrics['compute'].append(asdict(metrics.compute))
        self.overall_metrics['memory'].append(asdict(metrics.memory))
        self.overall_metrics['network'].append(asdict(metrics.network))
        self.overall_metrics['storage'].append(asdict(metrics.storage))
        if metrics.power:
            self.overall_metrics['power'].append(asdict(metrics.power))
        
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
            # 使用pickle估算其他类型
            try:
                import pickle
                return len(pickle.dumps(data))
            except:
                return 1024  # 默认1KB
    
    def save_metrics(self, output_dir: str, experiment_name: str):
        """保存监控指标"""
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
            'method': self.method_name,
            'model': getattr(self.args, 'model_name', 'unknown'),
            'timestamp': time.time(),
            'agent_metrics': agent_metrics_data,
            'overall_metrics': self.overall_metrics
        }
        
        filename = f"{experiment_name}_{self.method_name}_{int(time.time())}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(overall_data, f, indent=2)
        
        print(f"Infrastructure metrics saved to {filepath}")
        
        # 保存功耗指标（如果启用）
        if self.power_monitoring_enabled:
            power_data = {
                'experiment_name': experiment_name,
                'method': self.method_name,
                'power_metrics': [],
                'power_summary': self.get_power_summary()
            }
            
            while not self.power_metrics_queue.empty():
                power_data['power_metrics'].append(self.power_metrics_queue.get())
            
            power_filename = f"{experiment_name}_{self.method_name}_{int(time.time())}_power.json"
            power_filepath = os.path.join(output_dir, power_filename)
            
            with open(power_filepath, 'w') as f:
                json.dump(power_data, f, indent=2)
            
            print(f"Power metrics saved to {power_filepath}")
        
        return filepath
    
    def get_power_summary(self) -> Dict[str, float]:
        """获取功耗摘要信息"""
        if not self.power_monitoring_enabled or not self.overall_metrics['power']:
            return {}
        
        power_metrics = self.overall_metrics['power']
        
        total_energy = sum(m['total_energy_consumed'] for m in power_metrics)
        avg_power = sum(m['avg_power_draw'] for m in power_metrics) / len(power_metrics) if power_metrics else 0
        peak_power = max(m['peak_power_draw'] for m in power_metrics) if power_metrics else 0
        
        gpu_energy_fraction = sum(m['gpu_energy_fraction'] for m in power_metrics) / len(power_metrics) if power_metrics else 0
        cpu_energy_fraction = sum(m['cpu_energy_fraction'] for m in power_metrics) / len(power_metrics) if power_metrics else 0
        dram_energy_fraction = sum(m['dram_energy_fraction'] for m in power_metrics) / len(power_metrics) if power_metrics else 0
        
        tokens_per_watt = sum(m['avg_tokens_per_watt'] for m in power_metrics) / len(power_metrics) if power_metrics else 0
        samples_per_watt = sum(m['avg_samples_per_watt'] for m in power_metrics) / len(power_metrics) if power_metrics else 0
        
        return {
            'total_energy_consumed': total_energy,
            'avg_power_draw': avg_power,
            'peak_power_draw': peak_power,
            'gpu_energy_fraction': gpu_energy_fraction,
            'cpu_energy_fraction': cpu_energy_fraction,
            'dram_energy_fraction': dram_energy_fraction,
            'avg_tokens_per_watt': tokens_per_watt,
            'avg_samples_per_watt': samples_per_watt
        }
    
    def get_summary_statistics(self) -> Dict[str, Dict]:
        """获取摘要统计信息 - 修复NumPy序列化问题"""
        summary = {
            'compute': {},
            'memory': {},
            'network': {},
            'storage': {},
            'power': {}
        }
        
        # 计算每个维度的统计摘要
        for dimension in ['compute', 'memory', 'network', 'storage', 'power']:
            if self.overall_metrics[dimension]:
                # 转换为pandas DataFrame以便计算统计量
                import pandas as pd
                df = pd.DataFrame(self.overall_metrics[dimension])
                
                summary_stats = {}
                for column in df.columns:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        # 确保转换为Python原生类型
                        summary_stats[column] = {
                            'mean': float(df[column].mean()),
                            'std': float(df[column].std()),
                            'min': float(df[column].min()),
                            'max': float(df[column].max()),
                            'count': int(len(df))
                        }
                
                summary[dimension] = summary_stats
        
        # 使用转换函数确保可序列化
        return convert_numpy_types(summary)