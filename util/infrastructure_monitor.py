import time
import torch
import psutil
import numpy as np
import pynvml
import socket
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
from collections import defaultdict

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
    """功耗监控数据结构"""
    gpu_power_draw: float = 0.0          # 当前GPU功耗 (W)
    gpu_power_limit: float = 0.0         # GPU功耗限制 (W)
    gpu_energy_consumed: float = 0.0     # GPU累计能耗 (J)
    cpu_power_draw: float = 0.0          # CPU功耗估计 (W)
    cpu_energy_consumed: float = 0.0     # CPU累计能耗 (J)
    system_power_draw: float = 0.0       # 系统总功耗 (W)
    system_energy_consumed: float = 0.0  # 系统累计能耗 (J)
    tokens_per_joule: float = 0.0        # 每焦耳处理的token数
    samples_per_joule: float = 0.0       # 每焦耳处理的样本数
    timestamp: float = 0.0               # 时间戳

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
        self.power_metrics = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = 0.0
        self.network_baseline = self._get_network_stats()
        self._init_gpu_monitoring()

        # 功耗性能监控
        self.gpu_power_baseline = {}
        self.cpu_power_baseline = self._get_cpu_power_baseline()
        self.total_gpu_energy = defaultdict(float)  # 每个GPU的累计能耗
        self.total_cpu_energy = 0.0
        self.last_power_readings = {}
        
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
        self.tokens_processed_total = 0
        self.samples_processed_total = 0
        self._estimate_model_flops()
        print(f"InfrastructureMonitor initialized for method: {self.method_name}")
    
    def _init_gpu_monitoring(self):
        """初始化GPU监控"""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = []
            self.gpu_names = []
            self.gpu_total_mem = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)
                self.gpu_names.append(pynvml.nvmlDeviceGetName(handle).decode())
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_total_mem.append(mem_info.total / (1024**3))  # GB
                
                # 获取GPU功耗限制
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # mW to W
                print(f"GPU {i} ({self.gpu_names[i]}): Power limit = {power_limit:.1f}W")
            
            self.device_count = device_count
            print(f"Monitoring {device_count} GPU(s) for power consumption")
            
        except Exception as e:
            print(f"Warning: GPU monitoring initialization failed: {e}")
            self.gpu_handles = []
            self.device_count = 0
    
    def _get_cpu_power_baseline(self):
        """获取CPU功耗基准值"""
        try:
            # 尝试使用pyRAPL如果可用
            import pyRAPL
            pyRAPL.setup()
            return pyRAPL.get_power_info().avg_power
        except ImportError:
            # 估算CPU功耗
            cpu_count = psutil.cpu_count()
            return cpu_count * 5.0  # 每核心5W的粗略估计
    
    def _get_gpu_power_metrics(self) -> Dict[int, Dict]:
        """获取所有GPU的功耗指标"""
        power_metrics = {}
        
        for i, handle in enumerate(self.gpu_handles):
            try:
                # 获取当前功耗
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                
                # 获取功耗限制
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # mW to W
                
                # 计算能耗（需要时间积分）
                current_time = time.time()
                if i not in self.last_power_readings:
                    self.last_power_readings[i] = {'power': power_draw, 'time': current_time}
                    energy_consumed = 0.0
                else:
                    last_reading = self.last_power_readings[i]
                    time_delta = current_time - last_reading['time']
                    # 使用梯形法则计算能耗
                    energy_consumed = (last_reading['power'] + power_draw) * time_delta / 2.0
                    self.total_gpu_energy[i] += energy_consumed
                    
                    self.last_power_readings[i] = {'power': power_draw, 'time': current_time}
                
                power_metrics[i] = {
                    'gpu_power_draw': power_draw,
                    'gpu_power_limit': power_limit,
                    'gpu_energy_consumed': self.total_gpu_energy[i],
                    'timestamp': current_time
                }
                
            except Exception as e:
                print(f"Warning: Failed to get power metrics for GPU {i}: {e}")
                power_metrics[i] = {
                    'gpu_power_draw': 0.0,
                    'gpu_power_limit': 0.0,
                    'gpu_energy_consumed': self.total_gpu_energy.get(i, 0.0),
                    'timestamp': time.time()
                }
        
        return power_metrics
    
    def _get_cpu_power_metrics(self) -> Dict:
        """获取CPU功耗指标"""
        try:
            import pyRAPL
            pyRAPL.setup()
            measurement = pyRAPL.get_power_info()
            current_power = measurement.avg_power
            
            current_time = time.time()
            if not hasattr(self, 'last_cpu_reading'):
                self.last_cpu_reading = {'power': current_power, 'time': current_time}
                energy_consumed = 0.0
            else:
                last_reading = self.last_cpu_reading
                time_delta = current_time - last_reading['time']
                energy_consumed = (last_reading['power'] + current_power) * time_delta / 2.0
                self.total_cpu_energy += energy_consumed
                
                self.last_cpu_reading = {'power': current_power, 'time': current_time}
            
            return {
                'cpu_power_draw': current_power,
                'cpu_energy_consumed': self.total_cpu_energy,
                'timestamp': current_time
            }
            
        except ImportError:
            # 估算CPU功耗
            cpu_util = psutil.cpu_percent()
            estimated_power = self.cpu_power_baseline * (cpu_util / 100.0)
            
            current_time = time.time()
            if not hasattr(self, 'last_cpu_reading'):
                self.last_cpu_reading = {'power': estimated_power, 'time': current_time}
                energy_consumed = 0.0
            else:
                last_reading = self.last_cpu_reading
                time_delta = current_time - last_reading['time']
                energy_consumed = (last_reading['power'] + estimated_power) * time_delta / 2.0
                self.total_cpu_energy += energy_consumed
                
                self.last_cpu_reading = {'power': estimated_power, 'time': current_time}
            
            return {
                'cpu_power_draw': estimated_power,
                'cpu_energy_consumed': self.total_cpu_energy,
                'timestamp': current_time
            }
    
    def _get_system_power_metrics(self, gpu_power_data: Dict, cpu_power_data: Dict) -> Dict:
        """计算系统总功耗"""
        total_gpu_power = sum(data['gpu_power_draw'] for data in gpu_power_data.values())
        total_gpu_energy = sum(data['gpu_energy_consumed'] for data in gpu_power_data.values())
        
        system_power = total_gpu_power + cpu_power_data['cpu_power_draw']
        system_energy = total_gpu_energy + cpu_power_data['cpu_energy_consumed']
        
        return {
            'system_power_draw': system_power,
            'system_energy_consumed': system_energy,
            'timestamp': time.time()
        }
    
    def _estimate_model_flops(self):
        """估计模型FLOPS"""
        # 这里需要根据实际模型参数来估计，暂时使用通用公式
        # FLOPS ≈ 2 * 参数量 * 序列长度 * batch_size
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
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """获取GPU指标"""
        metrics = {
            'gpu_util': 0.0,
            'gpu_mem_used': 0.0,
            'gpu_mem_total': self.gpu_total_mem if hasattr(self, 'gpu_total_mem') else 0.0
        }
        
        if self.gpu_handle is None:
            return metrics
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            
            metrics['gpu_util'] = util.gpu
            metrics['gpu_mem_used'] = mem.used / (1024**3)  # GB
            metrics['gpu_mem_total'] = mem.total / (1024**3)  # GB
        except Exception as e:
            print(f"Warning: Failed to get GPU metrics: {e}")
        
        return metrics
    
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
    
    def start_monitoring(self):
        self.is_monitoring = True
        self.start_time = time.time()
        self.network_baseline = self._get_network_stats()
        
        # 重置功耗计数器
        self.total_gpu_energy = defaultdict(float)
        self.total_cpu_energy = 0.0
        self.last_power_readings = {}
        self.tokens_processed_total = 0
        self.samples_processed_total = 0
        
        # 启动常规监控线程
        def monitoring_loop():
            while self.is_monitoring:
                gpu_metrics = self._get_gpu_metrics()
                cpu_metrics = self._get_cpu_metrics()
                torch_mem = self._get_torch_memory_stats()
                gpu_power_data = self._get_gpu_power_metrics()
                cpu_power_data = self._get_cpu_power_metrics()
                system_power_data = self._get_system_power_metrics(gpu_power_data, cpu_power_data)
                
                # 计算能效指标
                elapsed_time = time.time() - self.start_time
                tokens_per_joule = self.tokens_processed_total / system_power_data['system_energy_consumed'] if system_power_data['system_energy_consumed'] > 0 else 0
                samples_per_joule = self.samples_processed_total / system_power_data['system_energy_consumed'] if system_power_data['system_energy_consumed'] > 0 else 0
                
                metrics = {
                    'timestamp': elapsed_time,
                    'gpu_util': gpu_metrics['gpu_util'],
                    'gpu_mem_used': gpu_metrics['gpu_mem_used'],
                    'cpu_util': cpu_metrics['cpu_util'],
                    'cpu_mem_used': cpu_metrics['cpu_mem_used'],
                    'torch_allocated': torch_mem['allocated'],
                    'torch_reserved': torch_mem['reserved'],
                    'torch_peak': torch_mem['peak'],
                    'power_metrics': {
                        'gpu_power_data': gpu_power_data,
                        'cpu_power_data': cpu_power_data,
                        'system_power_data': system_power_data,
                        'tokens_per_joule': tokens_per_joule,
                        'samples_per_joule': samples_per_joule
                    }
                }
                
                self.metrics_queue.put(metrics)
                time.sleep(0.1)  # 100ms采样间隔
        
        # 启动功耗专用监控线程（更高频率）
        def power_monitoring_loop():
            while self.is_monitoring:
                gpu_power_data = self._get_gpu_power_metrics()
                cpu_power_data = self._get_cpu_power_metrics()
                system_power_data = self._get_system_power_metrics(gpu_power_data, cpu_power_data)
                
                power_metrics = {
                    'timestamp': time.time() - self.start_time,
                    'gpu_power_data': gpu_power_data,
                    'cpu_power_data': cpu_power_data,
                    'system_power_data': system_power_data
                }
                
                self.power_queue.put(power_metrics)
                time.sleep(0.05)  # 50ms采样间隔，更高精度
        
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.power_thread = threading.Thread(target=power_monitoring_loop, daemon=True)
        
        self.monitor_thread.start()
        self.power_thread.start()
        
        print("Infrastructure monitoring started (including power monitoring)")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        if self.power_thread:
            self.power_thread.join(timeout=1.0)
        print("Infrastructure monitoring stopped")
    
    def get_accumulated_metrics(self) -> Dict[str, List]:
        """获取累积的监控数据"""
        accumulated = []
        while not self.metrics_queue.empty():
            accumulated.append(self.metrics_queue.get())
        
        # 按类别组织
        result = {
            'compute': [],
            'memory': [],
            'network': [],
            'storage': []
        }
        
        for metrics in accumulated:
            result['compute'].append({
                'timestamp': metrics['timestamp'],
                'gpu_util': metrics['gpu_util'],
                'cpu_util': metrics['cpu_util']
            })
            result['memory'].append({
                'timestamp': metrics['timestamp'],
                'gpu_mem_used': metrics['gpu_mem_used'],
                'cpu_mem_used': metrics['cpu_mem_used'],
                'torch_allocated': metrics['torch_allocated'],
                'torch_reserved': metrics['torch_reserved'],
                'torch_peak': metrics['torch_peak']
            })
        
        return result
    
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
        samples_processed: int = 1  # 每次推理处理的样本数
    ):
        """记录Agent通信的基础设施指标，包括功耗"""
        timestamp = time.time() - self.start_time
        
        # 累计处理的token和样本数
        self.tokens_processed_total += tokens_processed
        self.samples_processed_total += samples_processed
        
        # 获取最新的功耗数据
        power_metrics = None
        while not self.power_queue.empty():
            power_metrics = self.power_queue.get()
        
        if power_metrics is None:
            # 如果没有新数据，使用最后记录的值
            power_metrics = {
                'gpu_power_data': {},
                'cpu_power_data': {},
                'system_power_data': {},
                'timestamp': timestamp
            }
        
        # 处理GPU功耗数据
        gpu_power_draw = 0.0
        gpu_energy_consumed = 0.0
        if power_metrics['gpu_power_data']:
            gpu_power_draw = sum(data['gpu_power_draw'] for data in power_metrics['gpu_power_data'].values())
            gpu_energy_consumed = sum(data['gpu_energy_consumed'] for data in power_metrics['gpu_power_data'].values())
        
        # 处理CPU功耗数据
        cpu_power_draw = power_metrics['cpu_power_data'].get('cpu_power_draw', 0.0)
        cpu_energy_consumed = power_metrics['cpu_power_data'].get('cpu_energy_consumed', 0.0)
        
        # 处理系统功耗数据
        system_power_draw = power_metrics['system_power_data'].get('system_power_draw', gpu_power_draw + cpu_power_draw)
        system_energy_consumed = power_metrics['system_power_data'].get('system_energy_consumed', gpu_energy_consumed + cpu_energy_consumed)
        
        # 计算能效指标
        tokens_per_joule = tokens_processed / system_energy_consumed if system_energy_consumed > 0 else 0
        samples_per_joule = samples_processed / system_energy_consumed if system_energy_consumed > 0 else 0
        
        # 创建功耗指标
        power_data = PowerMetrics(
            gpu_power_draw=gpu_power_draw,
            gpu_power_limit=sum(data['gpu_power_limit'] for data in power_metrics['gpu_power_data'].values()) if power_metrics['gpu_power_data'] else 0.0,
            gpu_energy_consumed=gpu_energy_consumed,
            cpu_power_draw=cpu_power_draw,
            cpu_energy_consumed=cpu_energy_consumed,
            system_power_draw=system_power_draw,
            system_energy_consumed=system_energy_consumed,
            tokens_per_joule=tokens_per_joule,
            samples_per_joule=samples_per_joule,
            timestamp=timestamp
        )
        
        # 调用父类方法记录其他指标
        metrics = super().record_agent_communication(
            agent_name, role, step_idx, batch_size, input_data, output_data, 
            latent_vectors, kv_cache_size, inference_time, tokens_processed
        )
        
        # 更新计算指标中的功耗数据
        if metrics.compute:
            metrics.compute.power_metrics = power_data
        
        # 保存功耗指标
        self.overall_metrics['power'].append(asdict(power_data))
        
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
    
    def get_power_summary(self) -> Dict:
        """获取功耗摘要统计"""
        if not self.overall_metrics['power']:
            return {}
        
        power_data = self.overall_metrics['power']
        
        # 计算统计摘要
        summary = {
            'total_energy_consumed': sum(entry['system_energy_consumed'] for entry in power_data),
            'avg_power_draw': np.mean([entry['system_power_draw'] for entry in power_data]),
            'peak_power_draw': max(entry['system_power_draw'] for entry in power_data),
            'avg_tokens_per_joule': np.mean([entry['tokens_per_joule'] for entry in power_data if entry['tokens_per_joule'] > 0]),
            'avg_samples_per_joule': np.mean([entry['samples_per_joule'] for entry in power_data if entry['samples_per_joule'] > 0]),
            'gpu_energy_fraction': sum(entry['gpu_energy_consumed'] for entry in power_data) / 
                                  sum(entry['system_energy_consumed'] for entry in power_data) if sum(entry['system_energy_consumed'] for entry in power_data) > 0 else 0,
            'measurement_count': len(power_data)
        }
        
        return summary
    
    def save_metrics(self, output_dir: str, experiment_name: str):
        """保存监控指标，包括功耗数据"""
        filepath = super().save_metrics(output_dir, experiment_name)
        
        # 保存功耗详细数据
        power_data_path = filepath.replace('.json', '_power.json')
        power_data = {
            'experiment_name': experiment_name,
            'method': self.method_name,
            'model': getattr(self.args, 'model_name', 'unknown'),
            'power_metrics': self.overall_metrics['power'],
            'power_summary': self.get_power_summary(),
            'timestamp': time.time()
        }
        
        with open(power_data_path, 'w') as f:
            json.dump(power_data, f, indent=2)
        
        print(f"Power metrics saved to {power_data_path}")
        return filepath
    
    def get_summary_statistics(self) -> Dict[str, Dict]:
        """获取摘要统计信息"""
        summary = {
            'compute': {},
            'memory': {},
            'network': {},
            'storage': {}
        }
        
        # 计算每个维度的统计摘要
        for dimension in ['compute', 'memory', 'network', 'storage']:
            if self.overall_metrics[dimension]:
                # 转换为numpy数组以便计算统计量
                metrics_array = np.array([list(metric.values()) for metric in self.overall_metrics[dimension]])
                
                if len(metrics_array) > 0 and len(metrics_array.shape) == 2:
                    means = np.mean(metrics_array, axis=0)
                    stds = np.std(metrics_array, axis=0)
                    
                    # 创建统计摘要
                    summary[dimension] = {
                        key: {
                            'mean': float(mean),
                            'std': float(std),
                            'min': float(np.min(metrics_array[:, i])),
                            'max': float(np.max(metrics_array[:, i])),
                            'count': int(len(metrics_array))
                        }
                        for i, key in enumerate(self.overall_metrics[dimension][0].keys())
                        for mean, std in [(means[i], stds[i])]
                    }
        
        return summary