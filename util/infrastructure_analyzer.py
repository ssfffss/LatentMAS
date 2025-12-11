from datetime import datetime
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import math
import os
import re
from dataclasses import asdict
import time

def convert_numpy_types(obj):
    """
    递归地将NumPy数据类型转换为Python原生类型，使其可JSON序列化
    
    Args:
        obj: 任何Python对象
        
    Returns:
        转换后的对象，所有NumPy类型都被转换为Python原生类型
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, set):
        return {convert_numpy_types(item) for item in obj}
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        # 处理自定义对象
        return convert_numpy_types(obj.__dict__)
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        # 对于其他类型，尝试转换为字符串
        try:
            return str(obj)
        except:
            return "unserializable_object"


class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持NumPy数据类型"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


class InfrastructureAnalyzer:
    def __init__(self, metrics_dir: str = "metrics", output_dir: str = "analysis_results"):
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载和聚合实验数据
        self.raw_data = self._load_all_metrics()
        self.aggregated_data = self._aggregate_experiments()
        
        print(f"✅ Loaded {len(self.raw_data)} raw experiments")
        print(f"✅ Aggregated into {len(self.aggregated_data)} unique experiments")
    
    def _extract_experiment_key(self, experiment_name: str, method: str) -> str:
        return experiment_name
    
    def _load_all_metrics(self) -> Dict[str, List]:
        """加载所有指标数据，包括功耗数据"""
        all_data = defaultdict(list)
        
        # 加载主指标文件
        for filepath in self.metrics_dir.glob("*.json"):
            if '_power.json' in str(filepath) or 'report' in str(filepath):
                continue
            filename, _ = os.path.splitext(os.path.basename(filepath))
            
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    method = data.get('method', 'unknown')
                    experiment_name = data.get('experiment_name', 'unknown')
                    if 'gsm8k' in experiment_name:
                        for agent_name, steps in data['agent_metrics'].items():
                            for step_idx, metrics in steps.items():
                                for i in len(metrics):
                                    data['agent_metrics'][agent_name][step_idx][i]['compute'] = data['agent_metrics'][agent_name][step_idx][i]['compute'] * 8
                                    # compute = metric.get('compute', {})
                                    # compute['gpu_util'] = compute['gpu_util'] * 8
                    
                    # # 生成唯一标识
                    # exp_id = f"{method}_{experiment_name}_{Path(filepath).stem.split('_')[-1]}"
                    # data['file_path'] = str(filepath)
                    # data['experiment_id'] = exp_id

                    power_pattern = f"{filename}_power.json"
                    matching_files = list(self.metrics_dir.glob(power_pattern))
                    if matching_files:
                        power_file = matching_files[0]  # 取第一个匹配文件
                    
                    try:
                        with open(power_file) as f:
                            power_data = json.load(f)
                            data['power_data'] = power_data
                    except Exception as e:
                        print(f"⚠️  Error loading power data for {power_pattern}: {e}")

                    all_data[method].append(data)
                    
                    
            except Exception as e:
                print(f"❌ Error loading {filepath}: {e}")
    
        return all_data
    
    def _aggregate_experiments(self) -> Dict[str, Dict]:
        """聚合相同实验的不同batch数据"""
        aggregated = {}
        
        # 按实验键分组
        experiment_groups = defaultdict(list)
        
        for method, experiments in self.raw_data.items():
            for exp in experiments:
                exp_key = self._extract_experiment_key(exp['experiment_name'], method)
                experiment_groups[exp_key].append(exp)
        
        print(f"📊 Found {len(experiment_groups)} experiment groups: {experiment_groups.keys()} keys to aggregate")
        
        for exp_key, experiments in experiment_groups.items():
            if not experiments:
                continue
            print(f"exp_key: {exp_key}, num_experiments: {len(experiments)}")
            
            # 基础元数据（取第一个实验的）
            base_exp = experiments[0]
            method = base_exp['method']
            
            # 聚合指标
            aggregated_metrics = {
                'compute': [],
                'memory': [],
                'network': [],
                'storage': [],
                'power': []
            }
            
            # 聚合Agent指标
            aggregated_agents = defaultdict(lambda: defaultdict(list))
            
            total_samples = 0
            total_duration = 0
            total_tokens = 0
            
            for exp in experiments:
                # 聚合整体指标
                for dimension in ['compute', 'memory', 'network', 'storage', 'power']:
                    if dimension in exp['overall_metrics']:
                        aggregated_metrics[dimension].extend(exp['overall_metrics'][dimension])
                
                # 聚合Agent指标
                for agent_name, steps in exp['agent_metrics'].items():
                    for step_idx, step_metrics in steps.items():
                        aggregated_agents[agent_name][step_idx].extend(step_metrics)
                
                # 累计统计信息
                total_samples += exp['agent_metrics']['Planner']['step_0'][0]['batch_size']  
                total_duration += exp.get('duration', 0)
                
                # 估计token数量
                if 'compute' in exp['overall_metrics']:
                    for metric in exp['overall_metrics']['compute']:
                        total_tokens += metric.get('tokens_per_second', 0) * metric.get('inference_time', 0)
            
            # 计算平均值和统计信息
            summary_stats = {}
            for dimension, metrics in aggregated_metrics.items():
                if metrics:
                    df = pd.DataFrame(metrics)
                    summary_stats[dimension] = {}
                    for col in df.select_dtypes(include=[np.number]).columns:
                        summary_stats[dimension][col] = {
                            'mean': df[col].mean(),
                            'std': df[col].std(),
                            'min': df[col].min(),
                            'max': df[col].max(),
                            'count': len(df)
                        }
            
            # 创建聚合后的实验数据
            aggregated[exp_key] = {
                'experiment_key': exp_key,
                'method': method,
                'model': base_exp.get('model', 'unknown'),
                'task': exp_key.split('_')[3] if len(exp_key.split('_')) > 3 else 'unknown',
                'batch_count': len(experiments),
                'total_samples': total_samples,
                'total_duration': total_duration,
                'total_tokens': total_tokens,
                'aggregated_metrics': aggregated_metrics,
                'agent_metrics': dict(aggregated_agents),
                'summary_stats': summary_stats,
                'power_summary': self._aggregate_power_data(experiments),
                'experiments': experiments  # 保留原始实验数据
            }
        
        return aggregated
    
    def _aggregate_power_data(self, experiments: List[Dict]) -> Dict[str, float]:
        """聚合多个实验的功耗数据"""
        if not experiments:
            return {}
        
        total_energy = 0.0
        total_duration = 0.0
        peak_power = 0.0
        gpu_energy = 0.0
        cpu_energy = 0.0
        dram_energy = 0.0
        total_tokens = 0
        total_samples = 0
        valid_count = 0
        
        for exp in experiments:
            if 'power_data' in exp and 'power_summary' in exp['power_data']:
                summary = exp['power_data']['power_summary']
                duration = summary.get('measurement_duration', 0)
                
                if duration > 0:
                    total_energy += summary.get('total_energy_consumed', 0)
                    total_duration += duration
                    peak_power = max(peak_power, summary.get('peak_power_draw', 0))
                    
                    # 能量占比
                    energy = summary.get('total_energy_consumed', 0)
                    gpu_energy += energy * summary.get('gpu_energy_fraction', 0)
                    cpu_energy += energy * summary.get('cpu_energy_fraction', 0)
                    dram_energy += energy * summary.get('dram_energy_fraction', 0)
                    
                    # 性能指标
                    total_tokens += summary.get('avg_tokens_per_joule', 0) * energy
                    total_samples += summary.get('avg_samples_per_joule', 0) * energy
                    
                    valid_count += 1
        
        if valid_count == 0 or total_energy == 0:
            return {}
        
        # 计算平均指标
        avg_power = total_energy / total_duration if total_duration > 0 else 0
        gpu_fraction = gpu_energy / total_energy if total_energy > 0 else 0
        cpu_fraction = cpu_energy / total_energy if total_energy > 0 else 0
        dram_fraction = dram_energy / total_energy if total_energy > 0 else 0
        
        tokens_per_joule = total_tokens / total_energy if total_energy > 0 else 0
        samples_per_joule = total_samples / total_energy if total_energy > 0 else 0
        
        return {
            'total_energy_consumed': total_energy,
            'total_duration': total_duration,
            'avg_power_draw': avg_power,
            'peak_power_draw': peak_power,
            'gpu_energy_fraction': gpu_fraction,
            'cpu_energy_fraction': cpu_fraction,
            'dram_energy_fraction': dram_fraction,
            'avg_tokens_per_joule': tokens_per_joule,
            'avg_samples_per_joule': samples_per_joule,
            'experiment_count': valid_count
        }
    
    def compare_methods(self, methods: List[str] = None):
        """比较不同方法的性能，生成完整报告"""
        if methods is None:
            methods = list(set(exp['method'] for exp in self.aggregated_data.values()))
        
        print(f"\n{'='*80}")
        print(f"📊 INFRASTRUCTURE COMPARISON ANALYSIS")
        print(f"{'='*80}")
        print(f"Methods to compare: {', '.join(methods)}")
        print(f"Total unique experiments: {len(self.aggregated_data)}")
        
        # 1. 生成计算负载对比
        self._generate_compute_analysis(methods)
        
        # 2. 生成内存负载对比
        self._generate_memory_analysis(methods)

        # 3. 生成网络负载对比
        self._generate_network_analysis(methods)
        
        # 4. 生成功耗分析
        self._generate_power_analysis(methods)
        
        # 5. 生成Agent行为分析
        self._generate_agent_analysis(methods)
        
        # 6. 生成综合报告
        self.generate_comprehensive_report(methods)
        
        print(f"\n{'='*80}")
        print(f"✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"📈 All charts and reports saved to: {self.output_dir}")
        print(f"{'='*80}")
    
    def _generate_compute_analysis(self, methods):
        """生成计算负载分析图表"""
        print("\n⚡ Generating compute analysis...")
        
        for i in range(4):
            fig, axes = plt.subplots(figsize=(8, 6))
            if i == 0:
                self._plot_inference_time_comparison(methods, axes)
            elif i == 1:
                self._plot_gpu_utilization_comparison(methods, axes)
            elif i == 2:
                self._plot_tokens_per_second_comparison(methods, axes)
            elif i == 3:
                self._plot_flops_efficiency_comparison(methods, axes)
            
            plt.tight_layout()
            output_path = self.output_dir / f'compute_infrastructure_analysis_{i}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Compute analysis saved to: {output_path}")
    
    def _plot_inference_time_comparison(self, methods, ax):
        """绘制推理时间对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'compute' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['compute']
                if 'inference_time' in stats:
                    data_by_method[method][task].append(stats['inference_time']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No inference time data available', ha='center', va='center')
            return
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No GPU utilization data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average Inference Time (seconds)', fontweight='bold')
        ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_gpu_utilization_comparison(self, methods, ax):
        """绘制GPU利用率对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'compute' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['compute']
                if 'gpu_util' in stats:
                    data_by_method[method][task].append(stats['gpu_util']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No GPU utilization data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average GPU Usage (%)', fontweight='bold')
        ax.set_title('GPU Utilization by Task', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_tokens_per_second_comparison(self, methods, ax):
        """绘制Tokens/秒对比"""
        data_by_method = defaultdict(list)
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            if 'compute' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['compute']
                if 'tokens_per_second' in stats:
                    data_by_method[method].append(stats['tokens_per_second']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No tokens/second data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        avg_tps = [np.mean(data_by_method[method]) for method in method_names]
        std_tps = [np.std(data_by_method[method]) for method in method_names]
        
        x = np.arange(len(method_names))
        width = 0.6
        
        bars = ax.bar(x, avg_tps, width, yerr=std_tps, capsize=5,
                     color=[self._get_method_color(method) for method in method_names],
                     alpha=0.8, edgecolor='white')
        
        # 添加数据标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 10,
                   f'{avg_tps[i]:.0f}\n±{std_tps[i]:.0f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_ylabel('Average Tokens per Second', fontweight='bold')
        ax.set_title('Token Processing Throughput', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{method.upper()}\n(n={len(data_by_method[method])})" for method in method_names])
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(avg_tps) * 1.3 if avg_tps else 1000)
    
    def _plot_flops_efficiency_comparison(self, methods, ax):
        """绘制FLOPS效率对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'compute' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['compute']
                if 'flops_estimate' in stats:
                    data_by_method[method][task].append(stats['flops_estimate']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No FLOPS efficiency data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average FLOPS (GFLOPS)', fontweight='bold')
        ax.set_title('FLOPS Efficiency by Task', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _generate_memory_analysis(self, methods):
        """生成内存负载分析图表"""
        print("\n📦 Generating memory analysis...")
        
        for i in range(5):
            fig, axes = plt.subplots(figsize=(8, 6))
            if i == 0:
                self._plot_vram_usage_comparison(methods, axes)
            elif i == 1:
                self._plot_ram_usage_comparison(methods, axes)
            elif i == 2:
                self._plot_kv_cache_comparison(methods, axes)
            elif i == 3:
                self._plot_cpu_memory_comparison(methods, axes)
            elif i == 4:
                self._plot_gpu_memory_comparison(methods, axes)
        
            plt.tight_layout()
            output_path = self.output_dir / f'memory_infrastructure_analysis_{i}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Memory analysis saved to: {output_path}")
    
    def _generate_network_analysis(self, methods):
        """生成网络负载分析图表"""
        print("\n📦 Generating network analysis...")
        
        for i in range(1):
            fig, axes = plt.subplots(figsize=(8, 6))
            if i == 0:
                self._plot_network_comparison(methods, axes)
        
            plt.tight_layout()
            output_path = self.output_dir / f'network_infrastructure_analysis_{i}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Memory analysis saved to: {output_path}")

    def _plot_vram_usage_comparison(self, methods, ax):
        """绘制VRAM使用对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'memory' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['memory']
                if 'vram_allocated' in stats:
                    data_by_method[method][task].append(stats['vram_allocated']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No VRAM usage data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average VRAM Usage (GB)', fontweight='bold')
        ax.set_title('VRAM Usage Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_ram_usage_comparison(self, methods, ax):
        """绘制RAM使用对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'memory' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['memory']
                if 'ram_used' in stats:
                    data_by_method[method][task].append(stats['ram_used']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No RAM usage data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average RAM Usage (GB)', fontweight='bold')
        ax.set_title('System RAM Usage Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_network_comparison(self, methods, ax):
        """绘制网络通信对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'network' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['network']
                if 'comm_data_size' in stats:
                    data_by_method[method][task].append(stats['comm_data_size']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No network communication data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_comm_data = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_comm_data.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_comm_data, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average Network Data Transfer (MB)', fontweight='bold')
        ax.set_title('Communication Data by Task', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_kv_cache_comparison(self, methods, ax):
        """绘制KV缓存大小对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'memory' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['memory']
                if 'kv_cache_size' in stats:
                    data_by_method[method][task].append(stats['kv_cache_size']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No KV cache data available', ha='center', va='center')
            return
        
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average KV Cache Size (GB)', fontweight='bold')
        ax.set_title('KV Cache Size Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)

    def _plot_cpu_memory_comparison(self, methods, ax):
        """绘制内存效率对比（VRAM + RAM）"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'memory' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['memory']
                if 'ram_used' in stats:
                    data_by_method[method][task].append(stats['ram_used']['mean'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No memory efficiency data available', ha='center', va='center')
            return
        
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('CPU Memory Usage (GB)', fontweight='bold')
        ax.set_title('Memory Efficiency: RAM', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_gpu_memory_comparison(self, methods, ax):
        """绘制内存效率对比（VRAM + RAM）"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'memory' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['memory']
                if 'vram_allocated' in stats:
                    data_by_method[method][task].append(stats['vram_allocated']['mean'])

        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No memory efficiency data available', ha='center', va='center')
            return
        
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('GPU Memory Usage (GB)', fontweight='bold')
        ax.set_title('Memory Efficiency: GPU VRAM', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _generate_power_analysis(self, methods):
        """生成功耗分析图表"""
        print("\n⚡ Generating power analysis...")
        
        # 检查是否有功耗数据
        has_power_data = any('power_summary' in exp_data and exp_data['power_summary'] 
                           for exp_data in self.aggregated_data.values())
        
        if not has_power_data:
            print("⚠️  No power data available for analysis")
            return
        
        for i in range(4):
            fig, axes = plt.subplots(figsize=(8, 6))    
            if i == 0:        
                self._plot_energy_consumption(methods, axes)
            elif i == 1:
                self._plot_power_efficiency(methods, axes)
            elif i == 2:
                self._plot_energy_breakdown(methods, axes)
            elif i == 3:
                self._plot_cost_analysis(methods, axes)
            plt.tight_layout()
            output_path = self.output_dir / f'power_consumption_analysis_{i}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Power analysis saved to: {output_path}")
    
    def _plot_energy_consumption(self, methods, ax):
        """绘制能耗对比"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'power_summary' in exp_data and exp_data['power_summary']:
                summary = exp_data['power_summary']
                data_by_method[method][task].append(summary['total_energy_consumed'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No energy consumption data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Total Energy Consumed (Joules)', fontweight='bold')
        ax.set_title('Energy Consumption per Experiment', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_power_efficiency(self, methods, ax):
        """绘制能效对比（Tokens/Joule）"""
        data_by_method = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            task = exp_data['task']
            if 'power_summary' in exp_data and exp_data['power_summary']:
                summary = exp_data['power_summary']
                if 'avg_tokens_per_joule' in summary:
                    data_by_method[method][task].append(summary['avg_tokens_per_joule'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No power efficiency data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        
        for i, method in enumerate(method_names):
            avg_flops = []
            for task in tasks:
                values = data_by_method[method].get(task, [])
                avg_flops.append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, avg_flops, width, 
                         label=method.upper(),
                         color=self._get_method_color(method),
                         alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.1f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Tokens per Joule', fontweight='bold')
        ax.set_title('Energy Efficiency: Tokens per Joule', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max(max(flops for flops in data_by_method[method][tasks[0]]) for method in method_names) * 1.3 if data_by_method else 1000)
    
    def _plot_energy_breakdown(self, methods, ax):
        """绘制能耗分解（GPU/CPU/DRAM）"""
        data_by_method = defaultdict(lambda: defaultdict(lambda: {'gpu': [], 'cpu': [], 'dram': []}))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            task = exp_data['task']
            if 'power_summary' in exp_data and exp_data['power_summary']:
                summary = exp_data['power_summary']
                data_by_method[method][task]['gpu'].append(summary['gpu_energy_fraction'])
                data_by_method[method][task]['cpu'].append(summary['cpu_energy_fraction'])
                data_by_method[method][task]['dram'].append(summary['dram_energy_fraction'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No energy breakdown data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(data_by_method.keys())
        tasks = sorted(set(task for method_data in data_by_method.values() for task in method_data.keys()))
        
        x = np.arange(len(tasks))
        width = 0.8 / len(method_names)
        max_val = 0.0
        for i, method in enumerate(method_names):
            power = {'gpu': [], 'cpu': [], 'dram': []}
            for task in tasks:
                for part in ['cpu', 'dram', 'gpu']:
                    values = data_by_method[method].get(task).get(part)
                    power[part].append(np.mean(values) if values else 0)
            
            offset = i * width - (len(method_names) - 1) * width / 2

            # GPU部分
            bottom = np.zeros(len(tasks))
            gpu_bars = ax.bar(x + offset, power['gpu'], bottom=bottom, 
                         label='GPU Energy',
                         color='#FF6B6B',
                         alpha=0.8)
            bottom += power['gpu']
            
            # CPU部分
            cpu_bars = ax.bar(x + offset, power['cpu'], bottom=bottom, 
                         label='CPU Energy',
                         color='#4ECDC4',
                         alpha=0.8, edgecolor='white')
            bottom += power['cpu']
            
            # DRAM部分
            dram_bars = ax.bar(x + offset, power['dram'], bottom=bottom, 
                         label='DRAM Energy', color='#45B7D1', alpha=0.8)
            
            max_val = max(max_val, max(list(np.array(power['cpu']) + np.array(power['gpu']) + np.array(power['dram']))))

        
        ax.set_ylabel('Energy Consumption (Joule)', fontweight='bold')
        ax.set_title('Energy Consumption Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        print(f"data_by_method: {data_by_method['latent_mas'].values()} and {data_by_method['text_mas'].values()}")
        ax.set_ylim(0, max_val * 1.3)
    
    def _plot_cost_analysis(self, methods, ax):
        """绘制成本分析"""
        data_by_method = defaultdict(list)
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            if 'power_summary' in exp_data and exp_data['power_summary']:
                summary = exp_data['power_summary']
                data_by_method[method].append(summary['total_energy_consumed'])
        
        if not data_by_method:
            ax.text(0.5, 0.5, 'No cost analysis data available', ha='center', va='center')
            return
        
        # 估算成本（基于美国平均电价 $0.12/kWh）
        electricity_cost_per_kwh = 0.12
        samples_per_day = 1000000  # 1M样本/天
        days_per_year = 365
        
        # 准备数据
        method_names = list(data_by_method.keys())
        costs_per_year = []
        
        for method in method_names:
            avg_energy_per_sample = np.mean(data_by_method[method]) / samples_per_day
            yearly_energy_kwh = (avg_energy_per_sample * samples_per_day * days_per_year) / 3600000
            yearly_cost = yearly_energy_kwh * electricity_cost_per_kwh
            costs_per_year.append(yearly_cost)
        
        x = np.arange(len(method_names))
        width = 0.6
        
        bars = ax.bar(x, costs_per_year, width,
                     color=[self._get_method_color(method) for method in method_names],
                     alpha=0.8, edgecolor='white')
        
        # 添加数据标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 100,
                   f'${height:,.0f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_ylabel('Estimated Annual Cost ($)', fontweight='bold')
        ax.set_title(f'Cost Analysis at Scale\n({samples_per_day/1000000:.1f}M samples/day, ${electricity_cost_per_kwh:.2f}/kWh)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([method.upper() for method in method_names])
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(costs_per_year) * 1.3 if costs_per_year else 10000)
    
    def _generate_agent_analysis(self, methods):
        """生成Agent行为分析图表"""
        print("\n🤖 Generating agent analysis...")
        
        for i in range(4):
            fig, axes = plt.subplots(figsize=(8, 6))
            if i == 0:
                self._plot_agent_computation_breakdown(methods, axes)
            elif i == 1:
                self._plot_agent_memory_usage(methods, axes)
            elif i == 2:
                self._plot_agent_power_consumption(methods, axes)
            elif i == 3:
                self._plot_agent_efficiency(methods, axes)
            
            plt.tight_layout()
            output_path = self.output_dir / f'agent_behavior_analysis_{i}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Agent analysis saved to: {output_path}")
    
    def _plot_agent_computation_breakdown(self, methods, ax):
        """绘制Agent计算时间分解"""
        agent_data = defaultdict(lambda: defaultdict(lambda: {'cpu': [], 'gpu': []}))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            # 从Agent指标中提取数据
            for agent_name, steps in exp_data['agent_metrics'].items():
                for step_idx, metrics in steps.items():
                    for metric in metrics:
                        compute = metric.get('compute', {})
                        inference_time = compute.get('inference_time', 0)
                        total_time = compute.get('total_time', inference_time + 0.1)
                        
                        cpu_time = total_time - inference_time
                        gpu_time = inference_time
                        
                        cpu_ratio = cpu_time / total_time if total_time > 0 else 0
                        gpu_ratio = gpu_time / total_time if total_time > 0 else 0
                        
                        agent_data[method][agent_name]['cpu'].append(cpu_time)
                        agent_data[method][agent_name]['gpu'].append(gpu_time)
        
        if not agent_data:
            ax.text(0.5, 0.5, 'No agent computation data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(agent_data.keys())
        agent_names = sorted(set(agent for method_data in agent_data.values() 
                               for agent in method_data.keys()))
        
        x = np.arange(len(agent_names))
        width = 0.8 / len(method_names)
        max_val = 0.0
        for i, method in enumerate(method_names):
            cpu_ratios = []
            gpu_ratios = []
            
            for agent in agent_names:
                if agent in agent_data[method]:
                    cpu_vals = agent_data[method][agent]['cpu']
                    gpu_vals = agent_data[method][agent]['gpu']
                    cpu_ratios.append(np.mean(cpu_vals) if cpu_vals else 0)
                    gpu_ratios.append(np.mean(gpu_vals) if gpu_vals else 0)
                else:
                    cpu_ratios.append(0)
                    gpu_ratios.append(0)

            max_val = max(max_val, max(c for c in list(np.array(cpu_ratios) + np.array(gpu_ratios))))

            offset = i * width - (len(method_names) - 1) * width / 2
            bottom = np.zeros(len(agent_names))
            
            # CPU部分
            cpu_bars = ax.bar(x + offset, cpu_ratios, width, bottom=bottom,
                            label=f'{method.upper()} CPU',
                            color=self._get_method_color(method),
                            alpha=0.3, edgecolor='white')
            bottom += cpu_ratios
            
            # GPU部分
            gpu_bars = ax.bar(x + offset, gpu_ratios, width, bottom=bottom,
                            label=f'{method.upper()} GPU',
                            color=self._get_method_color(method),
                            alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, (cpu, gpu) in enumerate(zip(cpu_ratios, gpu_ratios)):
                total = cpu + gpu
                if total > 0.1:  # 只显示显著的值
                    cpu_pct = cpu / total * 100
                    gpu_pct = gpu / total * 100
                    
                    if cpu_pct > 15:
                        ax.text(x[j] + offset, bottom[j] - gpu/2, f'CPU:{cpu_pct:.0f}%', 
                               ha='center', va='center', fontsize=6, color='black')
                    if gpu_pct > 15:
                        ax.text(x[j] + offset, bottom[j] - gpu/2, f'GPU:{gpu_pct:.0f}%', 
                               ha='center', va='center', fontsize=6, color='white')
        
        ax.set_ylabel('Service Time (s)', fontweight='bold')
        ax.set_title('Agent Computation Time Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.legend(loc='upper right', ncol = 2)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.3*max_val)
    
    def _plot_agent_memory_usage(self, methods, ax):
        """绘制Agent内存使用情况"""
        agent_data = defaultdict(lambda: defaultdict(lambda: {'vram': [], 'ram': []}))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            for agent_name, steps in exp_data['agent_metrics'].items():
                for step_idx, metrics in steps.items():
                    for metric in metrics:
                        memory = metric.get('memory', {})
                        vram = memory.get('vram_allocated', 0)
                        ram = memory.get('ram_used', 0)
                        
                        agent_data[method][agent_name]['vram'].append(vram)
                        agent_data[method][agent_name]['ram'].append(ram)
        
        if not agent_data:
            ax.text(0.5, 0.5, 'No agent memory data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(agent_data.keys())
        agent_names = sorted(set(agent for method_data in agent_data.values() 
                               for agent in method_data.keys()))
        
        x = np.arange(len(agent_names))
        width = 0.8 / len(method_names)
        
        max_value = 0
        
        for i, method in enumerate(method_names):
            vram_means = []
            ram_means = []
            
            for agent in agent_names:
                if agent in agent_data[method]:
                    vram_vals = agent_data[method][agent]['vram']
                    ram_vals = agent_data[method][agent]['ram']
                    vram_means.append(np.mean(vram_vals) if vram_vals else 0)
                    ram_means.append(np.mean(ram_vals) if ram_vals else 0)
                else:
                    vram_means.append(0)
                    ram_means.append(0)
            
            max_value = max(max_value, max(c for c in list(np.array(vram_means) + np.array(ram_means))))
            
            offset = i * width - (len(method_names) - 1) * width / 2
            
            # VRAM部分
            vram_bars = ax.bar(x + offset, vram_means, width,
                             label=f'{method.upper()} VRAM',
                             color=self._get_method_color(method),
                             alpha=0.8, edgecolor='white')
            
            # RAM部分 (堆叠在VRAM上方)
            ram_bars = ax.bar(x + offset, ram_means, width, bottom=vram_means,
                            label=f'{method.upper()} RAM',
                            color=self._get_method_color(method),
                            alpha=0.8, hatch='//', edgecolor='white')
            
            # 添加数据标签
            for j, (vram, ram) in enumerate(zip(vram_means, ram_means)):
                if vram > 0.5:
                    ax.text(x[j] + offset, vram/2, f'{vram:.1f}GB', 
                           ha='center', va='center', fontsize=6, color='white')
                if ram > 0.5:
                    ax.text(x[j] + offset, vram + ram/2, f'{ram:.1f}GB', 
                           ha='center', va='center', fontsize=6, color='black')
        
        ax.set_ylabel('Memory Usage (GB)', fontweight='bold')
        ax.set_title('Agent Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.legend(loc='upper right', ncol = 2)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max_value * 1.3 if max_value > 0 else 24)
    
    def _plot_agent_power_consumption(self, methods, ax):
        """绘制Agent功耗分析"""
        agent_energy = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue

            for agent_name, steps in exp_data['agent_metrics'].items():
                for step_idx, metrics in steps.items():
                    for metric in metrics:
                        power_metrics = metric.get('power', {})
                        agent_energy[method][agent_name].append(power_metrics['total_energy_consumed'])
        
        if not agent_energy:
            print(f"Warning: agent energy data cannot be obtained!")
            ax.text(0.5, 0.5, 'No agent power data available', ha='center', va='center')
            return

        print(f"agent_energy: {agent_energy}")
        
        # 准备数据
        method_names = list(agent_energy.keys())
        agent_names = sorted(set(agent for method_data in agent_energy.values() 
                               for agent in method_data.keys()))
        
        x = np.arange(len(agent_names))
        width = 0.8 / len(method_names)
        
        max_val = 0
        for i, method in enumerate(method_names):
            energy_means = []
            
            for agent in agent_names:
                if agent in agent_energy[method]:
                    energy_means.append(np.mean(agent_energy[method][agent]))
                else:
                    energy_means.append(0)
            max_val = max(max_val, max(energy_means))
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, energy_means, width,
                        label=method.upper(),
                        color=self._get_method_color(method),
                        alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 1:  # 只显示显著的值
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                           f'{height:.1f}J', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Energy per Agent (Joules)', fontweight='bold')
        ax.set_title('Agent Power Consumption Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.legend(loc='upper right', ncol = 2)
        ax.grid(axis='y', alpha=0.3)

        print(f"max_val: {max_val}")
        ax.set_ylim(0, max_val * 1.3)
    
    def _plot_agent_efficiency(self, methods, ax):
        """绘制Agent效率分析（Tokens/Joule）"""
        agent_data = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue

            for agent_name, steps in exp_data['agent_metrics'].items():
                total_energy = 0.0
                total_tokens = 0.0
                for step_idx, metrics in steps.items():
                    for metric in metrics:
                        power_metrics = metric.get('power', {})
                        total_energy += power_metrics['total_energy_consumed']
                        total_tokens += power_metrics['total_energy_consumed'] * power_metrics['avg_tokens_per_joule']
                agent_data[method][agent_name].append(total_tokens/total_energy if total_energy > 0 else 0)
        
        if not agent_data:
            ax.text(0.5, 0.5, 'No agent efficiency data available', ha='center', va='center')
            return
        
        # 准备数据
        method_names = list(agent_data.keys())
        agent_names = sorted(set(agent for method_data in agent_data.values() 
                               for agent in method_data.keys()))
        
        x = np.arange(len(agent_names))
        width = 0.8 / len(method_names)
        max_val = 0.0
        for i, method in enumerate(method_names):
            efficiency_means = []
            
            for agent in agent_names:
                if agent in agent_data[method]:
                    efficiency_means.append(np.mean(agent_data[method][agent]))
                else:
                    efficiency_means.append(0)
            max_val = max(max_val, max(efficiency_means))
            
            offset = i * width - (len(method_names) - 1) * width / 2
            bars = ax.bar(x + offset, efficiency_means, width,
                        label=method.upper(),
                        color=self._get_method_color(method),
                        alpha=0.8, edgecolor='white')
            
            # 添加数据标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 10:  # 只显示显著的值
                    ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                           f'{height:.0f}', 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Tokens per Joule', fontweight='bold')
        ax.set_title('Agent Energy Efficiency', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.legend(loc='upper right', ncol = 2)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max_val * 1.3)
    
    def _get_method_color(self, method: str) -> str:
        """获取方法对应的颜色"""
        color_map = {
            'latent_mas': '#4ECDC4',  # 青色 - 表示优化
            'text_mas': '#FF6B6B',    # 红色 - 表示基线
            'baseline': '#45B7D1',    # 蓝色 - 表示单Agent
            'cpu': '#96CEB4',         # 绿色
            'gpu': '#FECA57',         # 橙色
            'ram': '#A78BFA',         # 紫色
            'vram': '#F08080',        # 粉色
        }
        return color_map.get(method.lower(), '#95A5A6')
    
    def generate_comprehensive_report(self, methods: List[str]):
        """生成综合报告"""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_version': '1.0',
                'metrics_dir': str(self.metrics_dir),
                'total_experiments': len(self.aggregated_data),
                'methods_analyzed': methods
            },
            'summary': {},
            'method_comparisons': {},
            'recommendations': [],
            'insights': {}
        }
        
        # 1. 计算关键指标
        method_stats = defaultdict(dict)
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            # 基础统计
            inference_time = exp_data['summary_stats']['compute']['inference_time']['mean'] if 'compute' in exp_data['summary_stats'] and 'inference_time' in exp_data['summary_stats']['compute'] else 0
            vram_usage = exp_data['summary_stats']['memory']['vram_allocated']['mean'] if 'memory' in exp_data['summary_stats'] and 'vram_allocated' in exp_data['summary_stats']['memory'] else 0
            tokens_per_second = exp_data['summary_stats']['compute']['tokens_per_second']['mean'] if 'compute' in exp_data['summary_stats'] and 'tokens_per_second' in exp_data['summary_stats']['compute'] else 0
            
            # 功耗统计
            energy_efficiency = 0
            cost_per_million = 0
            
            if 'power_summary' in exp_data and exp_data['power_summary']:
                power_summary = exp_data['power_summary']
                energy_efficiency = power_summary.get('avg_tokens_per_joule', 0)
                
                # 估算成本
                electricity_cost_per_kwh = 0.12
                samples_per_day = 1000000
                energy_per_sample = power_summary.get('total_energy_consumed', 0) / exp_data['total_samples'] if exp_data['total_samples'] > 0 else 0
                daily_energy_kwh = (energy_per_sample * samples_per_day) / 3600000
                daily_cost = daily_energy_kwh * electricity_cost_per_kwh
                cost_per_million = daily_cost
            
            method_stats[method][exp_key] = {
                'inference_time': inference_time,
                'vram_usage': vram_usage,
                'tokens_per_second': tokens_per_second,
                'energy_efficiency': energy_efficiency,
                'cost_per_million': cost_per_million
            }
        
        # 2. 生成方法对比
        if method_stats:
            report['method_comparisons'] = convert_numpy_types(method_stats)
            
            # 3. 生成建议
            if 'latent_mas' in method_stats and ('text_mas' in method_stats or 'baseline' in method_stats):
                latent_stats = method_stats['latent_mas']
                comparison_method = 'text_mas' if 'text_mas' in method_stats else 'baseline'
                comparison_stats = method_stats[comparison_method]
                
                # 性能对比
                latent_inference = np.mean([stats['inference_time'] for stats in latent_stats.values()])
                comparison_inference = np.mean([stats['inference_time'] for stats in comparison_stats.values()])
                
                # VRAM对比
                latent_vram = np.mean([stats['vram_usage'] for stats in latent_stats.values()])
                comparison_vram = np.mean([stats['vram_usage'] for stats in comparison_stats.values()])
                
                # 能效对比
                latent_efficiency = np.mean([stats['energy_efficiency'] for stats in latent_stats.values()])
                comparison_efficiency = np.mean([stats['energy_efficiency'] for stats in comparison_stats.values()])
                
                # 生成具体建议
                if latent_inference < comparison_inference * 0.7:
                    savings_pct = (1 - latent_inference / comparison_inference) * 100
                    report['recommendations'].append(
                        f"LatentMAS reduces inference time by {savings_pct:.1f}%, "
                        f"making it ideal for latency-sensitive applications."
                    )
                
                if latent_vram < comparison_vram * 0.85:
                    savings_pct = (1 - latent_vram / comparison_vram) * 100
                    report['recommendations'].append(
                        f"LatentMAS reduces VRAM usage by {savings_pct:.1f}%, "
                        f"enabling deployment on lower-memory GPUs."
                    )
                
                if latent_efficiency > comparison_efficiency * 1.5:
                    improvement_pct = (latent_efficiency / comparison_efficiency - 1) * 100
                    report['recommendations'].append(
                        f"LatentMAS improves energy efficiency by {improvement_pct:.1f}%, "
                        f"providing significant cost savings at scale."
                    )
        
        # 4. 生成关键洞察
        report['insights'] = {
            'compute_insights': self._generate_compute_insights(methods),
            'memory_insights': self._generate_memory_insights(methods),
            'power_insights': self._generate_power_insights(methods),
            'agent_insights': self._generate_agent_insights(methods)
        }
        
        # 5. 保存报告
        report_path = self.output_dir / 'infrastructure_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(convert_numpy_types(report), f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"Report saved to: {report_path}")
        print(f"\n📊 KEY INSIGHTS:")
        
        for category, insights in report['insights'].items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for insight in insights:
                print(f"• {insight}")
        
        if report['recommendations']:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"{'='*80}")
        
        return report
    
    def _generate_compute_insights(self, methods) -> List[str]:
        """生成计算相关的洞察"""
        insights = []
        
        # 分析推理时间
        inference_times = defaultdict(list)
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            if 'compute' in exp_data['summary_stats'] and 'inference_time' in exp_data['summary_stats']['compute']:
                inference_times[method].append(exp_data['summary_stats']['compute']['inference_time']['mean'])
        
        if inference_times:
            fastest_method = min(inference_times.items(), key=lambda x: np.mean(x[1]) if x[1] else float('inf'))[0]
            slowest_method = max(inference_times.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)[0]
            
            if fastest_method != slowest_method:
                insights.append(
                    f"{fastest_method.upper()} achieves {np.mean(inference_times[slowest_method])/np.mean(inference_times[fastest_method]):.1f}x "
                    f"faster inference compared to {slowest_method.upper()}"
                )
        
        return insights
    
    def _generate_memory_insights(self, methods) -> List[str]:
        """生成内存相关的洞察"""
        insights = []
        
        vram_usage = defaultdict(list)
        kv_cache_size = defaultdict(list)
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            if 'memory' in exp_data['summary_stats']:
                stats = exp_data['summary_stats']['memory']
                if 'vram_allocated' in stats:
                    vram_usage[method].append(stats['vram_allocated']['mean'])
                if 'kv_cache_size' in stats:
                    kv_cache_size[method].append(stats['kv_cache_size']['mean'])
        
        if vram_usage:
            most_efficient = min(vram_usage.items(), key=lambda x: np.mean(x[1]) if x[1] else float('inf'))[0]
            least_efficient = max(vram_usage.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)[0]
            
            if most_efficient != least_efficient:
                insights.append(
                    f"{most_efficient.upper()} uses {np.mean(vram_usage[least_efficient])/np.mean(vram_usage[most_efficient]):.1f}x "
                    f"less VRAM compared to {least_efficient.upper()}"
                )
        
        if kv_cache_size:
            smallest_kv = min(kv_cache_size.items(), key=lambda x: np.mean(x[1]) if x[1] else float('inf'))[0]
            largest_kv = max(kv_cache_size.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)[0]
            
            if smallest_kv != largest_kv:
                insights.append(
                    f"{smallest_kv.upper()} reduces KV cache size by {1 - np.mean(kv_cache_size[smallest_kv])/np.mean(kv_cache_size[largest_kv]):.1%}, "
                    f"significantly improving memory efficiency"
                )
        
        return insights
    
    def _generate_power_insights(self, methods) -> List[str]:
        """生成功耗相关的洞察"""
        insights = []
        
        energy_efficiency = defaultdict(list)
        total_energy = defaultdict(list)
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            if 'power_summary' in exp_data and exp_data['power_summary']:
                summary = exp_data['power_summary']
                energy_efficiency[method].append(summary.get('avg_tokens_per_joule', 0))
                total_energy[method].append(summary.get('total_energy_consumed', 0))
        
        if energy_efficiency:
            most_efficient = max(energy_efficiency.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)[0]
            least_efficient = min(energy_efficiency.items(), key=lambda x: np.mean(x[1]) if x[1] else float('inf'))[0]
            
            if most_efficient != least_efficient:
                insights.append(
                    f"{most_efficient.upper()} achieves {np.mean(energy_efficiency[most_efficient])/np.mean(energy_efficiency[least_efficient]):.1f}x "
                    f"better energy efficiency (tokens/joule) than {least_efficient.upper()}"
                )
        
        if total_energy:
            lowest_energy = min(total_energy.items(), key=lambda x: np.mean(x[1]) if x[1] else float('inf'))[0]
            highest_energy = max(total_energy.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)[0]
            
            if lowest_energy != highest_energy:
                insights.append(
                    f"{lowest_energy.upper()} consumes {np.mean(total_energy[highest_energy])/np.mean(total_energy[lowest_energy]):.1f}x "
                    f"less total energy than {highest_energy.upper()}, reducing operational costs"
                )
        
        return insights
    
    def _generate_agent_insights(self, methods) -> List[str]:
        """生成Agent相关的洞察"""
        insights = []
        
        # 分析Agent计算模式
        cpu_gpu_ratios = defaultdict(lambda: defaultdict(list))
        
        for exp_key, exp_data in self.aggregated_data.items():
            method = exp_data['method']
            if method not in methods:
                continue
            
            for agent_name, steps in exp_data['agent_metrics'].items():
                for step_idx, metrics in steps.items():
                    for metric in metrics:
                        compute = metric.get('compute', {})
                        inference_time = compute.get('inference_time', 0)
                        total_time = compute.get('total_time', inference_time + 0.1)
                        
                        gpu_ratio = inference_time / total_time if total_time > 0 else 0
                        cpu_ratio = 1 - gpu_ratio
                        
                        cpu_gpu_ratios[method][agent_name].append((cpu_ratio, gpu_ratio))
        
        if cpu_gpu_ratios:
            for method, agent_data in cpu_gpu_ratios.items():
                for agent_name, ratios in agent_data.items():
                    avg_cpu = np.mean([r[0] for r in ratios])
                    avg_gpu = np.mean([r[1] for r in ratios])
                    
                    if avg_gpu > 0.7:
                        insights.append(
                            f"{agent_name} in {method.upper()} is GPU-bound ({avg_gpu:.1%} GPU time), "
                            f"suggesting optimization opportunities for GPU utilization"
                        )
                    elif avg_cpu > 0.7:
                        insights.append(
                            f"{agent_name} in {method.upper()} is CPU-bound ({avg_cpu:.1%} CPU time), "
                            f"indicating potential bottlenecks in data preprocessing or memory management"
                        )
        
        return insights
    
    def generate_visualization_code(self):
        """生成可视化代码示例"""
        viz_code = f"""
# Infrastructure Analysis Visualization Code
# Generated on {datetime.now().isoformat()}

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load analysis data
analysis_dir = Path('{self.output_dir}')
report_path = analysis_dir / 'infrastructure_analysis_report.json'

with open(report_path) as f:
    report = json.load(f)

# Example: Plot inference time comparison
methods = {list(set(exp['method'] for exp in self.aggregated_data.values()))}
inference_times = {{}}

for exp_key, exp_data in {dict(self.aggregated_data)}.items():
    method = exp_data['method']
    if method in methods and 'compute' in exp_data['summary_stats']:
        inference_times.setdefault(method, []).append(exp_data['summary_stats']['compute']['inference_time']['mean'])

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
method_names = list(inference_times.keys())
avg_times = [np.mean(inference_times[method]) for method in method_names]

bars = ax.bar(method_names, avg_times, color=['#4ECDC4', '#FF6B6B', '#45B7D1'][:len(method_names)])
ax.set_ylabel('Average Inference Time (seconds)')
ax.set_title('Inference Time Comparison by Method')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            f'{{height:.3f}}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(analysis_dir / 'inference_time_comparison.png', dpi=300)
plt.show()
"""
        return viz_code

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Infrastructure Analysis for Multi-Agent Systems')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                       help='Directory containing metrics JSON files')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--methods', type=str, nargs='+',
                       help='Specific methods to analyze (e.g., latent_mas text_mas baseline)')
    
    args = parser.parse_args()
    
    print(f"🔍 Starting infrastructure analysis...")
    print(f"📁 Metrics directory: {args.metrics_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    # 创建分析器
    analyzer = InfrastructureAnalyzer(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir
    )
    
    # 获取要分析的方法
    methods_to_analyze = args.methods if args.methods else None
    
    # 生成完整分析
    analyzer.compare_methods(methods=methods_to_analyze)
    
    # 生成可视化代码示例
    viz_code = analyzer.generate_visualization_code()
    code_path = analyzer.output_dir / 'visualization_examples.py'
    with open(code_path, 'w') as f:
        f.write(viz_code)
    print(f"✅ Visualization examples saved to: {code_path}")
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📈 Find all results in: {analyzer.output_dir}")

if __name__ == "__main__":
    main()