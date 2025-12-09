from datetime import datetime
import json
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import math


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
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.data = self._load_all_metrics()
    
    def _load_all_metrics(self) -> Dict[str, List]:
        """加载所有指标数据"""
        all_data = defaultdict(list)
        
        for filepath in self.metrics_dir.glob("*.json"):
            if '_power.json' in str(filepath):
                continue  # 跳过功耗文件，稍后单独加载
            with open(filepath) as f:
                data = json.load(f)
                method = data['method']
                all_data[method].append(data)
        
        # 加载功耗指标
        for method, experiments in all_data.items():
            for exp in experiments:
                experiment_name = exp['experiment_name']
                power_file = self.metrics_dir / f"{experiment_name}_power.json"
                if power_file.exists():
                    with open(power_file) as f:
                        power_data = json.load(f)
                        exp['power_data'] = power_data
        
        return all_data
    
    def compare_methods(self, methods: List[str] = None):
        """比较不同方法的性能"""
        if methods is None:
            methods = list(self.data.keys())
        
        if not methods:
            print("No metrics data found. Please run experiments first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Multi-Agent Communication Methods: Infrastructure Impact Comparison', fontsize=16)
        
        self._plot_compute_comparison(methods, axes[0, 0])
        self._plot_memory_comparison(methods, axes[0, 1])
        self._plot_network_comparison(methods, axes[1, 0])
        self._plot_storage_comparison(methods, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('infrastructure_comparison.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    def _plot_compute_comparison(self, methods, ax):
        """绘制计算负载对比"""
        compute_data = defaultdict(list)
        
        for method in methods:
            for exp in self.data[method]:
                if 'overall_metrics' in exp and 'compute' in exp['overall_metrics']:
                    for metric in exp['overall_metrics']['compute']:
                        compute_data[method].append({
                            'gpu_util': metric.get('gpu_util', 0),
                            'inference_time': metric.get('inference_time', 0),
                            'flops_estimate': metric.get('flops_estimate', 0),
                            'tokens_per_second': metric.get('tokens_per_second', 0)
                        })
        
        # 计算平均值
        avg_metrics = defaultdict(dict)
        for method, metrics in compute_data.items():
            if metrics:
                avg_metrics[method] = {
                    'gpu_util': np.mean([m['gpu_util'] for m in metrics]),
                    'inference_time': np.mean([m['inference_time'] for m in metrics]),
                    'flops_estimate': np.mean([m['flops_estimate'] for m in metrics]),
                    'tokens_per_second': np.mean([m['tokens_per_second'] for m in metrics])
                }
        
        # 创建DataFrame
        df = pd.DataFrame(avg_metrics).T
        
        # 绘制多指标对比
        x = np.arange(len(methods))
        width = 0.2
        
        ax.bar(x - width*1.5, df['gpu_util'], width, label='GPU Util (%)', alpha=0.8)
        ax.bar(x - width/2, df['inference_time'] * 10, width, label='Inference Time x10 (s)', alpha=0.8)  # 缩放以便可视化
        ax.bar(x + width/2, df['flops_estimate'] / 100, width, label='GFLOPS / 100', alpha=0.8)  # 缩放
        ax.bar(x + width*1.5, df['tokens_per_second'] / 1000, width, label='Tokens/sec / 1000', alpha=0.8)  # 缩放
        
        ax.set_xlabel('Communication Method')
        ax.set_ylabel('Normalized Metrics')
        ax.set_title('Compute Load Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_memory_comparison(self, methods, ax):
        """绘制内存负载对比"""
        memory_data = defaultdict(list)
        
        for method in methods:
            for exp in self.data[method]:
                if 'overall_metrics' in exp and 'memory' in exp['overall_metrics']:
                    for metric in exp['overall_metrics']['memory']:
                        memory_data[method].append({
                            'vram_allocated': metric.get('vram_allocated', 0),
                            'kv_cache_size': metric.get('kv_cache_size', 0),
                            'latent_vector_size': metric.get('latent_vector_size', 0)
                        })
        
        # 计算平均值
        avg_metrics = defaultdict(dict)
        for method, metrics in memory_data.items():
            if metrics:
                avg_metrics[method] = {
                    'vram_allocated': np.mean([m['vram_allocated'] for m in metrics]),
                    'kv_cache_size': np.mean([m['kv_cache_size'] for m in metrics]),
                    'latent_vector_size': np.mean([m['latent_vector_size'] for m in metrics])
                }
        
        # 创建DataFrame
        df = pd.DataFrame(avg_metrics).T
        
        # 绘制堆叠柱状图
        x = np.arange(len(methods))
        width = 0.6
        
        bottom = np.zeros(len(methods))
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        for i, (component, color) in enumerate([('vram_allocated', '#ff6b6b'), 
                                               ('kv_cache_size', '#4ecdc4'), 
                                               ('latent_vector_size', '#45b7d1')]):
            values = [df[component].get(method, 0) for method in methods]
            ax.bar(x, values, width, bottom=bottom, label=component.replace('_', ' ').title(), color=color, alpha=0.8)
            bottom += values
        
        ax.set_xlabel('Communication Method')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('Memory Footprint Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_network_comparison(self, methods, ax):
        """绘制网络通信对比"""
        network_data = defaultdict(list)
        
        for method in methods:
            for exp in self.data[method]:
                if 'overall_metrics' in exp and 'network' in exp['overall_metrics']:
                    for metric in exp['overall_metrics']['network']:
                        network_data[method].append({
                            'comm_data_size': metric.get('comm_data_size', 0),
                            'comm_latency': metric.get('comm_latency', 0),
                            'bandwidth_usage': metric.get('bandwidth_usage', 0),
                            'agent_comm_count': metric.get('agent_comm_count', 0)
                        })
        
        # 计算平均值
        avg_metrics = defaultdict(dict)
        for method, metrics in network_data.items():
            if metrics:
                avg_metrics[method] = {
                    'comm_data_size': np.mean([m['comm_data_size'] for m in metrics]),
                    'comm_latency': np.mean([m['comm_latency'] for m in metrics]),
                    'bandwidth_usage': np.mean([m['bandwidth_usage'] for m in metrics]),
                    'agent_comm_count': np.mean([m['agent_comm_count'] for m in metrics])
                }
        
        # 创建DataFrame
        df = pd.DataFrame(avg_metrics).T
        
        # 绘制雷达图
        categories = ['comm_data_size', 'comm_latency', 'bandwidth_usage']
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(223, polar=True)
        
        for method in methods:
            if method in df.index:
                values = df.loc[method, categories].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=method)
                ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax.set_title('Network Communication Profile')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    def _plot_storage_comparison(self, methods, ax):
        """绘制存储对比"""
        storage_data = defaultdict(list)
        
        for method in methods:
            for exp in self.data[method]:
                if 'overall_metrics' in exp and 'storage' in exp['overall_metrics']:
                    for metric in exp['overall_metrics']['storage']:
                        storage_data[method].append({
                            'input_data_size': metric.get('input_data_size', 0),
                            'output_data_size': metric.get('output_data_size', 0),
                            'intermediate_data_size': metric.get('intermediate_data_size', 0),
                            'log_data_size': metric.get('log_data_size', 0)
                        })
        
        # 计算总存储需求
        total_storage = {}
        for method, metrics in storage_data.items():
            if metrics:
                total_storage[method] = {
                    'total': np.mean([
                        m['input_data_size'] + m['output_data_size'] + 
                        m['intermediate_data_size'] + m['log_data_size']
                        for m in metrics
                    ]),
                    'breakdown': {
                        'input': np.mean([m['input_data_size'] for m in metrics]),
                        'output': np.mean([m['output_data_size'] for m in metrics]),
                        'intermediate': np.mean([m['intermediate_data_size'] for m in metrics]),
                        'log': np.mean([m['log_data_size'] for m in metrics])
                    }
                }
        
        # 绘制饼图对比
        if total_storage:
            method = list(total_storage.keys())[0]  # 使用第一个方法
            breakdown = total_storage[method]['breakdown']
            
            labels = list(breakdown.keys())
            sizes = list(breakdown.values())
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Storage Breakdown ({method})')
    
    def generate_detailed_report(self):
        """生成详细报告 - 修复NumPy序列化问题"""
        report = {
            'summary': {},
            'method_comparisons': {},
            'recommendations': [],
            'generated_at': datetime.now().isoformat(),
            'analysis_version': '1.0'
        }
        
        # 计算方法对比
        method_stats = defaultdict(dict)
        
        for method, experiments in self.data.items():
            if not experiments:
                continue
            
            # 提取所有指标
            all_metrics = defaultdict(list)
            for exp in experiments:
                for dimension in ['compute', 'memory', 'network', 'storage', 'power']:
                    if dimension in exp['overall_metrics']:
                        for metric in exp['overall_metrics'][dimension]:
                            for key, value in metric.items():
                                if isinstance(value, (int, float, np.number)):
                                    all_metrics[f"{dimension}_{key}"].append(value)
            
            # 计算统计摘要
            for metric_name, values in all_metrics.items():
                if values:
                    # 确保values是Python原生类型
                    values = [float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for v in values]
                    
                    method_stats[method][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
        
        # 将method_stats转换为可序列化格式
        report['method_comparisons'] = convert_numpy_types(method_stats)
        
        # 生成建议
        if method_stats:
            latent_methods = [m for m in method_stats.keys() if 'latent' in m.lower()]
            text_methods = [m for m in method_stats.keys() if 'text' in m.lower() or 'baseline' in m.lower()]
            
            if latent_methods and text_methods:
                latent_method = latent_methods[0]
                text_method = text_methods[0]
                
                # 比较关键指标 - 确保转换为Python原生类型
                latent_compute = float(method_stats[latent_method].get('compute_inference_time', {}).get('mean', float('inf')))
                text_compute = float(method_stats[text_method].get('compute_inference_time', {}).get('mean', float('inf')))
                
                latent_memory = float(method_stats[latent_method].get('memory_vram_allocated', {}).get('mean', float('inf')))
                text_memory = float(method_stats[text_method].get('memory_vram_allocated', {}).get('mean', float('inf')))
                
                latent_network = float(method_stats[latent_method].get('network_comm_data_size', {}).get('mean', float('inf')))
                text_network = float(method_stats[text_method].get('network_comm_data_size', {}).get('mean', float('inf')))
                
                # 生成建议
                if latent_compute < text_compute * 0.7:
                    savings_pct = ((text_compute - latent_compute) / text_compute * 100)
                    report['recommendations'].append(
                        f"LatentMAS reduces inference time by {savings_pct:.1f}%, "
                        f"making it ideal for latency-sensitive applications."
                    )
                
                if latent_memory < text_memory * 0.85:
                    savings_pct = ((text_memory - latent_memory) / text_memory * 100)
                    report['recommendations'].append(
                        f"LatentMAS reduces VRAM usage by {savings_pct:.1f}%, "
                        f"enabling deployment on lower-memory GPUs."
                    )
                
                if latent_network < text_network * 0.3:
                    savings_pct = ((text_network - latent_network) / text_network * 100)
                    report['recommendations'].append(
                        f"LatentMAS reduces communication data size by {savings_pct:.1f}%, "
                        f"significantly lowering network bandwidth requirements."
                    )
        
        # 保存报告 - 使用转换函数
        report_path = 'infrastructure_analysis_report.json'
        with open(report_path, 'w') as f:
            # 使用转换函数确保所有数据可序列化
            json.dump(convert_numpy_types(report), f, indent=2, cls=NumpyJSONEncoder)
        
        print("\n" + "="*60)
        print("INFRASTRUCTURE ANALYSIS REPORT")
        print("="*60)
        print(f"Report generated and saved to: {report_path}")
        print(f"Analysis completed with {len(self.data)} methods analyzed")
        print("\nKEY RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        print("="*60)
        
        return convert_numpy_types(report)
    
    def plot_power_comparison(self, methods: List[str] = None):
        """绘制功耗对比图表"""
        if methods is None:
            methods = list(self.data.keys())
        
        if not methods:
            print("No metrics data found. Please run experiments first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Power Consumption Analysis: Multi-Agent Communication Methods', fontsize=16)
        
        self._plot_energy_consumption(methods, axes[0, 0])
        self._plot_power_draw_over_time(methods, axes[0, 1])
        self._plot_energy_efficiency(methods, axes[1, 0])
        self._plot_gpu_cpu_power_breakdown(methods, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('power_consumption_analysis.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    def _plot_energy_consumption(self, methods, ax):
        """绘制总能耗对比"""
        energy_data = {}
        tokens_per_joule = {}
        
        for method in methods:
            total_energy = 0
            total_tokens_per_joule = 0
            count = 0
            
            for exp in self.data[method]:
                if 'power_data' in exp and 'power_summary' in exp['power_data']:
                    summary = exp['power_data']['power_summary']
                    total_energy += summary.get('total_energy_consumed', 0)
                    total_tokens_per_joule += summary.get('avg_tokens_per_joule', 0)
                    count += 1
            
            if count > 0:
                energy_data[method] = total_energy / count
                tokens_per_joule[method] = total_tokens_per_joule / count
        
        if not energy_data:
            ax.text(0.5, 0.5, 'No power data available', ha='center', va='center')
            return
        
        # 创建双柱状图
        x = np.arange(len(methods))
        width = 0.35
        
        # 能耗柱状图
        energy_values = [energy_data.get(method, 0) for method in methods]
        bars1 = ax.bar(x - width/2, energy_values, width, label='Total Energy (J)', color='#ff6b6b', alpha=0.8)
        
        # 设置y轴
        ax.set_ylabel('Energy Consumption (joules)', color='#ff6b6b')
        ax.tick_params(axis='y', labelcolor='#ff6b6b')
        ax.grid(True, alpha=0.3)
        
        # 创建第二个y轴用于tokens per joule
        ax2 = ax.twinx()
        efficiency_values = [tokens_per_joule.get(method, 0) for method in methods]
        bars2 = ax2.bar(x + width/2, efficiency_values, width, label='Tokens/joule', color='#4ecdc4', alpha=0.8)
        
        ax2.set_ylabel('Tokens per joule', color='#4ecdc4')
        ax2.tick_params(axis='y', labelcolor='#4ecdc4')
        
        # 设置标题和标签
        ax.set_xlabel('Communication Method')
        ax.set_title('Energy Consumption vs Energy Efficiency')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 添加数据标签
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                   f'{energy_values[i]:.1f}J', ha='center', va='bottom', color='#ff6b6b')
            ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1,
                    f'{efficiency_values[i]:.1f}', ha='center', va='bottom', color='#4ecdc4')
    
    def _plot_power_draw_over_time(self, methods, ax):
        """绘制功耗随时间变化"""
        for method in methods:
            power_over_time = []
            time_points = []
            
            for exp in self.data[method]:
                if 'power_data' in exp and 'power_metrics' in exp['power_data']:
                    metrics = exp['power_data']['power_metrics']
                    for entry in metrics:
                        power_over_time.append(entry['total_energy_consumed'])
                        time_points.append(entry['timestamp'])
            
            if power_over_time:
                # 采样数据点避免过多
                sample_indices = np.linspace(0, len(power_over_time)-1, 100).astype(int)
                ax.plot([time_points[i] for i in sample_indices], 
                       [power_over_time[i] for i in sample_indices],
                       label=method, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Power Draw (joules)')
        ax.set_title('System Power Draw Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_energy_efficiency(self, methods, ax):
        """绘制能效对比（雷达图）"""
        categories = ['Tokens/Joule', 'Samples/Joule', 'GPU Efficiency', 'CPU Efficiency']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(223, polar=True)
        
        for method in methods:
            values = []
            
            for exp in self.data[method]:
                if 'power_data' in exp and 'power_summary' in exp['power_data']:
                    summary = exp['power_data']['power_summary']
                    
                    tokens_per_joule = summary.get('avg_tokens_per_joule', 0)
                    samples_per_joule = summary.get('avg_samples_per_joule', 0)
                    gpu_energy_fraction = summary.get('gpu_energy_fraction', 0)
                    cpu_energy_fraction = summary.get('cpu_energy_fraction', 0)
                    
                    # 归一化到0-100范围
                    values.append([
                        min(tokens_per_joule / 100 * 100, 100),  # 假设100 tokens/J是很好的效率
                        min(samples_per_joule / 10 * 100, 100),  # 假设10 samples/J是很好的效率
                        gpu_energy_fraction * 100,
                        cpu_energy_fraction * 100
                    ])
            
            if values:
                avg_values = np.mean(values, axis=0).tolist()
                avg_values += avg_values[:1]  # 闭合雷达图
                
                ax.plot(angles, avg_values, 'o-', linewidth=2, label=method)
                ax.fill(angles, avg_values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Energy Efficiency Profile')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    def _plot_gpu_cpu_power_breakdown(self, methods, ax):
        """绘制GPU/CPU功耗占比"""
        gpu_percentages = []
        method_names = []
        
        for method in methods:
            gpu_energy_fractions = []
            
            for exp in self.data[method]:
                if 'power_data' in exp and 'power_summary' in exp['power_data']:
                    summary = exp['power_data']['power_summary']
                    gpu_energy_fractions.append(summary.get('gpu_energy_fraction', 0) * 100)
            
            if gpu_energy_fractions:
                gpu_percentages.append(np.mean(gpu_energy_fractions))
                method_names.append(method)
        
        if not gpu_percentages:
            ax.text(0.5, 0.5, 'No GPU/CPU breakdown data', ha='center', va='center')
            return
        
        # 创建堆叠柱状图
        x = np.arange(len(method_names))
        width = 0.6
        
        # GPU部分
        gpu_bars = ax.bar(x, gpu_percentages, width, label='GPU Power', color='#45b7d1')
        
        # CPU部分
        cpu_percentages = [100 - gpu_pct for gpu_pct in gpu_percentages]
        ax.bar(x, cpu_percentages, width, bottom=gpu_percentages, label='CPU Power', color='#96ceb4')
        
        ax.set_ylabel('Power Consumption Percentage (%)')
        ax.set_title('GPU vs CPU Power Consumption Breakdown')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数据标签
        for i, (gpu_bar, gpu_pct) in enumerate(zip(gpu_bars, gpu_percentages)):
            height = gpu_bar.get_height()
            ax.text(gpu_bar.get_x() + gpu_bar.get_width()/2, height/2,
                   f'GPU: {gpu_pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')
            ax.text(gpu_bar.get_x() + gpu_bar.get_width()/2, height + cpu_percentages[i]/2,
                   f'CPU: {cpu_percentages[i]:.1f}%', ha='center', va='center', color='black', fontweight='bold')
    

    def generate_energy_efficiency_report(self):
        """生成能效优化报告 - 修复NumPy序列化问题"""
        report = {
            'power_summary': {},
            'efficiency_recommendations': [],
            'cost_analysis': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # 分析不同方法的能效
        method_efficiency = {}
        
        for method, experiments in self.data.items():
            if not experiments:
                continue
            
            total_energy = 0.0
            total_samples = 0
            total_tokens = 0
            peak_power = 0.0
            count = 0
            
            for exp in experiments:
                if 'power_data' in exp and 'power_summary' in exp['power_data']:
                    summary = exp['power_data']['power_summary']
                    total_energy += float(summary.get('total_energy_consumed', 0.0))
                    total_samples += len(exp.get('results', []))
                    # 估计处理的token数
                    if 'tokens_processed_total' in exp.get('experiment_config', {}):
                        total_tokens += int(exp['experiment_config']['tokens_processed_total'])
                    peak_power = max(peak_power, float(summary.get('peak_power_draw', 0.0)))
                    count += 1
            
            if count > 0 and total_samples > 0:
                energy_per_sample = total_energy / total_samples
                tokens_per_joule = total_tokens / total_energy if total_energy > 0 else 0.0
                
                method_efficiency[method] = {
                    'energy_per_sample': energy_per_sample,
                    'tokens_per_joule': tokens_per_joule,
                    'peak_power': peak_power,
                    'total_energy': total_energy,
                    'total_samples': total_samples,
                    'total_tokens': total_tokens
                }
        
        # 生成对比和建议
        if method_efficiency:
            # 找出最优方法
            best_method_energy = min(method_efficiency.items(), key=lambda x: x[1]['energy_per_sample'])
            best_method_efficiency = max(method_efficiency.items(), key=lambda x: x[1]['tokens_per_joule'])
            
            # 转换为可序列化格式
            report['power_summary'] = convert_numpy_types(method_efficiency)
            
            # 生成建议
            if 'latent_mas' in method_efficiency and 'text_mas' in method_efficiency:
                latent_energy = float(method_efficiency['latent_mas']['energy_per_sample'])
                text_energy = float(method_efficiency['text_mas']['energy_per_sample'])
                
                energy_savings_pct = (text_energy - latent_energy) / text_energy * 100 if text_energy > 0 else 0
                
                report['efficiency_recommendations'].append(
                    f"LatentMAS reduces energy consumption per sample by {energy_savings_pct:.1f}% compared to Text-MAS"
                )
                
                # 估算成本节省
                electricity_cost_per_kwh = 0.12  # $0.12 per kWh (美国平均)
                samples_per_day = 1000000  # 假设每天100万样本
                
                text_daily_energy_kwh = (text_energy * samples_per_day) / 3600000  # J to kWh
                latent_daily_energy_kwh = (latent_energy * samples_per_day) / 3600000
                
                text_daily_cost = text_daily_energy_kwh * electricity_cost_per_kwh
                latent_daily_cost = latent_daily_energy_kwh * electricity_cost_per_kwh
                
                daily_savings = text_daily_cost - latent_daily_cost
                yearly_savings = daily_savings * 365
                
                report['cost_analysis'] = {
                    'electricity_cost_per_kwh': electricity_cost_per_kwh,
                    'samples_per_day': samples_per_day,
                    'text_daily_cost': text_daily_cost,
                    'latent_daily_cost': latent_daily_cost,
                    'daily_savings': daily_savings,
                    'yearly_savings': yearly_savings,
                    'energy_savings_pct': energy_savings_pct
                }
                
                report['efficiency_recommendations'].append(
                    f"At scale (1M samples/day), LatentMAS saves ${daily_savings:.2f}/day or ${yearly_savings:.2f}/year in electricity costs"
                )
                
                # 硬件要求分析
                if 'peak_power' in method_efficiency['latent_mas'] and 'peak_power' in method_efficiency['text_mas']:
                    latent_peak_power = float(method_efficiency['latent_mas']['peak_power'])
                    text_peak_power = float(method_efficiency['text_mas']['peak_power'])
                    
                    power_reduction_pct = (text_peak_power - latent_peak_power) / text_peak_power * 100 if text_peak_power > 0 else 0
                    
                    report['efficiency_recommendations'].append(
                        f"LatentMAS reduces peak power requirements by {power_reduction_pct:.1f}%, enabling deployment on lower-power infrastructure"
                    )
        
        # 保存报告 - 使用转换函数
        report_path = 'energy_efficiency_report.json'
        with open(report_path, 'w') as f:
            json.dump(convert_numpy_types(report), f, indent=2)
        
        print("\n" + "="*60)
        print("ENERGY EFFICIENCY ANALYSIS REPORT")
        print("="*60)
        
        if report['efficiency_recommendations']:
            print("\nKEY RECOMMENDATIONS:")
            for i, rec in enumerate(report['efficiency_recommendations'], 1):
                print(f"{i}. {rec}")
        
        if report['cost_analysis']:
            cost = report['cost_analysis']
            print(f"\nCOST SAVINGS AT SCALE:")
            print(f"  Daily savings: ${cost['daily_savings']:.2f}")
            print(f"  Yearly savings: ${cost['yearly_savings']:.2f}")
            print(f"  Power reduction: {cost['energy_savings_pct']:.1f}%")
        
        print("="*60)
        
        return convert_numpy_types(report)