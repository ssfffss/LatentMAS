# 计算优化建议图表
import matplotlib.pyplot as plt
import numpy as np

methods = ['Text-MAS', 'LatentMAS']
gpu_util = [78, 65]  # %
inference_time = [2.8, 1.2]  # seconds
flops_efficiency = [45, 72]  # GFLOPS per watt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# GPU利用率对比
x = np.arange(len(methods))
width = 0.35
ax1.bar(x, gpu_util, width, color=['#ff6b6b', '#4ecdc4'])
ax1.set_ylabel('GPU Utilization (%)')
ax1.set_title('GPU Utilization by Communication Method')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.grid(True, alpha=0.3)

# 能效对比
ax2.bar(x, flops_efficiency, width, color=['#ff6b6b', '#4ecdc4'])
ax2.set_ylabel('Computational Efficiency (GFLOPS/Watt)')
ax2.set_title('Energy Efficiency Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('compute_optimization.png', dpi=300)
plt.show()