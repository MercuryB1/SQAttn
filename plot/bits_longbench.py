import matplotlib.pyplot as plt
import numpy as np

# 设置高质量绘图参数
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['font.size'] = 12

# 数据定义
# MSE数据
mse_bit = [7.55, 7.39, 7.11, 6.89, 6.54, 6.2, 5.88, 5.45, 5.06, 4.98, 4.72, 4.46]
mse_acceleration = [1.325, 1.352, 1.402, 1.443, 1.513, 1.585, 1.658, 1.762, 1.863, 1.884, 1.956, 2.031]
mse_longbench = [30.46, 30.41, 30.4, 30.22, 29.99, 29.68, 28.24, 25.44, 20.18, 15.42, 6.54, 5.99]

# IPW数据
ipw_bit = [7.48, 7.29, 6.85, 6.47, 6.09, 5.91, 5.64, 5.11]
ipw_acceleration = [1.337, 1.369, 1.451, 1.528, 1.608, 1.648, 1.707, 1.857]
ipw_longbench = [30.39, 30.37, 30.41, 30.25, 30.21, 29.99, 27.72, 23.48]

# 基准模型
full_score = 30.38
full_acceleration = 1.0
sageattn_score = 29.93
sageattn_acceleration = 1.355

# 创建双轴图表 - 横轴加速比，左轴LongBench，右轴bit数
fig, ax1 = plt.subplots(figsize=(16, 10), dpi=120)
ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴

# 配色方案
mse_color = '#2563eb'  # 蓝色
ipw_color = '#dc2626'  # 红色

# === 主轴：LongBench性能 (加速比 vs LongBench) ===
# MSE 性能曲线
line1 = ax1.plot(mse_acceleration, mse_longbench, 
                color=mse_color, linewidth=4, marker='o', markersize=10,
                markerfacecolor='white', markeredgecolor=mse_color, markeredgewidth=3,
                label='MSE LongBench', zorder=3)

# IPW 性能曲线
line2 = ax1.plot(ipw_acceleration, ipw_longbench, 
                color=ipw_color, linewidth=4, marker='^', markersize=10,
                markerfacecolor='white', markeredgecolor=ipw_color, markeredgewidth=3,
                label='IPW LongBench', zorder=3)

# === 右轴：Bit数 (加速比 vs Bit数) ===
# MSE bit数曲线
line3 = ax2.plot(mse_acceleration, mse_bit, 
                color=mse_color, linewidth=3, marker='s', markersize=8,
                markerfacecolor=mse_color, markeredgecolor='white', markeredgewidth=2,
                label='MSE Bit Precision', linestyle='--', alpha=0.7, zorder=2)

# IPW bit数曲线
line4 = ax2.plot(ipw_acceleration, ipw_bit, 
                color=ipw_color, linewidth=3, marker='D', markersize=8,
                markerfacecolor=ipw_color, markeredgecolor='white', markeredgewidth=2,
                label='IPW Bit Precision', linestyle='--', alpha=0.7, zorder=2)

# 基准线
ax1.axhline(y=full_score, color='#059669', linestyle='-', linewidth=3, alpha=0.8,
           label=f'FP Full ({full_score})')
ax1.axhline(y=sageattn_score, color='#7c3aed', linestyle='-', linewidth=3, alpha=0.8,
           label=f'SageAttn ({sageattn_score})')

ax1.axvline(x=sageattn_acceleration, color='#7c3aed', linestyle=':', linewidth=2, alpha=0.6)

# === 图表设置 ===
# 横轴设置 (加速比)
ax1.set_xlabel('Acceleration Factor (×)', fontsize=16, fontweight='bold', color='#374151')
ax1.set_xlim(1.2, 2.1)

# 左轴设置 (LongBench)
ax1.set_ylabel('LongBench Score', fontsize=16, fontweight='bold', color='#374151')
ax1.set_ylim(5, 32)
ax1.tick_params(axis='y', labelcolor='#374151', labelsize=12)

# 右轴设置 (Bit数)
ax2.set_ylabel('Bit Precision', fontsize=16, fontweight='bold', color='#6b7280')
ax2.set_ylim(4.2, 7.8)
ax2.tick_params(axis='y', labelcolor='#6b7280', labelsize=12)

# 主标题
fig.suptitle('Quantization Analysis: Acceleration vs Performance & Bit Precision', 
            fontsize=20, fontweight='bold', color='#1f2937', y=0.95)

# === 关键点标注 ===
# MSE 最佳性能点
mse_best_idx = mse_longbench.index(max(mse_longbench))
ax1.annotate(f'MSE Peak\n{max(mse_longbench):.1f} @ {mse_acceleration[mse_best_idx]:.2f}×', 
            xy=(mse_acceleration[mse_best_idx], mse_longbench[mse_best_idx]),
            xytext=(mse_acceleration[mse_best_idx]+0.15, mse_longbench[mse_best_idx]+2),
            arrowprops=dict(arrowstyle='->', color=mse_color, lw=2),
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#dbeafe', alpha=0.9, edgecolor=mse_color))

# IPW 最佳性能点
ipw_best_idx = ipw_longbench.index(max(ipw_longbench))
ax1.annotate(f'IPW Peak\n{max(ipw_longbench):.1f} @ {ipw_acceleration[ipw_best_idx]:.2f}×', 
            xy=(ipw_acceleration[ipw_best_idx], ipw_longbench[ipw_best_idx]),
            xytext=(ipw_acceleration[ipw_best_idx]-0.1, ipw_longbench[ipw_best_idx]+1.5),
            arrowprops=dict(arrowstyle='->', color=ipw_color, lw=2),
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#fee2e2', alpha=0.9, edgecolor=ipw_color))

# 最大加速点
mse_max_acc_idx = mse_acceleration.index(max(mse_acceleration))
ax1.annotate(f'Max Accel\n{max(mse_acceleration):.2f}×\nScore: {mse_longbench[mse_max_acc_idx]:.1f}', 
            xy=(max(mse_acceleration), mse_longbench[mse_max_acc_idx]),
            xytext=(max(mse_acceleration)-0.15, mse_longbench[mse_max_acc_idx]+3),
            arrowprops=dict(arrowstyle='->', color=mse_color, lw=2),
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#f0f9ff', alpha=0.9, edgecolor=mse_color))

# === 基准标注 ===
ax1.text(1.22, full_score + 0.5, f'FP Full\n{full_score}', 
        fontsize=11, fontweight='bold', color='#059669', ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#d1fae5', alpha=0.9, edgecolor='#059669'))

ax1.text(sageattn_acceleration + 0.02, sageattn_score + 0.5, f'SageAttn\n{sageattn_score}', 
        fontsize=11, fontweight='bold', color='#7c3aed', ha='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='#ede9fe', alpha=0.9, edgecolor='#7c3aed'))

# === 性能区域标注 ===
ax1.axhspan(30.0, 32, alpha=0.1, color='#059669')
ax1.axhspan(25.0, 30.0, alpha=0.1, color='#f59e0b')
ax1.axhspan(5, 25.0, alpha=0.1, color='#ef4444')

ax1.text(2.05, 31, 'Excellent', fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#d1fae5', alpha=0.8))
ax1.text(2.05, 27.5, 'Good', fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#fef3c7', alpha=0.8))
ax1.text(2.05, 15, 'Poor', fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#fecaca', alpha=0.8))

# === 网格和样式 ===
ax1.grid(True, linestyle='-', alpha=0.3, color='#9ca3af')
ax1.set_axisbelow(True)

# 美化坐标轴
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#d1d5db')
ax1.spines['top'].set_visible(False)

for spine in ax2.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#d1d5db')
ax2.spines['top'].set_visible(False)

# === 图例 ===
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11, 
          frameon=True, fancybox=True, shadow=True, framealpha=0.95)

# === 简洁的关键信息 ===
info_text = f"""Key Findings:
• MSE: Peak {max(mse_longbench):.1f} @ {mse_acceleration[mse_best_idx]:.2f}× accel
• IPW: Peak {max(ipw_longbench):.1f} @ {ipw_acceleration[ipw_best_idx]:.2f}× accel  
• Max acceleration: {max(mse_acceleration):.2f}× (MSE)
• Both exceed baselines at low acceleration"""

ax1.text(0.02, 0.5, info_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='#f8fafc', alpha=0.95, 
                 edgecolor='#64748b', linewidth=1))

plt.tight_layout()
plt.show()

# 可选：保存高质量图片
plt.savefig('unified_quantization_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='png')