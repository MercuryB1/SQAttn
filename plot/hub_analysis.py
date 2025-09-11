import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = {
    'Head': [1, 11, 0, 6, 10, 9, 8, 3, 5, 7, 4, 2],
    'PPL_rand': [51.0153, 51.0276, 51.0276, 51.0196, 51.0276, 51.0189, 51.0276, 51.0316, 51.0276, 51.0276, 51.0276, 51.0115],
    'Δrand%': [0.50, 0.52, 0.52, 0.51, 0.52, 0.50, 0.52, 0.53, 0.52, 0.52, 0.52, 0.49],
    'PPL_hub': [51.6707, 51.4342, 51.2519, 51.2268, 51.2200, 51.1219, 51.1043, 51.0883, 51.0707, 51.0451, 51.0391, 50.8803],
    'Δhub%': [1.79, 1.32, 0.96, 0.91, 0.90, 0.71, 0.67, 0.64, 0.61, 0.56, 0.54, 0.23],
    'PPL_baseline': [50.7627] * 12
}

df = pd.DataFrame(data)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表和双轴
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 8))
ax2 = ax1.twinx()

# 在主轴（左轴）上绘制条形图 - 百分比变化
x = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x - width/2, df['Δrand%'], width, label='Δrand%', alpha=0.7, color='skyblue')
bars2 = ax1.bar(x + width/2, df['Δhub%'], width, label='Δhub%', alpha=0.7, color='lightcoral')

# 在次轴（右轴）上绘制折线图 - PPL值
line1 = ax2.plot(x, df['PPL_rand'], 'o-', label='PPL_rand', linewidth=2, markersize=6, color='blue')
line2 = ax2.plot(x, df['PPL_hub'], 's-', label='PPL_hub', linewidth=2, markersize=6, color='red')
line3 = ax2.axhline(y=df['PPL_baseline'][0], color='green', linestyle='--', label='PPL_baseline', linewidth=2)

# 设置左轴（条形图）
ax1.set_xlabel('Head', fontsize=12)
ax1.set_ylabel('Degradation Percentage (%)', fontsize=12, color='black')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Head'])
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

# 设置右轴（折线图）
ax2.set_ylabel('PPL Value', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 添加数值标签到条形图上
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
             f'{height1:.2f}', ha='center', va='bottom', fontsize=8)
    ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.02,
             f'{height2:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# 设置标题
ax1.set_title('PPL Values and Degradation Percentage Analysis', fontsize=12, fontweight='bold', pad=20)

# 第二个子图：按Δhub%排序的图
sorted_df = df.sort_values('Δhub%', ascending=True)  # 从低到高排序便于显示
y_pos = np.arange(len(sorted_df))

# 创建水平条形图
bars = ax3.barh(y_pos, sorted_df['Δhub%'], color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_df))), 
                alpha=0.8, edgecolor='black', linewidth=0.5)

# 设置标签和标题
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f'Head {h}' for h in sorted_df['Head']])
ax3.set_xlabel('Δhub% (Degradation Percentage)', fontsize=12)
ax3.set_title('Heads Ranked by Δhub% Degradation', fontsize=12, fontweight='bold')

# 在柱子上添加数值标签
for i, (bar, value) in enumerate(zip(bars, sorted_df['Δhub%'])):
    width = bar.get_width()
    ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{value:.2f}%', ha='left', va='center', fontweight='bold', fontsize=10)

# 添加网格
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim(0, max(sorted_df['Δhub%']) * 1.15)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('ppl_combined_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('ppl_combined_analysis.pdf', bbox_inches='tight', facecolor='white')
print("Chart saved as ppl_combined_analysis.png and ppl_combined_analysis.pdf")



plt.show()

# 打印统计摘要
print("\n=== Statistical Summary ===")
sorted_df = df.sort_values('Δhub%', ascending=False)
print(f"Best performing Head: {sorted_df.iloc[0]['Head']} (Δhub% = {sorted_df.iloc[0]['Δhub%']:.2f}%)")
print(f"Worst performing Head: {sorted_df.iloc[-1]['Head']} (Δhub% = {sorted_df.iloc[-1]['Δhub%']:.2f}%)")
print(f"Average Δhub%: {df['Δhub%'].mean():.2f}%")
print(f"Δhub% Standard Deviation: {df['Δhub%'].std():.2f}%")
print(f"PPL_hub Range: {df['PPL_hub'].min():.4f} - {df['PPL_hub'].max():.4f}")