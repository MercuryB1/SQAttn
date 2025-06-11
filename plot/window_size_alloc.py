import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 示例数据：layer数为4，head数为8
# 你可以用自己的 window size 替换这个二维数组
window_sizes = [
    [16, 32, 64, 64, 128, 128, 256, 256],   # layer 0
    [16, 32, 64, 64, 128, 128, 256, 256],   # layer 1
    [32, 64, 64, 64, 128, 128, 256, 512],   # layer 2
    [64, 64, 64, 128, 128, 256, 512, 512],  # layer 3
]

# 转换为 NumPy 数组
data = np.array(window_sizes)

# 创建 heatmap 图
plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    data,
    annot=True, fmt="d", linewidths=.5, cmap="YlGnBu", cbar_kws={"label": "Window Size"}
)

# 设置坐标轴
ax.set_xlabel("Head ID", fontsize=12)
ax.set_ylabel("Layer ID", fontsize=12)
ax.set_title("Window Size per Head per Layer", fontsize=14)

# y 轴顺序从上到下为 0->N（可根据需要倒序）
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("window_size_alloc.png")
plt.show()
