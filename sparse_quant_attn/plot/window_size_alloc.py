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


def plot_window_size_alloc(bits_per_head: dict, save_path: str):
    """
    绘制每层每个 head 的平均比特数热图。

    参数：
        bits_per_head: dict[int, List[float]]
            每一层的 head 分配的 bit 数，key 是 layer id，value 是 head 的 float list。
        save_path: str
            图片保存路径（例如 'bit_alloc_heatmap.png'）。
    """
    # 按照 layer 顺序排序
    sorted_layers = sorted(bits_per_head.keys())
    data = [bits_per_head[layer] for layer in sorted_layers]
    data = np.array(data)  # shape: (num_layers, num_heads)

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        data,
        annot=True, fmt=".2f", linewidths=.5, cmap="YlOrRd", cbar_kws={"label": "Bits"},
    )

    ax.set_xlabel("Head ID", fontsize=12)
    ax.set_ylabel("Layer ID", fontsize=12)
    ax.set_title("Per-Head Bit Allocation Heatmap", fontsize=14)

    # 设置 ytick label 为实际的 layer id
    ax.set_yticks(np.arange(len(sorted_layers)) + 0.5)
    ax.set_yticklabels(sorted_layers)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()