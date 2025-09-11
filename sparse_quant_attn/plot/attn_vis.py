import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch.nn.functional as F


def visualize_attention_weights(attn_weights, layer_idx, save_dir="attention_vis", 
                              colormap='viridis', save_format='png', dpi=150, 
                              figsize=(10, 8), show_values=False, threshold=None):
    """
    可视化注意力权重并按层/头索引保存
    
    Args:
        attn_weights: torch.Tensor, 形状为 [num_heads, seq_len, seq_len] 或 [batch, num_heads, seq_len, seq_len]
        layer_idx: int, 层索引
        save_dir: str, 保存目录
        colormap: str, 颜色映射 ('viridis', 'hot', 'Blues', 'RdYlBu_r' 等)
        save_format: str, 图片格式 ('png', 'pdf', 'svg', 'jpg')
        dpi: int, 图片分辨率
        figsize: tuple, 图片大小
        show_values: bool, 是否在热力图上显示数值
        threshold: float, 阈值，小于该值的权重不显示
    """
    
    # 转换为 numpy 并处理维度
    if isinstance(attn_weights, torch.Tensor):
        weights = attn_weights.detach().cpu().float().numpy()
    else:
        weights = attn_weights
    
    # 如果有 batch 维度，取第一个样本
    if len(weights.shape) == 4:  # [batch, heads, seq, seq]
        weights = weights[0]  # [heads, seq, seq]
    
    num_heads, seq_len, _ = weights.shape
    
    # 创建保存目录
    layer_dir = Path(save_dir) / f"layer_{layer_idx}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个头生成可视化
    for head_idx in range(num_heads):
        head_dir = layer_dir / f"head_{head_idx}"
        head_dir.mkdir(exist_ok=True)
        
        # 获取当前头的注意力权重
        head_weights = weights[head_idx]
        
        # 应用阈值
        if threshold is not None:
            head_weights = np.where(head_weights > threshold, head_weights, 0)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0]*2, figsize[1]*2))
        fig.suptitle(f'Layer {layer_idx} - Head {head_idx} Attention Weights', fontsize=16)
        
        # 1. 完整热力图
        ax1 = axes[0, 0]
        im1 = ax1.imshow(head_weights, cmap=colormap, aspect='auto')
        ax1.set_title('Full Attention Matrix')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        if show_values and seq_len <= 20:  # 只在序列长度较短时显示数值
            for i in range(seq_len):
                for j in range(seq_len):
                    text = ax1.text(j, i, f'{head_weights[i, j]:.2f}',
                                   ha="center", va="center", color="white" if head_weights[i, j] > 0.5 else "black")
        
        # 2. 注意力模式分析（下三角，体现因果掩码）
        ax2 = axes[0, 1]
        causal_weights = np.tril(head_weights)  # 只显示下三角
        im2 = ax2.imshow(causal_weights, cmap=colormap, aspect='auto')
        ax2.set_title('Causal Attention Pattern')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. 注意力权重分布直方图
        ax3 = axes[1, 0]
        non_zero_weights = head_weights[head_weights > (threshold or 0)]
        ax3.hist(non_zero_weights.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Attention Weight')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Attention Weight Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Mean: {non_zero_weights.mean():.4f}\nStd: {non_zero_weights.std():.4f}\nMax: {non_zero_weights.max():.4f}\nSparsity: {((head_weights == 0).sum() / head_weights.size * 100):.1f}%'
        ax3.text(0.7, 0.7, stats_text, transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 4. 行求和（每个 query 的总注意力）
        ax4 = axes[1, 1]
        row_sums = head_weights.sum(axis=1)
        ax4.plot(range(seq_len), row_sums, marker='o', linewidth=2, markersize=4)
        ax4.set_xlabel('Query Position')
        ax4.set_ylabel('Total Attention')
        ax4.set_title('Attention Sum per Query')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, max(row_sums) * 1.1)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = head_dir / f"attention_layer{layer_idx}_head{head_idx}.{save_format}"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 额外保存一个简单的热力图
        plt.figure(figsize=figsize)
        sns.heatmap(head_weights, cmap=colormap, cbar=True, square=True,
                   xticklabels=False, yticklabels=False)
        plt.title(f'Layer {layer_idx} Head {head_idx} - Attention Heatmap')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        simple_save_path = head_dir / f"simple_heatmap_L{layer_idx}H{head_idx}.{save_format}"
        plt.savefig(simple_save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
    print(f"✅ Layer {layer_idx} visualization saved to {layer_dir}")
    return layer_dir


def visualize_all_heads_summary(attn_weights, layer_idx, save_dir="attention_vis", 
                               colormap='viridis', figsize=(15, 10)):
    """
    生成一个总览图，显示所有注意力头
    """
    if isinstance(attn_weights, torch.Tensor):
        weights = attn_weights.detach().cpu().float().numpy()
    else:
        weights = attn_weights
    
    if len(weights.shape) == 4:
        weights = weights[0]
    
    num_heads, seq_len, _ = weights.shape
    
    # 计算网格大小
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Layer {layer_idx} - All Attention Heads Summary', fontsize=16)
    
    for head_idx in range(num_heads):
        row = head_idx // cols
        col = head_idx % cols
        
        im = axes[row, col].imshow(weights[head_idx], cmap=colormap, aspect='auto')
        axes[row, col].set_title(f'Head {head_idx}')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        # 添加颜色条到每个子图
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for head_idx in range(num_heads, rows * cols):
        row = head_idx // cols
        col = head_idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存总览图
    summary_dir = Path(save_dir) / f"layer_{layer_idx}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"all_heads_summary_layer{layer_idx}.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Layer {layer_idx} summary saved to {summary_path}")
    return summary_path
    