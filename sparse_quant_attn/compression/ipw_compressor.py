import torch
import numpy as np
from typing import Dict, Tuple, Union
from loguru import logger
import matplotlib.pyplot as plt
import os


class HybridIPWCompressor:
    """
    注意力压缩的混合逆概率加权（Hybrid IPW）算法
    
    核心思想：一个注意力连接的"真实价值"由两部分相乘决定：
    1. 结构价值: 与观察者（Query）的距离d，代表其固有的"全局性"
    2. 统计价值: 该距离d在该头自身行为中出现的概率的倒数，代表其"稀有性"或"意外性"
    """
    
    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-9, 
                 preservation_threshold: float = 0.99):
        """
        初始化Hybrid IPW压缩器
        
        Args:
            alpha: 非局部性溢价，控制"距离"本身的重要性 (推荐: 0.01-0.1)
            epsilon: 数值稳定常数，防止除零 (推荐: 1e-9)
            preservation_threshold: 加权信息保留阈值 (推荐: 0.99)
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.preservation_threshold = preservation_threshold
        
        logger.info(f"初始化Hybrid IPW压缩器:")
        logger.info(f"  - 非局部性溢价α: {self.alpha}")
        logger.info(f"  - 数值稳定常数ε: {self.epsilon}")
        logger.info(f"  - 保留阈值: {self.preservation_threshold}")
    
    def compute_single_head_window(self, attention_tensor: Union[torch.Tensor, np.ndarray],
                                 layer_idx: int, head_idx: int, 
                                 visualize: bool = False) -> int:
        """
        为单个注意力头计算最优窗口大小
        
        Args:
            attention_tensor: 形状为 (N, L, L) 的注意力张量
            layer_idx: 层索引
            head_idx: 头索引
            visualize: 是否可视化分析过程
            
        Returns:
            W_p_h_star: 最优窗口大小
        """
        logger.info(f"\n处理 Layer {layer_idx} Head {head_idx}")
        
        # 转换为numpy以便处理
        if isinstance(attention_tensor, torch.Tensor):
            attention_tensor = attention_tensor.detach().cpu().numpy()
        
        N, L, _ = attention_tensor.shape
        logger.info(f"  输入形状: {attention_tensor.shape}")
        
        # 步骤1: 计算平均注意力图谱 A_avg_h
        A_avg_h = np.mean(attention_tensor, axis=0)  # (L, L)
        logger.info(f"  步骤1: 计算平均注意力图谱完成")
        
        # 步骤2: 计算经验距离概率分布 P_h(d)
        H_h = np.zeros(L)  # 距离直方图
        
        for i in range(L):
            for j in range(L):
                if i >= j:  # 因果约束
                    d = i - j
                    H_h[d] += A_avg_h[i, j]
        
        E_total_h = np.sum(H_h)
        P_h = H_h / (E_total_h + self.epsilon)  # 归一化为概率分布
        
        logger.info(f"  步骤2: 距离概率分布计算完成")
        logger.info(f"    总能量: {E_total_h:.6f}")
        logger.info(f"    距离0-4的概率: {P_h[:5]}")
        
        # 步骤3: 计算最终的混合权重函数 W_final(d)
        W_final = np.zeros(L)
        
        for d in range(L):
            # 结构权重: 距离越远，权重越高
            W_structural = 1.0 + self.alpha * d
            
            # 统计权重: 概率越小（越稀有），权重越高
            W_statistical = 1.0 / (P_h[d] + self.epsilon)
            
            # 混合权重: 结构 × 统计
            W_final[d] = W_structural * W_statistical
        
        logger.info(f"  步骤3: 混合权重函数计算完成")
        logger.info(f"    距离0-4的权重: {W_final[:5]}")
        
        # 步骤4: 计算每个对角线的总加权能量
        E_weighted_per_diagonal = np.zeros(L)
        
        for i in range(L):
            for j in range(L):
                if i >= j:  # 因果约束
                    d = i - j
                    E_weighted_per_diagonal[d] += A_avg_h[i, j] * W_final[d]
        
        E_weighted_total = np.sum(E_weighted_per_diagonal)
        
        logger.info(f"  步骤4: 加权能量计算完成")
        logger.info(f"    总加权能量: {E_weighted_total:.6f}")
        
        # 步骤5: 搜索满足阈值的最小窗口 W_p_h*
        E_weighted_preserved = 0.0
        W_p_h_star = L - 1  # 默认最大窗口
        
        for W_p in range(L):
            E_weighted_preserved += E_weighted_per_diagonal[W_p]
            preservation_ratio = E_weighted_preserved / E_weighted_total
            
            if preservation_ratio >= self.preservation_threshold:
                W_p_h_star = W_p
                logger.info(f"  步骤5: 找到最优窗口 W_p*={W_p_h_star}")
                logger.info(f"    保留比例: {preservation_ratio:.6f}")
                break
        
        # 可视化分析过程
        if visualize:
            self.visualize_analysis(A_avg_h, P_h, W_final, E_weighted_per_diagonal,
                                  W_p_h_star, layer_idx, head_idx)
        
        return W_p_h_star
    
    def compute_all_windows(self, attention_maps: Dict[Tuple[int, int], Union[torch.Tensor, np.ndarray]],
                          visualize_sample: bool = False) -> Dict[Tuple[int, int], int]:
        """
        为所有注意力头计算最优窗口大小
        
        Args:
            attention_maps: 注意力图谱字典 {(layer_idx, head_idx): tensor(N, L, L)}
            visualize_sample: 是否对样本头进行可视化
            
        Returns:
            quantization_windows: 最优窗口字典 {(layer_idx, head_idx): W_p_h_star}
        """
        logger.info(f"开始处理 {len(attention_maps)} 个注意力头")
        
        quantization_windows = {}
        sample_visualized = False
        
        for (layer_idx, head_idx), attn_tensor in attention_maps.items():
            # 是否对第一个头进行可视化示例
            should_visualize = visualize_sample and not sample_visualized
            
            W_p_h_star = self.compute_single_head_window(
                attn_tensor, layer_idx, head_idx, visualize=should_visualize
            )
            
            quantization_windows[(layer_idx, head_idx)] = W_p_h_star
            
            if should_visualize:
                sample_visualized = True
        
        # 输出统计信息
        self.print_statistics(quantization_windows)
        
        return quantization_windows
    
    def visualize_analysis(self, A_avg_h: np.ndarray, P_h: np.ndarray, 
                          W_final: np.ndarray, E_weighted_per_diagonal: np.ndarray,
                          W_p_h_star: int, layer_idx: int, head_idx: int):
        """
        可视化IPW分析过程
        """
        L = len(P_h)
        distances = np.arange(L)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 平均注意力图谱
        im1 = ax1.imshow(A_avg_h, cmap='Blues', aspect='auto')
        ax1.set_title(f'Layer {layer_idx} Head {head_idx}: 平均注意力图谱')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # 子图2: 距离概率分布 P_h(d)
        ax2.plot(distances, P_h, 'g-', linewidth=2, label='P_h(d)')
        ax2.set_title('距离概率分布 P_h(d)')
        ax2.set_xlabel('Distance d')
        ax2.set_ylabel('Probability')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 子图3: 混合权重函数 W_final(d)
        ax3.plot(distances, W_final, 'r-', linewidth=2, label='W_final(d)')
        ax3.set_title('混合权重函数 W_final(d)')
        ax3.set_xlabel('Distance d')
        ax3.set_ylabel('Weight')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')  # 使用对数尺度，因为权重可能变化很大
        
        # 子图4: 每个对角线的加权能量 + 最优窗口
        ax4.bar(distances, E_weighted_per_diagonal, alpha=0.7, label='Weighted Energy')
        ax4.axvline(x=W_p_h_star, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal Window: {W_p_h_star}')
        
        # 计算累积保留比例
        cumulative_energy = np.cumsum(E_weighted_per_diagonal)
        total_energy = cumulative_energy[-1]
        cumulative_ratio = cumulative_energy / total_energy
        
        ax4_twin = ax4.twinx()
        ax4_twin.plot(distances, cumulative_ratio, 'orange', linewidth=2, 
                     label='Cumulative Preservation Ratio')
        ax4_twin.axhline(y=self.preservation_threshold, color='orange', linestyle=':', 
                        alpha=0.7, label=f'Threshold: {self.preservation_threshold}')
        
        ax4.set_title('对角线加权能量 & 最优窗口')
        ax4.set_xlabel('Distance d (Window Size)')
        ax4.set_ylabel('Weighted Energy')
        ax4_twin.set_ylabel('Cumulative Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('ipw_analysis', exist_ok=True)
        save_path = f'ipw_analysis/layer_{layer_idx}_head_{head_idx}_ipw_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"IPW分析可视化已保存到: {save_path}")
        
        plt.show()
        plt.close()
    
    def print_statistics(self, quantization_windows: Dict[Tuple[int, int], int]):
        """
        打印窗口大小统计信息
        """
        windows = list(quantization_windows.values())
        
        logger.info(f"\n=== IPW压缩统计结果 ===")
        logger.info(f"处理的头数量: {len(windows)}")
        logger.info(f"窗口大小统计:")
        logger.info(f"  - 最小: {min(windows)}")
        logger.info(f"  - 最大: {max(windows)}")
        logger.info(f"  - 平均: {np.mean(windows):.2f}")
        logger.info(f"  - 中位数: {np.median(windows):.2f}")
        logger.info(f"  - 标准差: {np.std(windows):.2f}")
        
        # 窗口大小分布
        unique_windows, counts = np.unique(windows, return_counts=True)
        logger.info(f"窗口大小分布:")
        for window, count in zip(unique_windows, counts):
            percentage = count / len(windows) * 100
            logger.info(f"  - W={window:3d}: {count:3d} heads ({percentage:5.1f}%)")

# 使用示例和集成函数
def collect_attention_maps_ipw(model, dataloader, num_samples: int = 10) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    收集注意力图谱用于IPW分析
    
    Args:
        model: 模型
        dataloader: 数据加载器
        num_samples: 收集的样本数量
        
    Returns:
        attention_maps: {(layer_idx, head_idx): tensor(num_samples, seq_len, seq_len)}
    """
    logger.info(f"开始收集 {num_samples} 个样本的注意力图谱")
    
    attention_maps = {}
    model.eval()
    
    with torch.no_grad():
        for sample_idx, batch in enumerate(dataloader):
            if sample_idx >= num_samples:
                break
                
            # 运行模型并收集注意力
            # 这里需要根据你的模型实现来调整
            # 假设使用之前的attention捕获方法
            
            # 为每层每头收集注意力
            for layer_idx in range(len(model.layers)):
                for head_idx in range(model.layers[layer_idx].self_attn.config.num_attention_heads):
                    # 获取attention weights (这里需要你的实现)
                    # attention_weights = get_attention_for_head(model, batch, layer_idx, head_idx)
                    
                    key = (layer_idx, head_idx)
                    if key not in attention_maps:
                        # 初始化存储
                        seq_len = batch['input_ids'].size(1)  # 根据实际情况调整
                        attention_maps[key] = torch.zeros(num_samples, seq_len, seq_len)
                    
                    # 存储当前样本的attention
                    # attention_maps[key][sample_idx] = attention_weights
            
            if sample_idx % 5 == 0:
                logger.info(f"已收集 {sample_idx + 1}/{num_samples} 个样本")
    
    logger.info(f"注意力图谱收集完成，共 {len(attention_maps)} 个头")
    return attention_maps

def apply_ipw_compression(model, dataloader, args, 
                         alpha: float = 0.01, 
                         preservation_threshold: float = 0.99,
                         num_calibration_samples: int = 10):
    """
    应用IPW压缩到模型
    
    Args:
        model: 模型
        dataloader: 校准数据
        args: 参数
        alpha: 非局部性溢价
        preservation_threshold: 保留阈值
        num_calibration_samples: 校准样本数量
        
    Returns:
        quantization_windows: 最优窗口配置
    """
    logger.info(f"开始IPW压缩分析")
    
    # 1. 收集注意力图谱
    attention_maps = collect_attention_maps_ipw(model, dataloader, num_calibration_samples)
    
    # 2. 创建IPW压缩器
    ipw_compressor = HybridIPWCompressor(
        alpha=alpha,
        epsilon=1e-9,
        preservation_threshold=preservation_threshold
    )
    
    # 3. 计算所有头的最优窗口
    quantization_windows = ipw_compressor.compute_all_windows(
        attention_maps, 
        visualize_sample=True  # 对第一个头进行可视化示例
    )
    
    return quantization_windows

# 转换为你需要的格式
def convert_ipw_windows_to_arrays(quantization_windows: Dict[Tuple[int, int], int], 
                                layer_idx: int, num_heads: int) -> Tuple[list, list]:
    """
    将IPW结果转换为指定层的窗口数组
    
    Args:
        quantization_windows: IPW结果
        layer_idx: 目标层索引
        num_heads: 头数量
        
    Returns:
        bit8_window_sizes: 8-bit窗口大小列表
        bit4_window_sizes: 4-bit窗口大小列表 (这里设为与8-bit相同或稍大)
    """
    bit8_window_sizes = []
    bit4_window_sizes = []
    
    for head_id in range(num_heads):
        key = (layer_idx, head_id)
        if key in quantization_windows:
            w_optimal = quantization_windows[key]
            bit8_window_sizes.append(w_optimal)
            # 4-bit窗口设为8-bit的1.5倍，或者根据需要调整
            bit4_window_sizes.append(int(w_optimal * 1.5))
        else:
            # 默认值
            bit8_window_sizes.append(128)
            bit4_window_sizes.append(256)
    
    return bit8_window_sizes, bit4_window_sizes