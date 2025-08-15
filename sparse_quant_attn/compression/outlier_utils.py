import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Set, Dict
import matplotlib.pyplot as plt
from sparse_quant_attn.compression.attn_replacer import replace_sdpa_for_block_with_attn_weights
from loguru import logger


class HierarchicalCompressionStrategy:
    """
    分层量化策略 (Hierarchical Compression Strategy, HCS)
    
    核心哲学：
    1. 8-bit (W_8): 负责绝对安全，采用"底线思维"，保护最脆弱的关键信息
    2. 4-bit (W_4): 负责经济实用，采用"整体思维"，在性能和压缩率之间做权衡
    """
    
    def __init__(self, z_threshold: float = 1, mhsr_threshold: float = 0.995, 
                 ter_threshold: float = 0.95, search_step: int = 8):
        """
        Args:
            z_threshold: Z-score阈值，用于识别枢纽
            mhsr_threshold: 最弱枢纽保留率阈值（安全底线）
            ter_threshold: 整体能量保留率阈值（经济目标）
            search_step: 搜索步长
        """
        self.z_threshold = z_threshold
        self.mhsr_threshold = mhsr_threshold
        self.ter_threshold = ter_threshold
        self.search_step = search_step
        
    def identify_outliers(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        预处理：识别枢纽 (Outlier Diagnosis)
        使用因果感知的归一化 (Causal-Aware Normalization)
        
        Args:
            attn_weights: [batch, heads, seq, seq] 或 [heads, seq, seq]
            
        Returns:
            outlier_indices: [seq] bool tensor，True表示该位置是枢纽
        """
        if attn_weights.dim() == 4:
            # 对batch维度求平均
            attn_weights = attn_weights.mean(dim=0)  # [heads, seq, seq]
        
        if attn_weights.dim() == 3:
            # 对heads维度求平均
            attn_weights = attn_weights.mean(dim=0)  # [seq, seq]
        
        seq_len = attn_weights.size(0)
        
        # 第一步：计算每个Key的"单位曝光能量 (Energy Per Exposure, EPE)"
        epe_values = []
        
        for j in range(seq_len):
            # 1. 计算列能量总和 (Sum)
            sum_j = attn_weights[:, j].sum().item()
            
            # 2. 计算有效曝光窗口大小 (Effective Window)
            # 在因果模型中，Key k_j 只能被 Query q_j, q_{j+1}, ..., q_{L-1} 关注
            n_effective = seq_len - j
            
            # 3. 计算单位曝光能量 (EPE)
            # 为数值稳定性，加上小的epsilon
            epsilon = 1e-8
            epe_j = sum_j / (n_effective + epsilon)
            epe_values.append(epe_j)
        
        # 转换为tensor
        epe_tensor = torch.tensor(epe_values, device=attn_weights.device)
        
        # 第二步：在公平的数据上进行Z-score分析
        mu_epe = epe_tensor.mean()
        sigma_epe = epe_tensor.std()
        
        # 计算因果感知Z-score
        z_causal = (epe_tensor - mu_epe) / (sigma_epe + 1e-8)
        
        # 识别枢纽
        outlier_mask = z_causal > self.z_threshold
        
        logger.info(f"因果感知枢纽识别结果:")
        logger.info(f"  - 序列长度: {seq_len}")
        logger.info(f"  - EPE均值μ: {mu_epe:.6f}")
        logger.info(f"  - EPE标准差σ: {sigma_epe:.6f}")
        logger.info(f"  - Z-score阈值: {self.z_threshold}")
        logger.info(f"  - 识别出枢纽数量: {outlier_mask.sum().item()}")
        logger.info(f"  - 枢纽位置: {outlier_mask.nonzero().flatten().tolist()}")
        
        # 打印前几个和后几个位置的详细信息
        logger.info(f"  - 详细EPE信息:")
        for i in [0, 1, 2, seq_len-3, seq_len-2, seq_len-1]:
            if 0 <= i < seq_len:
                effective_window = seq_len - i
                logger.info(f"    位置{i:3d}: 有效窗口={effective_window:3d}, EPE={epe_values[i]:.6f}, Z-score={z_causal[i].item():.3f}")
        
        return outlier_mask
    
    def calculate_mhsr(self, attn_weights: torch.Tensor, outlier_indices: torch.Tensor, 
                      w8: int) -> float:
        """
        计算最弱枢纽保留率 (Minimum Hub Slice Ratio, MHSR)
        
        Args:
            attn_weights: [seq, seq] 注意力权重矩阵
            outlier_indices: [seq] 枢纽位置掩码
            w8: 8-bit窗口大小
            
        Returns:
            mhsr: 最弱枢纽保留率
        """
        seq_len = attn_weights.size(0)
        outlier_positions = outlier_indices.nonzero().flatten()
        
        if len(outlier_positions) == 0:
            return 1.0  # 没有枢纽，返回完美保留率
        
        slice_ratios = []
        
        for hub_j in outlier_positions:
            hub_j = hub_j.item()
            
            # 计算枢纽j的总能量
            total_energy = attn_weights[:, hub_j].sum()
            
            # 计算落在窗口内的能量
            # distance < w8 意味着 (i - j) < w8，即 i < j + w8
            windowed_energy = 0.0
            for i in range(seq_len):
                if (i - hub_j) < w8:  # 在窗口内
                    windowed_energy += attn_weights[i, hub_j]
            
            # 计算保留率
            if total_energy > 0:
                slice_ratio = windowed_energy / total_energy
            else:
                slice_ratio = 1.0
                
            slice_ratios.append(slice_ratio.item())
        
        # 返回最弱（最小）的保留率
        mhsr = min(slice_ratios)
        
        return mhsr
    
    def calculate_ter(self, attn_weights: torch.Tensor, w4: int) -> float:
        """
        计算整体能量保留率 (Total Energy Retention, TER)
        
        Args:
            attn_weights: [seq, seq] 注意力权重矩阵
            w4: 4-bit窗口大小（实际上是8-bit+4-bit的总覆盖范围）
            
        Returns:
            ter: 整体能量保留率
        """
        seq_len = attn_weights.size(0)
        
        # 计算总能量
        total_energy = attn_weights.sum()
        
        # 计算窗口内的能量
        windowed_energy = 0.0
        for i in range(seq_len):
            for j in range(seq_len):
                if (i - j) < w4:  # 在窗口内
                    windowed_energy += attn_weights[i, j]
        
        # 计算保留率
        ter = windowed_energy / total_energy if total_energy > 0 else 1.0
        
        return ter.item()
    
    def stage1_find_safety_boundary(self, attn_weights: torch.Tensor, 
                                   outlier_indices: torch.Tensor,
                                   max_w8: int = 512) -> int:
        """
        第一阶段：确定8-bit"安全边界" W_8*
        
        目标：找到能保护好最弱那个枢纽的最小窗口
        
        Args:
            attn_weights: [seq, seq] 注意力权重矩阵
            outlier_indices: [seq] 枢纽位置掩码
            max_w8: 最大搜索范围
            
        Returns:
            w8_optimal: 最优8-bit窗口大小
        """
        logger.info(f"\n【第一阶段：确定8-bit安全边界】")
        logger.info(f"目标MHSR阈值: {self.mhsr_threshold}")
        logger.info(f"搜索步长: {self.search_step}")
        
        # 从0开始搜索
        for w8 in range(0, max_w8 + 1, self.search_step):
            mhsr = self.calculate_mhsr(attn_weights, outlier_indices, w8)
            
            logger.info(f"  W_8={w8:3d}: MHSR={mhsr:.4f}", end="")
            
            if mhsr >= self.mhsr_threshold:
                logger.info(f" ✓ (满足安全要求)")
                logger.info(f"找到最优安全边界: W_8* = {w8}")
                return w8
            else:
                logger.info(f" ✗ (未达到安全要求)")
        
        logger.info(f"警告: 在搜索范围内未找到满足安全要求的W_8，返回最大值: {max_w8}")
        return max_w8
    
    def stage2_find_economic_boundary(self, attn_weights: torch.Tensor, 
                                     w8_optimal: int,
                                     max_w4: int = 1024) -> int:
        """
        第二阶段：确定4-bit"经济边界" W_4*
        
        目标：在W_8*确定的安全区之外，找到能保留足够整体能量的最小延伸窗口
        
        Args:
            attn_weights: [seq, seq] 注意力权重矩阵
            w8_optimal: 已确定的最优8-bit窗口大小
            max_w4: 最大搜索范围
            
        Returns:
            w4_optimal: 最优4-bit窗口大小
        """
        logger.info(f"\n【第二阶段：确定4-bit经济边界】")
        logger.info(f"固定安全边界: W_8* = {w8_optimal}")
        logger.info(f"目标TER阈值: {self.ter_threshold}")
        
        # 从W_8*开始搜索（W_4必须大于等于W_8）
        for w4 in range(w8_optimal, max_w4 + 1, self.search_step):
            ter = self.calculate_ter(attn_weights, w4)
            
            logger.info(f"  W_4={w4:3d}: TER={ter:.4f}", end="")
            
            if ter >= self.ter_threshold:
                logger.info(f" ✓ (满足经济要求)")
                logger.info(f"找到最优经济边界: W_4* = {w4}")
                return w4
            else:
                logger.info(f" ✗ (未达到经济要求)")
        
        logger.info(f"警告: 在搜索范围内未找到满足经济要求的W_4，返回最大值: {max_w4}")
        return max_w4
    
    def apply_hcs(self, attn_weights: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
        """
        应用完整的HCS框架
        
        Args:
            attn_weights: [batch, heads, seq, seq] 或简化维度的注意力权重
            
        Returns:
            w8_optimal: 最优8-bit窗口大小
            w4_optimal: 最优4-bit窗口大小
            outlier_indices: 枢纽位置掩码
        """
        logger.info("=" * 80)
        logger.info("开始分层量化策略 (HCS)")
        logger.info("=" * 80)
        
        # 降维处理，得到单个注意力矩阵
        if attn_weights.dim() == 4:
            attn_matrix = attn_weights.mean(dim=(0, 1))  # [seq, seq]
        elif attn_weights.dim() == 3:
            attn_matrix = attn_weights.mean(dim=0)  # [seq, seq]
        else:
            attn_matrix = attn_weights  # [seq, seq]
        
        # 预处理：识别枢纽
        outlier_indices = self.identify_outliers(attn_weights)
        return 0, 0, 0
        
        # 第一阶段：确定安全边界
        w8_optimal = self.stage1_find_safety_boundary(attn_matrix, outlier_indices)
        
        # 第二阶段：确定经济边界
        w4_optimal = self.stage2_find_economic_boundary(attn_matrix, w8_optimal)
        
        # 输出最终结果
        logger.info(f"\n【HCS框架完成】")
        logger.info(f"最优窗口配置: (W_8*={w8_optimal}, W_4*={w4_optimal})")
        
        # 计算最终的量化区域统计
        seq_len = attn_matrix.size(0)
        total_elements = seq_len * seq_len
        
        # 统计三个区域的大小
        region_8bit = 0
        region_4bit = 0
        region_pruned = 0
        
        for i in range(seq_len):
            for j in range(seq_len):
                distance = i - j
                if distance < w8_optimal:
                    region_8bit += 1
                elif distance < w4_optimal:
                    region_4bit += 1
                else:
                    region_pruned += 1
        
        logger.info(f"量化区域分布:")
        logger.info(f"  - 8-bit区域: {region_8bit:5d} ({region_8bit/total_elements:.1%})")
        logger.info(f"  - 4-bit区域: {region_4bit:5d} ({region_4bit/total_elements:.1%})")
        logger.info(f"  - 剪枝区域: {region_pruned:5d} ({region_pruned/total_elements:.1%})")
        
        logger.info("=" * 80)
        
        return w8_optimal, w4_optimal, outlier_indices
    
    def visualize_quantization_regions(self, seq_len: int, w8: int, w4: int, 
                                     outlier_indices: torch.Tensor,
                                     save_path: str = None):
        """
        可视化量化区域划分
        
        Args:
            seq_len: 序列长度
            w8: 8-bit窗口大小
            w4: 4-bit窗口大小
            outlier_indices: 枢纽位置
            save_path: 保存路径
        """
        # 创建区域矩阵
        regions = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            for j in range(seq_len):
                distance = i - j
                if distance < w8:
                    regions[i, j] = 3  # 8-bit (最高精度)
                elif distance < w4:
                    regions[i, j] = 2  # 4-bit (中等精度)
                else:
                    regions[i, j] = 1  # 剪枝 (最低精度)
        
        # 标记枢纽列
        outlier_positions = outlier_indices.nonzero().flatten()
        for pos in outlier_positions:
            regions[:, pos] = torch.maximum(regions[:, pos], torch.tensor(3.5))  # 枢纽标记
        
        # 绘制
        plt.figure(figsize=(12, 10))
        im = plt.imshow(regions.numpy(), cmap='RdYlBu_r', aspect='auto')
        
        # 设置颜色条
        cbar = plt.colorbar(im)
        cbar.set_ticks([1, 2, 3, 3.5])
        cbar.set_ticklabels(['Pruned', '4-bit', '8-bit', 'Hub'])
        
        # 标记枢纽位置
        for pos in outlier_positions:
            plt.axvline(x=pos.item(), color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        plt.title(f'HCS量化区域划分 (W_8={w8}, W_4={w4})')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # 添加说明
        plt.text(0.02, 0.98, f'8-bit窗口: {w8}\\n4-bit窗口: {w4}\\n枢纽数量: {len(outlier_positions)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存到: {save_path}")
        
        plt.show()

# 集成到你的搜索函数中
def search_with_hcs(layers, layer_idx, head_id, inps, ori_outputs, 
                    layer_kwargs, args, 
                    mhsr_threshold: float = 0.995,
                    ter_threshold: float = 0.95) -> Tuple[int, int]:
    """
    使用HCS框架的搜索函数
    
    Returns:
        w8_optimal: 最优8-bit窗口大小
        w4_optimal: 最优4-bit窗口大小
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"为 Layer {layer_idx} Head {head_id} 应用HCS框架")
    logger.info(f"{'='*80}")
    
    # 获取attention weights
    args.need_attention = True
    
    # 运行一次获取原始attention
    num_heads = layers[layer_idx].self_attn.config.num_attention_heads
    bit8_window_sizes = [512] * num_heads  # 使用大窗口获取完整attention
    bit4_window_sizes = [0] * num_heads
    
    replace_sdpa_for_block_with_attn_weights(layers[layer_idx], layer_idx, args,
                          bit8_window_sizes=bit8_window_sizes,
                          bit4_window_sizes=bit4_window_sizes,
                          sink_window_size=32)
    
    # _ = layers_infer(layers, layer_idx, inps, layer_kwargs, args)
    # import pdb; pdb.set_trace()
    _ = layers[layer_idx](inps, **layer_kwargs)[0]
    
    if hasattr(args, 'current_attention'):
        # 提取特定head的attention
        head_attn = args.current_attention[:, head_id, :, :]  # [batch, seq, seq]
        
        # 应用HCS框架
        hcs = HierarchicalCompressionStrategy(
            mhsr_threshold=mhsr_threshold,
            ter_threshold=ter_threshold
        )
        
        w8_optimal, w4_optimal, outlier_indices = hcs.apply_hcs(head_attn)
        
        # 可视化结果
        seq_len = head_attn.size(-1)
        # hcs.visualize_quantization_regions(
        #     seq_len, w8_optimal, w4_optimal, outlier_indices,
        #     save_path=f'hcs_vis_layer_{layer_idx}_head_{head_id}.png'
        # )
        
        return w8_optimal, w4_optimal
    
    else:
        logger.info("警告: 未能获取attention权重，返回默认值")
        return 128, 256

# 批量处理所有头的函数
def apply_hcs_to_all_heads(model, layer, layer_idx, sample_input, layer_kwargs, args):
    """
    为模型的所有注意力头应用HCS框架
    
    Returns:
        head_configs: Dict[Tuple[int, int], Tuple[int, int]]
                     键为(layer_idx, head_id)，值为(w8_optimal, w4_optimal)
    """
    head_configs = {}
    

    num_heads = layer.self_attn.config.num_attention_heads
    
    for head_id in range(num_heads):
        logger.info(f"\\n处理 Layer {layer_idx} Head {head_id}...")
        
        w8_opt, w4_opt = search_with_hcs(
            model.model.layers, layer_idx, head_id, 
            sample_input, None, layer_kwargs, args
        )
        
        head_configs[(layer_idx, head_id)] = (w8_opt, w4_opt)
        
        logger.info(f"Layer {layer_idx} Head {head_id}: W_8*={w8_opt}, W_4*={w4_opt}")
    
    return head_configs