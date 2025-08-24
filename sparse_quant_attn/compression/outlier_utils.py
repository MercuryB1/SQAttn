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
    
    def __init__(self, p_threshold: float = 0.45, mhsr_threshold: float = 0.95, 
                 ter_threshold: float = 0.80, search_step: int = 1):
        """
        Args:
            p_threshold: 固定比例阈值，用于识别枢纽 (建议0.1-0.15)
            mhsr_threshold: 最弱枢纽保留率阈值（安全底线）
            ter_threshold: 整体能量保留率阈值（经济目标）
            search_step: 搜索步长
        """
        self.p_threshold = p_threshold
        self.mhsr_threshold = mhsr_threshold
        self.ter_threshold = ter_threshold
        self.search_step = search_step
        
    def identify_outliers(self, attn_weights: torch.Tensor, layer_idx, head_id, sink_size: int = 32) -> torch.Tensor:
        """
        预处理：识别枢纽 (Outlier Diagnosis)
        使用基于固定比例阈值的异常检测 (Fixed-Ratio Thresholding, FRT)
        
        Args:
            attn_weights: [batch, heads, seq, seq] 或 [heads, seq, seq]
            layer_idx: 层索引
            head_id: 头索引
            sink_size: sink token数量，前sink_size个位置默认为8-bit，不参与异常检测
            
        Returns:
            outlier_indices: [seq] bool tensor，True表示该位置是枢纽
        """
       
        
        if attn_weights.dim() == 4:
            # 对batch维度求平均
            attn_weights = attn_weights.mean(dim=0)  # [heads, seq, seq]
        
        if attn_weights.dim() == 3:
            # 对heads维度求平均
            attn_weights = attn_weights.mean(dim=0)  # [seq, seq]
        
        
        # 保存原始维度用于可视化
        original_attn = attn_weights.clone()
        
        seq_len = attn_weights.size(0)
        
        # 初始化异常值掩码，前sink_size个位置默认为False（不是异常值，因为它们是固定的sink token）
        outlier_mask = torch.zeros(seq_len, dtype=torch.bool, device=attn_weights.device)
        
        # 第一步：计算每个列的"被关注强度"分数 - CAAB (仅对sink_size之后的位置)
        # 1. 设定激活阈值 τ = 1/L
        tau = 2.0 / seq_len
        
        caab_scores = []
        vab_raw_scores = []
        
        for j in range(seq_len):
            if j < sink_size:
                # sink token，不参与异常检测，设置默认值
                vab_raw_scores.append(0)
                caab_scores.append(0.0)
            else:
                # 2. 计算原始激活广度 VAB_raw(j)
                # 数一下 A_avg[i, j] > τ 的行数
                column_j = attn_weights[:, j]
                activated_count = (column_j > tau).sum().item()
                vab_raw_j = activated_count
                
                # 3. 计算CAAB分数: CAAB(j) = VAB_raw(j) / (L - j)
                # 因果感知的激活广度，归一化到 [0, 1]
                effective_window = seq_len - j
                caab_j = vab_raw_j / effective_window if effective_window > 0 else 0.0
                
                vab_raw_scores.append(vab_raw_j)
                caab_scores.append(caab_j)
        
        # 转换为tensor
        caab_tensor = torch.tensor(caab_scores, device=attn_weights.device)
        
        # 第二步：应用固定阈值进行决策 (仅对sink_size之后的位置)
        # 如果 CAAB(j) > P_threshold 且 j >= sink_size，则列j是异常值
        for j in range(sink_size, seq_len):
            if caab_tensor[j] > self.p_threshold:
                outlier_mask[j] = True
        
        # 计算统计信息（排除sink token）
        non_sink_caab = caab_tensor[sink_size:]
        candidate_positions = list(range(sink_size, seq_len))
        identified_outliers = outlier_mask[sink_size:].nonzero().flatten() + sink_size
        
        logger.info(f"[Layer {layer_idx} Head {head_id}] FRT异常检测结果:")
        logger.info(f"  - 序列长度: {seq_len}")
        logger.info(f"  - Sink token数量: {sink_size} (位置0-{sink_size-1}固定为8-bit)")
        logger.info(f"  - 候选检测位置: {sink_size}-{seq_len-1}")
        logger.info(f"  - 激活阈值τ: {tau:.6f} (1/L)")
        logger.info(f"  - 固定比例阈值: {self.p_threshold}")
        logger.info(f"  - 识别出枢纽数量: {outlier_mask.sum().item()} (不含sink token)")
        logger.info(f"  - 枢纽位置: {identified_outliers.tolist()}")
        
        # 打印一些关键位置的详细信息
        logger.info(f"  - 详细CAAB信息 (排除sink token):")
        key_positions = []
        if sink_size < seq_len:
            key_positions.extend([sink_size, sink_size+1, sink_size+2])  # 前几个非sink位置
        if seq_len > 3:
            key_positions.extend([seq_len-3, seq_len-2, seq_len-1])  # 最后几个位置
        
        for i in key_positions:
            if sink_size <= i < seq_len:
                effective_window = seq_len - i
                activated_count = vab_raw_scores[i]
                caab_score = caab_scores[i]
                outlier_status = "🔥枢纽" if outlier_mask[i] else "普通"
                logger.info(f"    位置{i:3d}: 有效窗口={effective_window:3d}, 激活数={activated_count:3d}, CAAB={caab_score:.4f} [{outlier_status}]")
        
        # 可视化attention weights和异常值
        self.visualize_attention_with_outliers_frt(
            original_attn, outlier_mask, vab_raw_scores, caab_scores, tau,
            layer_idx, head_id, sink_size
        )
        
        return outlier_mask
    
    def visualize_attention_with_outliers_frt(self, attn_weights: torch.Tensor, 
                                            outlier_mask: torch.Tensor,
                                            vab_raw_scores: list,
                                            caab_scores: list,
                                            tau: float,
                                            layer_idx: int, head_id: int, sink_size: int = 32):
        """
        可视化attention weights和FRT识别出的异常值
        
        Args:
            attn_weights: 原始attention权重
            outlier_mask: 异常值掩码
            vab_raw_scores: VAB原始分数列表
            caab_scores: CAAB分数列表
            tau: 激活阈值
            layer_idx: 层索引
            head_id: 头索引
        """
        # 处理维度，获取单个attention矩阵用于可视化
        if attn_weights.dim() == 4:
            # [batch, heads, seq, seq] -> [seq, seq]
            attn_matrix = attn_weights[0, head_id].float().cpu().numpy()
        elif attn_weights.dim() == 3:
            # [heads, seq, seq] -> [seq, seq]  
            attn_matrix = attn_weights[head_id].float().cpu().numpy()
        else:
            # [seq, seq]
            attn_matrix = attn_weights.float().cpu().numpy()
        
        seq_len = attn_matrix.shape[0]
        outlier_positions = outlier_mask.nonzero().flatten().cpu().numpy()
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 原始attention weights
        im1 = ax1.imshow(attn_matrix, cmap='Blues', aspect='auto')
        ax1.set_title(f'Layer {layer_idx} Head {head_id}: Attention Weights')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        
        # 标记异常值列（红色虚线）和sink区域（绿色虚线）
        for pos in outlier_positions:
            ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax1.text(pos, seq_len * 0.95, f'{pos}', color='red', fontweight='bold', 
                    ha='center', va='bottom', fontsize=8)
        
        # 标记sink区域边界
        if sink_size > 0:
            ax1.axvline(x=sink_size-0.5, color='green', linestyle='-', alpha=0.8, linewidth=3)
            ax1.text(sink_size/2, seq_len * 0.02, f'Sink (0-{sink_size-1})', color='green', fontweight='bold', 
                    ha='center', va='bottom', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # 子图2: 列能量分布 + 激活阈值 + sink区域标记
        column_sums = np.sum(attn_matrix, axis=0)
        ax2.plot(range(seq_len), column_sums, 'b-', linewidth=1, label='Column Sum')
        ax2.axhline(y=tau, color='orange', linestyle=':', alpha=0.7, label=f'Activation Threshold τ={tau:.4f}')
        
        # 标记sink区域
        if sink_size > 0:
            ax2.axvspan(0, sink_size-1, alpha=0.2, color='green', label=f'Sink Tokens (0-{sink_size-1})')
        
        # 只标记非sink的异常值
        non_sink_outliers = [pos for pos in outlier_positions if pos >= sink_size]
        if non_sink_outliers:
            ax2.scatter(non_sink_outliers, [column_sums[i] for i in non_sink_outliers], 
                       color='red', s=50, zorder=5, label=f'Outliers ({len(non_sink_outliers)})')
        
        ax2.set_title('Column Energy Distribution')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Total Attention (Sum)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: VAB原始激活广度 + sink区域标记
        ax3.plot(range(seq_len), vab_raw_scores, 'g-', linewidth=1, label='VAB Raw Count')
        
        # 标记sink区域
        if sink_size > 0:
            ax3.axvspan(0, sink_size-1, alpha=0.2, color='green', label=f'Sink Tokens (0-{sink_size-1})')
        
        # 只标记非sink的异常值
        if non_sink_outliers:
            ax3.scatter(non_sink_outliers, [vab_raw_scores[i] for i in non_sink_outliers],
                       color='red', s=50, zorder=5, label=f'Outliers ({len(non_sink_outliers)})')
        
        ax3.set_title('Raw Activation Breadth (VAB)')
        ax3.set_xlabel('Key Position')
        ax3.set_ylabel('Activated Query Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: CAAB分数分布 + sink区域标记
        ax4.plot(range(seq_len), caab_scores, 'purple', linewidth=1, label='CAAB Score')
        
        # 标记sink区域
        if sink_size > 0:
            ax4.axvspan(0, sink_size-1, alpha=0.2, color='green', label=f'Sink Tokens (0-{sink_size-1})')
        
        # 只标记非sink的异常值
        if non_sink_outliers:
            ax4.scatter(non_sink_outliers, [caab_scores[i] for i in non_sink_outliers],
                       color='red', s=50, zorder=5, label=f'Outliers ({len(non_sink_outliers)})')
        
        ax4.axhline(y=self.p_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Threshold ({self.p_threshold})')
        ax4.set_title('Causal-Aware Activation Breadth (CAAB)')
        ax4.set_xlabel('Key Position')
        ax4.set_ylabel('CAAB Score')
        ax4.set_ylim(0, 1.0)  # CAAB分数在[0,1]范围内
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 创建保存目录并保存图片
        import os
        os.makedirs(f'vis/attn_vis/layer_{layer_idx}', exist_ok=True)
        save_path = f'vis/attn_vis/layer_{layer_idx}/head_{head_id}_frt_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"FRT异常值分析可视化已保存到: {save_path}")
        
        # 显示图片
        plt.show()
        
        # 输出详细的异常值分析（排除sink token）
        # non_sink_outliers = [pos for pos in outlier_positions if pos >= sink_size]
        # if len(non_sink_outliers) > 0:
        #     logger.info(f"  - 异常值详细分析 (排除sink token):")
        #     for i, pos in enumerate(non_sink_outliers):
        #         effective_window = seq_len - pos
        #         col_sum = column_sums[pos]
        #         vab_raw = vab_raw_scores[pos]
        #         caab_score = caab_scores[pos]
                
        #         # logger.info(f"    异常值 {i+1}: 位置={pos:3d}, 有效窗口={effective_window:3d}")
        #         # logger.info(f"              列总和={col_sum:.6f}, 激活数={vab_raw:3d}, CAAB={caab_score:.4f}")
                
        #         # 分析该列的attention模式
        #         col_attention = attn_matrix[:, pos]
        #         max_attn_pos = np.argmax(col_attention)
        #         max_attn_val = col_attention[max_attn_pos]
                
        #         # 计算超过阈值的激活比例
        #         activated_queries = np.sum(col_attention > tau)
        #         activation_ratio = activated_queries / effective_window
                
                # logger.info(f"              最强关注: Query{max_attn_pos} -> {max_attn_val:.6f}")
                # logger.info(f"              激活比例: {activation_ratio:.2%} ({activated_queries}/{effective_window})")
                
                # 检查是否主要在后半段被关注
                # second_half_start = seq_len // 2
                # first_half_sum = np.sum(col_attention[:second_half_start])
                # second_half_sum = np.sum(col_attention[second_half_start:])
                
                # if second_half_sum > first_half_sum:
                #     logger.info(f"              模式: 主要在序列后半段被关注 (后半段:{second_half_sum:.4f} > 前半段:{first_half_sum:.4f})")
                # else:
                #     logger.info(f"              模式: 主要在序列前半段被关注 (前半段:{first_half_sum:.4f} > 后半段:{second_half_sum:.4f})")
        # else:
        #     logger.info(f"  - 未发现异常值 (排除sink token)")
        
        plt.close()  # 关闭图形以释放内存
    
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
            logger.warning(f"MHSR计算: 没有枢纽，返回1.0")
            return 1.0  # 没有枢纽，返回完美保留率
        
        slice_ratios = []
        
        logger.info(f"MHSR计算: W_8={w8}, 枢纽位置={outlier_positions.tolist()}, 序列长度={seq_len}")
        
        for hub_j in outlier_positions:
            hub_j = hub_j.item()
            
            # 计算枢纽j的总能量（该列的总和）
            total_energy = attn_weights[:, hub_j].sum().item()  # 转为标量
            
            # 计算落在窗口内的能量
            windowed_energy = 0.0
            valid_queries = 0  # 统计有效的query数量
            
            for i in range(seq_len):
                # Query i 能关注 Key hub_j（因果约束）
                if i >= hub_j:
                    # 检查是否在窗口内
                    if (i - hub_j) < w8:
                        windowed_energy += attn_weights[i, hub_j].item()  # 转为标量
                        valid_queries += 1
            
            # 计算保留率（窗口内能量 / 总能量）
            if total_energy > 0:
                slice_ratio = windowed_energy / total_energy
            else:
                slice_ratio = 1.0
                
            slice_ratios.append(slice_ratio)  # 现在slice_ratio已经是标量了
            
            # 详细调试信息
            total_valid_queries = seq_len - hub_j  # 因果约束下的总有效query数
            # logger.info(f"  枢纽{hub_j}: 总能量={total_energy:.6f}, 窗口内能量={windowed_energy:.6f}")
            # logger.info(f"    总有效query={total_valid_queries}, 窗口内query={valid_queries}, 保留率={slice_ratio:.6f}")
            
            # 如果窗口覆盖了所有有效query，说明窗口太大了
            # if valid_queries >= total_valid_queries:
            #     logger.warning(f"    警告: 枢纽{hub_j}的窗口{w8}已覆盖所有有效query!")
        
        # 返回最弱（最小）的保留率
        mhsr = min(slice_ratios)
        # logger.info(f"MHSR计算结果: 所有保留率={[f'{r:.6f}' for r in slice_ratios]}, 最小值={mhsr:.6f}")
        
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
        total_energy = attn_weights.sum().item()
        
        if total_energy <= 1e-8:
            logger.warning(f"TER计算: 总能量过小({total_energy:.8f})，返回1.0")
            return 1.0
        
        # 向量化计算窗口内的能量
        # 创建位置索引矩阵
        i_indices = torch.arange(seq_len, device=attn_weights.device).unsqueeze(1)  # [seq, 1]
        j_indices = torch.arange(seq_len, device=attn_weights.device).unsqueeze(0)  # [1, seq]
        
        # 因果约束: i >= j
        causal_mask = i_indices >= j_indices
        
        # 窗口约束: (i - j) < w4
        window_mask = (i_indices - j_indices) < w4
        
        # 组合掩码: 因果 + 窗口
        valid_mask = causal_mask & window_mask
        
        # 计算窗口内的总能量
        windowed_energy = (attn_weights * valid_mask.float()).sum().item()
        
        # 计算保留率
        ter = windowed_energy / total_energy
        
        # 确保TER不超过1.0（数值稳定性）
        ter = min(ter, 1.0)
        
        # 统计信息
        total_elements = seq_len * seq_len
        valid_elements = valid_mask.sum().item()
        coverage_ratio = valid_elements / total_elements
        
        logger.info(f"TER计算: W_4={w4}, 总能量={total_energy:.6f}, 窗口内能量={windowed_energy:.6f}")
        logger.info(f"  TER={ter:.6f}, 覆盖元素={valid_elements}/{total_elements} ({coverage_ratio:.2%})")
        
        # 检查异常情况
        if ter > 1.0:
            logger.warning(f"  异常: TER>1.0! 窗口内能量({windowed_energy:.8f}) > 总能量({total_energy:.8f})")
        
        return ter
    
    def stage1_find_safety_boundary(self, attn_weights: torch.Tensor, 
                                   outlier_indices: torch.Tensor,
                                   max_w8: int = 512, step: int = 1) -> int:
        """
        第一阶段：确定8-bit"安全边界" W_8* (使用步长为32的二分搜索)
        
        目标：找到能保护好最弱那个枢纽的最小窗口
        
        Args:
            attn_weights: [seq, seq] 注意力权重矩阵
            outlier_indices: [seq] 枢纽位置掩码
            max_w8: 最大搜索范围
            step: 搜索步长，默认32
            
        Returns:
            w8_optimal: 最优8-bit窗口大小
        """
        logger.info(f"\n【第一阶段：确定8-bit安全边界 - 步长二分搜索】")
        logger.info(f"目标MHSR阈值: {self.mhsr_threshold}")
        logger.info(f"搜索范围: [0, {max_w8}], 步长: {step}")
        
        # 生成候选值列表 (按步长)
        candidates = list(range(0, max_w8 + 1, step))
        if candidates[-1] < max_w8:
            candidates.append(max_w8)  # 确保包含最大值
        
        logger.info(f"候选值数量: {len(candidates)}, 范围: {candidates[0]} -> {candidates[-1]}")
        
        # 二分搜索
        left, right = 0, len(candidates) - 1
        result = candidates[-1]  # 默认返回最大值
        
        while left <= right:
            mid_idx = (left + right) // 2
            w8_candidate = candidates[mid_idx]
            mhsr = self.calculate_mhsr(attn_weights, outlier_indices, w8_candidate)
            
            logger.info(f"  二分搜索: W_8={w8_candidate:3d} (索引{mid_idx}), MHSR={mhsr:.4f}", end="")
            
            if mhsr >= self.mhsr_threshold:
                result = w8_candidate
                logger.info(f" ✓ (满足安全要求)")
                # 满足条件，尝试找更小的值
                right = mid_idx - 1
            else:
                logger.info(f" ✗ (未达到安全要求)")
                # 不满足条件，需要更大的值
                left = mid_idx + 1
        
        logger.info(f"找到最优安全边界: W_8* = {result}")
        return result
    
    def stage2_find_economic_boundary(self, attn_weights: torch.Tensor, 
                                     w8_optimal: int,
                                     max_w4: int = 1024, step: int = 1) -> int:
        """
        第二阶段：确定4-bit"经济边界" W_4* (使用步长为32的二分搜索)
        
        目标：在W_8*确定的安全区之外，找到能保留足够整体能量的最小延伸窗口
        
        Args:
            attn_weights: [seq, seq] 注意力权重矩阵
            w8_optimal: 已确定的最优8-bit窗口大小
            max_w4: 最大搜索范围
            step: 搜索步长，默认32
            
        Returns:
            w4_optimal: 最优4-bit窗口大小
        """
        logger.info(f"\n【第二阶段：确定4-bit经济边界 - 步长二分搜索】")
        logger.info(f"固定安全边界: W_8* = {w8_optimal}")
        logger.info(f"目标TER阈值: {self.ter_threshold}")
        logger.info(f"搜索范围: [{w8_optimal}, {max_w4}], 步长: {step}")
        
        # 生成候选值列表 (从w8_optimal开始，按步长)
        # 确保w8_optimal在候选列表中
        start_val = ((w8_optimal + step - 1) // step) * step  # 向上取整到最近的step倍数
        if start_val < w8_optimal:
            start_val = w8_optimal
            
        candidates = [w8_optimal]  # 确保w8_optimal在列表中
        candidates.extend(range(start_val, max_w4 + 1, step))
        if candidates[-1] < max_w4:
            candidates.append(max_w4)  # 确保包含最大值
            
        # 去重并排序
        candidates = sorted(list(set(candidates)))
        
        logger.info(f"候选值数量: {len(candidates)}, 范围: {candidates[0]} -> {candidates[-1]}")
        
        # 二分搜索
        left, right = 0, len(candidates) - 1
        result = candidates[-1]  # 默认返回最大值
        
        while left <= right:
            mid_idx = (left + right) // 2
            w4_candidate = candidates[mid_idx]
            ter = self.calculate_ter(attn_weights, w4_candidate)
            
            logger.info(f"  二分搜索: W_4={w4_candidate:3d} (索引{mid_idx}), TER={ter:.4f}", end="")
            
            if ter >= self.ter_threshold:
                result = w4_candidate
                logger.info(f" ✓ (满足经济要求)")
                # 满足条件，尝试找更小的值
                right = mid_idx - 1
            else:
                logger.info(f" ✗ (未达到经济要求)")
                # 不满足条件，需要更大的值
                left = mid_idx + 1
        
        logger.info(f"找到最优经济边界: W_4* = {result}")
        return result
    
    def apply_hcs(self, attn_weights: torch.Tensor, layer_idx: int = -1, head_id: int = -1) -> Tuple[int, int, torch.Tensor]:
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

        outlier_indices = self.identify_outliers(attn_weights, layer_idx, head_id, 4)
        
        # logger.info("=== 手动验证MHSR计算 ===")
        # for test_w8 in [32, 64, 128, 256]:
        #     mhsr = self.calculate_mhsr(attn_weights, outlier_indices, test_w8)
        #     logger.info(f"手动测试: W_8={test_w8}, MHSR={mhsr:.8f}")

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
                if i < j:
                    # 上三角：causal mask区域，设为0（将显示为白色）
                    regions[i, j] = 0
                else:
                    # 下三角：根据窗口大小分配
                    distance = i - j
                    if distance < w8:
                        regions[i, j] = 3  # 8-bit (最高精度)
                    elif distance < w4:
                        regions[i, j] = 2  # 4-bit (中等精度)
                    else:
                        regions[i, j] = 1  # 剪枝 (最低精度)
        
        # 标记枢纽列 - 在下三角区域内
        outlier_positions = outlier_indices.nonzero().flatten()
        for pos in outlier_positions:
            # 只在下三角区域标记枢纽
            for i in range(pos.item(), seq_len):
                regions[i, pos] = torch.maximum(regions[i, pos], torch.tensor(3.5))  # 枢纽标记
        
        # 绘制
        plt.figure(figsize=(12, 10))
        
        # 使用自定义颜色映射
        from matplotlib.colors import ListedColormap, BoundaryNorm
        
        # 定义颜色：白色(causal), 红色(剪枝), 橙色(4-bit), 蓝色(8-bit), 深红色(枢纽)
        colors = ['white', 'lightcoral', 'orange', 'lightblue', 'darkred']
        cmap = ListedColormap(colors)
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.0]
        norm = BoundaryNorm(bounds, cmap.N)
        
        im = plt.imshow(regions.numpy(), cmap=cmap, norm=norm, aspect='auto')
        
        # 设置颜色条
        cbar = plt.colorbar(im, ticks=[0.25, 1, 2, 3, 3.75])
        cbar.set_ticklabels(['Causal Mask', 'Pruned', '4-bit', '8-bit', 'Hub'])
        
        # 标记枢纽位置的垂直线
        for pos in outlier_positions:
            plt.axvline(x=pos.item(), color='red', linestyle='--', alpha=0.7, linewidth=1)
        
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
                    p_threshold: float = 0.15,
                    mhsr_threshold: float = 0.995,
                    ter_threshold: float = 0.95) -> Tuple[int, int]:
    """
    使用HCS框架的搜索函数
    
    Args:
        p_threshold: FRT固定比例阈值 (建议0.1-0.15)
        mhsr_threshold: 最弱枢纽保留率阈值
        ter_threshold: 整体能量保留率阈值
    
    Returns:
        w8_optimal: 最优8-bit窗口大小
        w4_optimal: 最优4-bit窗口大小
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"为 Layer {layer_idx} Head {head_id} 应用HCS框架 (FRT方法)")
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
    _ = layers[layer_idx](inps, **layer_kwargs)[0]
    
    if hasattr(args, 'current_attention'):
        # 提取特定head的attention
        head_attn = args.current_attention[:, head_id, :, :]  # [batch, seq, seq]
        
        # 应用HCS框架 (使用FRT方法)
        hcs = HierarchicalCompressionStrategy(
            p_threshold=p_threshold,
            mhsr_threshold=mhsr_threshold,
            ter_threshold=ter_threshold
        )
        
        w8_optimal, w4_optimal, outlier_indices = hcs.apply_hcs(head_attn, layer_idx, head_id)
        
        # 可视化结果
        seq_len = head_attn.size(-1)
        import os
        os.makedirs(f'vis/quant_vis/layer_{layer_idx}', exist_ok=True)
        hcs.visualize_quantization_regions(
            seq_len, w8_optimal, w4_optimal, outlier_indices,
            save_path=f'vis/quant_vis/layer_{layer_idx}/head_{head_id}_hcs_regions.png'
        )
        
        return w8_optimal, w4_optimal
    
    else:
        logger.info("警告: 未能获取attention权重，返回默认值")
        return 128, 256

# 批量处理所有头的函数
def apply_hcs_to_all_heads(model, layer, layer_idx, inps, layer_kwargs, args):
    """
    为当前层的所有注意力头应用HCS框架
    
    Returns:
        bit8_window_sizes: List[int] - 每个头的8-bit窗口大小
        bit4_window_sizes: List[int] - 每个头的4-bit窗口大小
    """
    num_heads = layer.self_attn.config.num_attention_heads
    
    bit8_window_sizes = []
    bit4_window_sizes = []
    
    for head_id in range(num_heads):
        logger.info(f"处理 Layer {layer_idx} Head {head_id}...")
        
        w8_opt, w4_opt = search_with_hcs(
            model.model.layers, layer_idx, head_id, 
            inps, None, layer_kwargs, args
        )
        
        bit8_window_sizes.append(w8_opt)
        bit4_window_sizes.append(w4_opt)
        
        logger.info(f"Layer {layer_idx} Head {head_id}: W_8*={w8_opt}, W_4*={w4_opt}")
    
    return bit8_window_sizes, bit4_window_sizes