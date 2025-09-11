import torch, math
import triton
import triton.language as tl

@triton.jit
def attn_fwd_inner(acc, l_i, m_i, q, kv_len,
                   K_ptrs, V_ptrs, stride_kn, stride_vn,
                   start_m,
                   BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                   STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                   is_P_fp8_quant: tl.constexpr,
                   current_int8_window_ratio,  # 改为相对窗口比例
                   current_int4_window_ratio,  # 改为相对窗口比例
                   SINK_SIZE: tl.constexpr     # 保持固定大小
                   ):
    """
    Window Attention 前向计算的内部循环函数，支持 Sink + 分层窗口（相对窗口版本）。
    窗口大小现在是相对于序列长度的比例 (0-1)，但会对齐到128的倍数
    Sink size保持固定大小
    """
    
    # 根据 STAGE 决定处理范围
    if STAGE == 1:
        # STAGE 1: 处理 Sink 窗口 (前 SINK_SIZE 个 tokens，固定大小)
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        
        lo, hi = 0, tl.minimum(SINK_SIZE, query_start)
        
    elif STAGE == 2:
        # STAGE 2: 4-bit 窗口扩展区域 (排除 sink 和 INT8 窗口)
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        
        # 计算绝对窗口大小，并对齐到128的倍数
        int4_window_size_raw = tl.cast(current_int4_window_ratio * kv_len, tl.int32)
        int8_window_size_raw = tl.cast(current_int8_window_ratio * kv_len, tl.int32)
        
        # 对齐到128的倍数
        int4_window_size = ((int4_window_size_raw + 127) // 128) * 128
        int8_window_size = ((int8_window_size_raw + 127) // 128) * 128
        
        # 4-bit 窗口左边界
        int4_window_left = tl.maximum(SINK_SIZE, query_end - int4_window_size)
        # INT8 窗口左边界
        int8_window_left = tl.maximum(SINK_SIZE, query_end - int8_window_size)
        
        condition = int4_window_left < int8_window_left
        lo = tl.where(condition, int4_window_left, 0)
        hi = tl.where(condition, int8_window_left, 0)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
        
    elif STAGE == 3:
        # STAGE 3: INT8 窗口内对角线之前的块
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        
        # 计算绝对窗口大小，并对齐到128的倍数
        int8_window_size_raw = tl.cast(current_int8_window_ratio * kv_len, tl.int32)
        int8_window_size = ((int8_window_size_raw + 127) // 128) * 128
        
        # INT8 窗口范围 (排除 sink)
        int8_window_left = tl.maximum(SINK_SIZE, query_end - int8_window_size)
        int8_window_right = query_start
        
        condition = int8_window_left < int8_window_right
        lo = tl.where(condition, int8_window_left, 0)
        hi = tl.where(condition, int8_window_right, 0)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
        
    elif STAGE == 4:
        # STAGE 4: 对角线上的块
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        lo = query_start
        hi = query_end
        # 确保 lo 是 BLOCK_M 的倍数，并相应地移动 K/V 指针
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    
    # --- 主循环：迭代遍历窗口内的 Key/Value 序列 ---
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # --- 1. 加载 Key 块并计算 QK^T ---
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        qk = tl.dot(q, k).to(tl.float32)
        
        # --- 2. 应用窗口掩码 ---
        if STAGE == 4:
            # 对角线块上的精确窗口掩码
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(causal_mask, 0, -1.0e6)
        
        # 在线 Softmax 算法
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        
        # --- 3. 计算 P = softmax(QK^T) 和更新累加器 ---
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # --- 4. 计算 P * V 并累加到输出 ---
        v_mask = offs_n[:, None] < (kv_len - start_n)
        v = tl.load(V_ptrs, mask=v_mask)
        
        if is_P_fp8_quant:
            p = (p*448).to(tl.float8e4nv).to(tl.float16)/448
        
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)
        
        # --- 5. 更新循环变量 ---
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    
    return acc, l_i, m_i


@triton.jit
def attn_prefill_fwd(Q_int8, K_int8, Q_int4, K_int4, V, Out, Lse, Int8WindowRatios, Int4WindowRatios,
                     stride_qz, stride_qh, stride_qn,
                     stride_kz, stride_kh, stride_kn,
                     stride_vz, stride_vh, stride_vn,
                     stride_oz, stride_oh, stride_on,
                     qo_len, kv_len, H:tl.constexpr, num_kv_groups:tl.constexpr,
                     HEAD_DIM:tl.constexpr,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr,
                     STAGE: tl.constexpr,
                     RETURN_LSE: tl.constexpr,
                     is_P_fp8_quant: tl.constexpr,
                     SINK_SIZE: tl.constexpr  # 保持固定大小
                     ):
    """
    分层 Window Attention Prefill Kernel，支持 Sink + 多精度窗口（相对窗口版本）
    窗口大小是相对比例，但会对齐到128的倍数；Sink size保持固定
    """
    # --- 1. 获取当前 program instance 的 ID ---
    start_m = tl.program_id(0)
    off_z = tl.program_id(2)  # Batch 维度
    off_h = tl.program_id(1)  # Head 维度
    
    # --- 2. 计算内存指针 ---
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # INT8 精度指针
    Q_int8_ptrs = Q_int8 + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    K_int8_ptrs = K_int8 + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    
    # INT4 精度指针
    Q_int4_ptrs = Q_int4 + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    K_int4_ptrs = K_int4 + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    
    # V 和输出指针
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    # --- 3. 初始化在线 Softmax 的统计量 ---
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # --- 4. 加载当前 Query 块 ---
    q_mask = offs_m[:, None] < qo_len
    q_int8 = tl.load(Q_int8_ptrs, mask=q_mask)
    q_int4 = tl.load(Q_int4_ptrs, mask=q_mask)
    
    # 加载当前 head 的窗口比例
    current_int8_window_ratio = tl.load(Int8WindowRatios + off_h)
    current_int4_window_ratio = tl.load(Int4WindowRatios + off_h)
    
    # --- 5. 按 STAGE 顺序调用内部循环 ---
    # STAGE 1: Sink 窗口 (使用 INT8 精度)
    acc, l_i, m_i = attn_fwd_inner(acc, l_i, m_i, q_int8, kv_len, K_int8_ptrs, V_ptrs, stride_kn, stride_vn,
                                   start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                   1, offs_m, offs_n, is_P_fp8_quant,
                                   current_int8_window_ratio, current_int4_window_ratio, SINK_SIZE)
    
    # STAGE 2: 4-bit 窗口扩展区域 (使用 4-bit 精度)
    acc, l_i, m_i = attn_fwd_inner(acc, l_i, m_i, q_int4, kv_len, K_int4_ptrs, V_ptrs, stride_kn, stride_vn,
                                   start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                   2, offs_m, offs_n, is_P_fp8_quant,
                                   current_int8_window_ratio, current_int4_window_ratio, SINK_SIZE)
    
    # STAGE 3: INT8 窗口内对角线之前的块 (使用 INT8 精度)
    acc, l_i, m_i = attn_fwd_inner(acc, l_i, m_i, q_int8, kv_len, K_int8_ptrs, V_ptrs, stride_kn, stride_vn,
                                   start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                   3, offs_m, offs_n, is_P_fp8_quant,
                                   current_int8_window_ratio, current_int4_window_ratio, SINK_SIZE)
    
    # STAGE 4: 对角线上的块 (使用 INT8 精度)
    acc, l_i, m_i = attn_fwd_inner(acc, l_i, m_i, q_int8, kv_len, K_int8_ptrs, V_ptrs, stride_kn, stride_vn,
                                   start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                   4, offs_m, offs_n, is_P_fp8_quant,
                                   current_int8_window_ratio, current_int4_window_ratio, SINK_SIZE)
    
    # --- 6. 后处理和存储结果 ---
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))
    
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def attn_hierarchical_relative_window(q_int8, k_int8, q_int4, k_int4, v,
                            int8_window_ratios, int4_window_ratios, sink_size,  # sink_size保持固定
                            tensor_layout="HND", output_dtype=torch.float16,
                            return_lse=False, is_P_fp8_quant=False):
    """
    分层 Window Attention 接口，支持 Sink + 多精度窗口（相对窗口版本）
    
    Args:
        q_int8, k_int8: INT8 精度的 Query, Key 张量
        q_int4, k_int4: 4-bit 精度的 Query, Key 张量
        v: 单一精度的 Value 张量
        int8_window_ratios: INT8 窗口大小比例 (0-1 的浮点数)，会对齐到128倍数
        int4_window_ratios: 4-bit 窗口大小比例 (0-1 的浮点数)，会对齐到128倍数
        sink_size: Sink 窗口大小 (固定的整数值)
    """
    seq_dim = 1 if tensor_layout == "NHD" else 2
    qo_len = q_int8.size(seq_dim)
    kv_len = k_int8.size(seq_dim)
    
    # 处理窗口比例参数
    if tensor_layout == "HND":
        H = q_int8.size(1)
    else:  # NHD
        H = q_int8.size(2)
    
    # 处理各种窗口比例参数，确保在 [0, 1] 范围内
    if isinstance(int8_window_ratios, (int, float)):
        int8_ratios_tensor = torch.full([H], float(int8_window_ratios), dtype=torch.float32, device=q_int8.device)
    elif isinstance(int8_window_ratios, (list, tuple)):
        int8_ratios_tensor = torch.tensor(int8_window_ratios, dtype=torch.float32, device=q_int8.device)
    else:
        int8_ratios_tensor = int8_window_ratios.to(device=q_int8.device, dtype=torch.float32)
    
    if isinstance(int4_window_ratios, (int, float)):
        int4_ratios_tensor = torch.full([H], float(int4_window_ratios), dtype=torch.float32, device=q_int8.device)
    elif isinstance(int4_window_ratios, (list, tuple)):
        int4_ratios_tensor = torch.tensor(int4_window_ratios, dtype=torch.float32, device=q_int8.device)
    else:
        int4_ratios_tensor = int4_window_ratios.to(device=q_int8.device, dtype=torch.float32)
    
    # 确保比例在合理范围内
    int8_ratios_tensor = torch.clamp(int8_ratios_tensor, 0.0, 1.0)
    int4_ratios_tensor = torch.clamp(int4_ratios_tensor, 0.0, 1.0)
    
    # sink_size保持为固定整数值
    if not isinstance(sink_size, int) or sink_size < 0:
        raise ValueError(f"sink_size must be a non-negative integer, got {sink_size}")
    
    # 输入预处理
    head_dim_og = q_int8.size(-1)
    if head_dim_og < 64:
        pad_size = 64 - head_dim_og
        q_int8, k_int8 = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q_int8, k_int8)]
        q_int4, k_int4 = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q_int4, k_int4)]
        v = torch.nn.functional.pad(v, (0, pad_size))
    elif head_dim_og > 64 and head_dim_og < 128:
        pad_size = 128 - head_dim_og
        q_int8, k_int8 = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q_int8, k_int8)]
        q_int4, k_int4 = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q_int4, k_int4)]
        v = torch.nn.functional.pad(v, (0, pad_size))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    # 缩放
    sm_scale = 1.0 / (head_dim_og ** 0.5) * 1.44269504
    q_int8 = q_int8 * sm_scale
    q_int4 = q_int4 * sm_scale
    
    o = torch.empty(q_int8.shape, dtype=output_dtype, device=q_int8.device)
    
    if tensor_layout == "HND":
        b, h_qo, _, head_dim = q_int8.shape
        _, h_kv, _, _ = k_int8.shape
        stride_bz_q, stride_h_q, stride_seq_q = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, _, h_qo, head_dim = q_int8.shape
        _, _, h_kv, _ = k_int8.shape
        stride_bz_q, stride_h_q, stride_seq_q = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    
    num_kv_groups = h_qo // h_kv
    lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q_int8.device) if return_lse else torch.empty([0], dtype=torch.float32, device='cpu')
    
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    
    attn_prefill_fwd[grid](
        q_int8, k_int8, q_int4, k_int4, v, o, lse, int8_ratios_tensor, int4_ratios_tensor,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        qo_len, kv_len, h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
        STAGE=4,  # 总共 4 个 STAGE
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
        is_P_fp8_quant=is_P_fp8_quant,
        SINK_SIZE=sink_size
    )
    
    o = o[..., :head_dim_og]
    return (o, lse) if return_lse else o


# 使用示例
# 相对窗口比例版本（窗口大小会自动对齐到128的倍数）
"""
output = attn_hierarchical_window(q_int8, k_int8, q_int4, k_int4, v,
                                int8_window_ratios=0.25,  # 25%的序列长度，会对齐到128倍数
                                int4_window_ratios=0.5,   # 50%的序列长度，会对齐到128倍数
                                sink_size=64)             # 固定的sink大小
"""