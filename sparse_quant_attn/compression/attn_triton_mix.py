import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, kv_len,
                    K_ptrs, V_ptrs, stride_kn, stride_vn, 
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr, 
                    is_P_fp8_quant: tl.constexpr,
                    current_int8_window_size,
                    current_int4_window_size,
                    SINK_SIZE: tl.constexpr
                    ):
    """
    Window Attention 前向计算的内部循环函数，支持 Sink + 分层窗口。
    """
    # 根据 STAGE 决定处理范围
    if STAGE == 1:
        # STAGE 1: 处理 Sink 窗口 (前 SINK_SIZE 个 tokens)
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        
        if SINK_SIZE > 0 and query_end > SINK_SIZE:  # 只有当前 query 块超过 sink 范围才需要处理
            lo, hi = 0, SINK_SIZE
        else:
            lo, hi = 0, 0
            
    elif STAGE == 2:
        # STAGE 2: 4-bit 窗口扩展区域 (排除 sink 和 INT8 窗口)
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        
        # 4-bit 窗口左边界
        int4_window_left = tl.maximum(SINK_SIZE, query_end - current_int4_window_size)
        # INT8 窗口左边界  
        int8_window_left = tl.maximum(SINK_SIZE, query_end - current_int8_window_size)
        
        if int4_window_left < int8_window_left:
            lo, hi = int4_window_left, int8_window_left
        else:
            lo, hi = 0, 0
            
    elif STAGE == 3:
        # STAGE 3: INT8 窗口内对角线之前的块
        query_start = start_m * BLOCK_M
        query_end = (start_m + 1) * BLOCK_M
        
        # INT8 窗口范围 (排除 sink)
        int8_window_left = tl.maximum(SINK_SIZE, query_end - current_int8_window_size)
        int8_window_right = query_start
        
        if int8_window_left < int8_window_right:
            lo, hi = int8_window_left, int8_window_right
        else:
            lo, hi = 0, 0
            
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
            query_global_idx = start_m * BLOCK_M + offs_m
            key_global_idx = start_n + offs_n
            
            # 因果掩码：只能看到自己位置之前的 key
            causal_mask = key_global_idx[None, :] <= query_global_idx[:, None]
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
def _attn_prefill_fwd(Q_int8, K_int8, Q_int4, K_int4, V, Out, Lse, Int8WindowSizes, Int4WindowSizes,
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
                     SINK_SIZE: tl.constexpr
                     ):
    """
    分层 Window Attention Prefill Kernel，支持 Sink + 多精度窗口
    """
    # --- 1. 获取当前 program instance 的 ID ---
    start_m = tl.program_id(0)
    off_z = tl.program_id(2) # Batch 维度
    off_h = tl.program_id(1) # Head 维度

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
    
    # 加载当前 head 的窗口大小
    current_int8_window_size = tl.load(Int8WindowSizes + off_h)
    current_int4_window_size = tl.load(Int4WindowSizes + off_h)
    
    # --- 5. 按 STAGE 顺序调用内部循环 ---
    # STAGE 1: Sink 窗口 (使用 INT8 精度)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_int8, kv_len, K_int8_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    1, offs_m, offs_n, is_P_fp8_quant, 
                                    current_int8_window_size, current_int4_window_size, SINK_SIZE)

    # STAGE 2: 4-bit 窗口扩展区域 (使用 4-bit 精度)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_int4, kv_len, K_int4_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                    2, offs_m, offs_n, is_P_fp8_quant,
                                    current_int8_window_size, current_int4_window_size, SINK_SIZE)
    
    # STAGE 3: INT8 窗口内对角线之前的块 (使用 INT8 精度)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_int8, kv_len, K_int8_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                    3, offs_m, offs_n, is_P_fp8_quant,
                                    current_int8_window_size, current_int4_window_size, SINK_SIZE)
    
    # STAGE 4: 对角线上的块 (使用 INT8 精度)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_int8, kv_len, K_int8_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4, offs_m, offs_n, is_P_fp8_quant,
                                    current_int8_window_size, current_int4_window_size, SINK_SIZE)
    
    # --- 6. 后处理和存储结果 ---
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def attn_hierarchical_window(q_int8, k_int8, q_int4, k_int4, v, 
                            int8_window_sizes, int4_window_sizes, sink_size,
                            tensor_layout="HND", output_dtype=torch.float16, 
                            return_lse=False, is_P_fp8_quant=False):
    """
    分层 Window Attention 接口，支持 Sink + 多精度窗口
    """
    seq_dim = 1 if tensor_layout == "NHD" else 2
    qo_len = q_int8.size(seq_dim)
    kv_len = k_int8.size(seq_dim)
    
    # 处理窗口大小参数
    if tensor_layout == "HND":
        H = q_int8.size(1)
    else:  # NHD
        H = q_int8.size(2)
    
    # 处理各种窗口大小参数
    if isinstance(int8_window_sizes, int):
        int8_sizes_tensor = torch.full([H], int8_window_sizes, dtype=torch.int32, device=q_int8.device)
    else:
        int8_sizes_tensor = int8_window_sizes.to(device=q_int8.device, dtype=torch.int32)
    
    if isinstance(int4_window_sizes, int):
        int4_sizes_tensor = torch.full([H], int4_window_sizes, dtype=torch.int32, device=q_int8.device)
    else:
        int4_sizes_tensor = int4_window_sizes.to(device=q_int8.device, dtype=torch.int32)
    
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

    # 去中心化和缩放
    km_int8 = k_int8.mean(dim=seq_dim, keepdim=True)
    k_int8 = k_int8 - km_int8
    km_int4 = k_int4.mean(dim=seq_dim, keepdim=True)
    k_int4 = k_int4 - km_int4
    
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
    _attn_prefill_fwd[grid](
        q_int8, k_int8, q_int4, k_int4, v, o, lse, int8_sizes_tensor, int4_sizes_tensor,
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


def naive_sink_window_attention(q_int8, k_int8, q_int4, k_int4, v, int8_window_size, int4_window_size, sink_size, tensor_layout="HND"):
    """
    朴素的 Sink + Window Attention 参考实现，用于验证
    """
    import torch.nn.functional as F
    
    if tensor_layout == "NHD":
        q_int8 = q_int8.permute(0, 2, 1, 3)  # B,N,H,D -> B,H,N,D
        k_int8 = k_int8.permute(0, 2, 1, 3)
        q_int4 = q_int4.permute(0, 2, 1, 3)
        k_int4 = k_int4.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
    
    B, H, N, D = q_int8.shape
    
    # 去中心化和缩放（与 Triton 版本保持一致）
    k_int8_centered = k_int8 - k_int8.mean(dim=2, keepdim=True)
    k_int4_centered = k_int4 - k_int4.mean(dim=2, keepdim=True)
    
    scale = 1.0 / (D ** 0.5)
    q_int8_scaled = q_int8 * scale
    q_int4_scaled = q_int4 * scale

    outputs = []
    
    for b in range(B):
        batch_outputs = []
        for h in range(H):
            q_int8_h = q_int8_scaled[b, h]  # [N, D]
            k_int8_h = k_int8_centered[b, h]  # [N, D] 
            q_int4_h = q_int4_scaled[b, h]  # [N, D]
            k_int4_h = k_int4_centered[b, h]  # [N, D]
            v_h = v[b, h]  # [N, D]
            
            # 初始化最终的注意力权重矩阵
            final_scores = torch.full((N, N), float('-inf'), device=q_int8.device)
            
            for i in range(N):
                # STAGE 1: Sink 窗口 (使用 INT8 精度)
                if sink_size > 0 and i >= sink_size:  # 只有超过 sink 范围的 query 才需要关注 sink
                    sink_end = min(sink_size, N)
                    sink_scores = torch.matmul(q_int8_h[i:i+1], k_int8_h[:sink_end].T)  # [1, sink_size]
                    final_scores[i, :sink_end] = torch.maximum(final_scores[i, :sink_end], sink_scores[0])
                
                # STAGE 2: 4-bit 窗口扩展区域 (使用 4-bit 精度)
                int4_window_left = max(sink_size, i - int4_window_size + 1)
                int8_window_left = max(sink_size, i - int8_window_size + 1)
                
                if int4_window_left < int8_window_left:
                    # 4-bit 扩展区域: [int4_window_left, int8_window_left)
                    int4_k = k_int4_h[int4_window_left:int8_window_left]  # 使用 4-bit K
                    int4_scores = torch.matmul(q_int4_h[i:i+1], int4_k.T)  # 使用 4-bit Q
                    final_scores[i, int4_window_left:int8_window_left] = torch.maximum(
                        final_scores[i, int4_window_left:int8_window_left], int4_scores[0])
                
                # STAGE 3: INT8 窗口内对角线之前的块 (使用 INT8 精度)
                if int8_window_left < i:
                    # INT8 窗口对角线前: [int8_window_left, i)
                    int8_k_before = k_int8_h[int8_window_left:i]
                    int8_scores_before = torch.matmul(q_int8_h[i:i+1], int8_k_before.T)
                    final_scores[i, int8_window_left:i] = torch.maximum(
                        final_scores[i, int8_window_left:i], int8_scores_before[0])
                
                # STAGE 4: 对角线上的块 (使用 INT8 精度)
                # 当前位置: [i, i+1)
                diag_scores = torch.matmul(q_int8_h[i:i+1], k_int8_h[i:i+1].T)
                final_scores[i, i] = torch.maximum(final_scores[i, i], diag_scores[0, 0])
            
            # 计算 softmax（处理 -inf）
            attn_weights = F.softmax(final_scores, dim=-1)
            
            # 计算输出
            output_h = torch.matmul(attn_weights.to(v_h.dtype), v_h)
            batch_outputs.append(output_h)
        
        outputs.append(torch.stack(batch_outputs, dim=0))
    
    output = torch.stack(outputs, dim=0)
    
    if tensor_layout == "NHD":
        output = output.permute(0, 2, 1, 3)
    
    return output


def test_simple_case():
    """使用简单测试样例：Q,K都是ones，V是arange"""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16
    
    # 简单的测试参数
    B, H, N, D = 1, 1, 8, 1  # 小尺寸方便手工验证
    sink_size = 2
    int8_window_size = 4
    int4_window_size = 6
    
    print("=== 简单测试样例 ===")
    print(f"形状: B={B}, H={H}, N={N}, D={D}")
    print(f"Sink={sink_size}, INT8窗口={int8_window_size}, INT4窗口={int4_window_size}")
    
    # 生成简单测试数据
    q_int8 = torch.ones(B, N, H, D, device=device, dtype=dtype)
    k_int8 = torch.ones(B, N, H, D, device=device, dtype=dtype)
    
    # INT4 版本稍有不同（模拟精度损失）
    q_int4 = q_int8 + 0.01  # 轻微差异
    k_int4 = k_int8 + 0.01
    
    # V 使用 arange 便于验证
    v_data = torch.arange(N * D, device=device, dtype=dtype).reshape(1, N, 1, D)
    v = v_data.expand(B, N, H, D)
    
    print(f"\n输入数据:")
    print(f"Q_int8: 全1矩阵 {q_int8.shape}")
    print(f"K_int8: 全1矩阵 {k_int8.shape}")
    print(f"V: arange重塑 {v.shape}")
    print(f"V数据: {v[0, :, 0, :].cpu().numpy()}")
    
    try:
        # 朴素参考实现
        print(f"\n=== 参考实现 ===")
        output_naive = naive_sink_window_attention(
            q_int8, k_int8, q_int4, k_int4, v,
            int8_window_size, int4_window_size, sink_size,
            tensor_layout="NHD"
        )
        print(f"参考实现输出形状: {output_naive.shape}")
        print(f"参考实现结果:\n{output_naive[0, :, 0, :].cpu().numpy()}")
        
        # Triton 实现
        print(f"\n=== Triton实现 ===")
        output_triton = attn_hierarchical_window(
            q_int8, k_int8, q_int4, k_int4, v,
            int8_window_sizes=int8_window_size,
            int4_window_sizes=int4_window_size,
            sink_size=sink_size,
            tensor_layout="NHD"
        )
        print(f"Triton实现输出形状: {output_triton.shape}")
        print(f"Triton实现结果:\n{output_triton[0, :, 0, :].cpu().numpy()}")
        
        # 比较结果
        diff = (output_triton.float() - output_naive.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\n=== 结果对比 ===")
        print(f"最大绝对误差: {max_diff:.6f}")
        print(f"平均绝对误差: {mean_diff:.6f}")
        print(f"结果一致性: {torch.allclose(output_triton.float(), output_naive.float(), rtol=1e-2, atol=1e-3)}")
        
        # 手工验证说明
        print(f"\n=== 手工验证说明 ===")
        print("由于Q,K都是全1，去中心化后K变为全0，所以QK^T也是全0")
        print("Softmax(全0) = 均匀分布，每个位置权重相等")
        print("因此输出应该是可见范围内V值的加权平均")
        
        # 分析每个位置的可见范围
        print(f"\n各位置的可见范围分析:")
        for i in range(N):
            visible_positions = []
            
            # Sink窗口
            if sink_size > 0 and i >= sink_size:
                visible_positions.extend(list(range(sink_size)))
            
            # 4-bit窗口
            int4_left = max(sink_size, i - int4_window_size + 1)
            int8_left = max(sink_size, i - int8_window_size + 1)
            if int4_left < int8_left:
                visible_positions.extend(list(range(int4_left, int8_left)))
            
            # INT8窗口
            if int8_left < i:
                visible_positions.extend(list(range(int8_left, i)))
            
            # 对角线
            visible_positions.append(i)
            
            # 去重并排序
            visible_positions = sorted(list(set(visible_positions)))
            print(f"位置 {i}: 可见 {visible_positions}")
            
            # 计算期望输出
            if visible_positions:
                expected_output = v[0, visible_positions, 0, :].mean(dim=0)
                actual_output = output_naive[0, i, 0, :]
                print(f"  期望输出: {expected_output.cpu().numpy()}")
                print(f"  实际输出: {actual_output.cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_attention_matrix():
    """调试注意力矩阵，显示具体的权重分布"""
    print("\n=== 调试注意力矩阵 ===")
    
    device = "cuda"
    dtype = torch.float16
    B, H, N, D = 1, 1, 8, 4
    sink_size = 2
    int8_window_size = 4
    int4_window_size = 6
    
    # 创建测试数据
    q_int8 = torch.ones(B, N, H, D, device=device, dtype=dtype)
    k_int8 = torch.ones(B, N, H, D, device=device, dtype=dtype)
    q_int4 = q_int8 + 0.01
    k_int4 = k_int8 + 0.01
    
    # 去中心化
    k_int8_centered = k_int8 - k_int8.mean(dim=1, keepdim=True)  # NHD格式
    k_int4_centered = k_int4 - k_int4.mean(dim=1, keepdim=True)
    
    print(f"原始K: {k_int8[0, :, 0, :].cpu().numpy()}")
    print(f"去中心化后的K: {k_int8_centered[0, :, 0, :].cpu().numpy()}")
    
    # 计算注意力分数
    scale = 1.0 / (D ** 0.5)
    q_scaled = q_int8 * scale
    
    # 计算QK^T
    scores = torch.matmul(q_scaled[0, :, 0, :], k_int8_centered[0, :, 0, :].T)
    print(f"QK^T 分数矩阵:\n{scores.cpu().numpy()}")
    
    # 应用窗口掩码并计算softmax
    import torch.nn.functional as F
    
    for i in range(N):
        # 初始化掩码
        mask = torch.full((N,), float('-inf'), device=device)
        
        # Sink窗口
        if sink_size > 0 and i >= sink_size:
            mask[:sink_size] = 0
        
        # 4-bit窗口
        int4_left = max(sink_size, i - int4_window_size + 1)
        int8_left = max(sink_size, i - int8_window_size + 1)
        if int4_left < int8_left:
            mask[int4_left:int8_left] = 0
        
        # INT8窗口
        if int8_left < i:
            mask[int8_left:i] = 0
        
        # 对角线
        mask[i] = 0
        
        # 应用掩码
        masked_scores = scores[i] + mask
        attn_weights = F.softmax(masked_scores, dim=-1)
        
        print(f"位置 {i}:")
        print(f"  掩码: {mask.cpu().numpy()}")
        print(f"  掩码后分数: {masked_scores.cpu().numpy()}")
        print(f"  注意力权重: {attn_weights.cpu().numpy()}")


if __name__ == "__main__":
    print("开始简单测试样例...")
    success = test_simple_case()
    
    if success:
        debug_attention_matrix()
        print("\n🎉 测试通过！")
    else:
        print("\n❌ 测试失败")