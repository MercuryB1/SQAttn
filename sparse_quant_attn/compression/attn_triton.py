import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, kv_len,
                    K_ptrs,  V_ptrs, stride_kn, stride_vn, 
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr, is_P_fp8_quant: tl.constexpr,
                    ):
    """
    Attention 前向计算的内部循环函数 (在 Triton JIT kernel 中运行)。
    这个函数负责计算一个 Query 块与多个 Key/Value 块的注意力。
    它实现了 FlashAttention 中的在线 Softmax (online softmax) 算法，
    通过迭代更新累加器 `acc`、行最大值 `m_i` 和行累加和 `l_i` 来避免实例化大的 QK^T 矩阵。

    Args:
        acc (tl.tensor): 累加器，存储当前 Query 块的输出 O。形状为 [BLOCK_M, HEAD_DIM]。
        l_i (tl.tensor): 行累加和，是在线 Softmax 的分母部分。形状为 [BLOCK_M]。
        m_i (tl.tensor): 行最大值，用于稳定 Softmax 计算。形状为 [BLOCK_M]。
        q (tl.tensor): 一个 Query 块。形状为 [BLOCK_M, HEAD_DIM]。
        kv_len (int): Key/Value 序列的完整长度。
        K_ptrs (tl.pointer): 指向当前 Key 块的指针。
        V_ptrs (tl.pointer): 指向当前 Value 块的指针。
        stride_kn (int): Key 矩阵在序列长度维度上的步长。
        stride_vn (int): Value 矩阵在序列长度维度上的步长。
        start_m (int): 当前 Query 块的起始索引。
        BLOCK_M (tl.constexpr): Query 块的大小。
        HEAD_DIM (tl.constexpr): 注意力头的维度。
        BLOCK_N (tl.constexpr): Key/Value 块的大小。
        STAGE (tl.constexpr): 用于控制计算范围，实现 Causal Mask (因果掩码)。
                               STAGE=1: 计算对角线之前的块。
                               STAGE=2: 计算对角线上的块。
        offs_m (tl.constexpr): 当前 Query 块内每行的偏移量。
        offs_n (tl.constexpr): 当前 Key/Value 块内每列的偏移量。
        is_P_fp8_quant (tl.constexpr): 是否对中间的注意力概率矩阵 P 进行 FP8 量化。
    """
    # 根据 STAGE 决定内循环的范围，这是为了高效实现因果掩码
    if STAGE == 1:
        # STAGE 1: 处理对角线之前的 Key/Value 块。
        # 对于第 `start_m` 个 Query 块，它只能关注 `0` 到 `start_m * BLOCK_M` 之间的 Key。
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # STAGE 2: 处理对角线所在的 Key/Value 块。
        # Query 块只能关注到自己位置之前的 Key。
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        # 确保 lo 是 BLOCK_M 的倍数，并相应地移动 K/V 指针
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    
    # --- 主循环：迭代遍历 Key/Value 序列，每次处理一个 BLOCK_N 大小的块 ---
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # --- 1. 加载 Key 块并计算 QK^T ---
        # 创建掩码，防止加载超出 kv_len 范围的数据
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask=k_mask)
        # 计算点积 QK^T，得到当前块的 attention scores
        qk = tl.dot(q, k).to(tl.float32) 

        # --- 2. 应用因果掩码并更新在线 Softmax 的统计量 ---
        if STAGE == 2:
            # 在对角线块上，应用精确的因果掩码。
            # offs_m 是行偏移，offs_n 是列偏移。如果行索引小于列索引，则该位置被掩码。
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            # 将被掩码的位置设置为一个很大的负数，这样在 softmax 后会趋近于 0
            qk = qk + tl.where(mask, 0, -1.0e6)
        
        # 在线 Softmax 算法：
        # a. 找到当前块和历史最大值中的新最大值 m_ij
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # b. 从 qk 中减去新的最大值，以提高数值稳定性
        qk -= m_ij[:, None]
        
        # --- 3. 计算 P = softmax(QK^T) 和更新累加器 ---
        # c. 计算 p = exp2(qk)，因为 q 已经被 sm_scale * (1/log(2)) 缩放过
        p = tl.math.exp2(qk)
        # d. 计算当前块的 softmax 分母 l_ij
        l_ij = tl.sum(p, 1)
        
        # e. 更新全局的 softmax 统计量
        #    alpha 用于缩放旧的累加值，以适应新的最大值 m_ij
        alpha = tl.math.exp2(m_i - m_ij)
        #    更新全局分母 l_i
        l_i = l_i * alpha + l_ij
        
        # f. 缩放旧的输出累加器 acc
        acc = acc * alpha[:, None]
        
        # --- 4. 计算 P * V 并累加到输出 ---
        # 加载对应的 Value 块
        v_mask = offs_n[:, None] < (kv_len - start_n)
        v = tl.load(V_ptrs, mask=v_mask)
        
        # 可选：将 P 矩阵量化为 FP8 再转换回 FP16，以模拟硬件支持P fp8量化
        if is_P_fp8_quant:
            p = (p*448).to(tl.float8e4nv).to(tl.float16)/448
        
        # 计算当前块的输出 (p * v) 并加到总累加器 acc 上
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)   
        
        # --- 5. 更新循环变量 ---
        # 将当前块的最大值 m_ij 作为下一次迭代的旧最大值 m_i
        m_i = m_ij
        # 移动 K 和 V 的指针到下一个块
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
        
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, Out, Lse,
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
              ):
    """
    这个 kernel 启动并行计算，每个 program instance 负责计算输出矩阵 O 的一个块。

    Args:
        Q, K, V, Out: 输入的 Query, Key, Value 和输出的 Output 张量。
        Lse: Log-Sum-Exp，用于反向传播计算。形状为 [B, H, N]。
        stride_*: 各个张量在不同维度上的步长 (stride)。
        qo_len, kv_len: Query 和 Key/Value 的序列长度。
        H: 总的 Query 头数。
        num_kv_groups: KV 组的数量，用于 Grouped-Query Attention (GQA)。
        HEAD_DIM, BLOCK_M, BLOCK_N: 编译时常量，定义头维度和计算块大小。
        STAGE: 用于控制 kernel 行为，这里似乎与外部传入的 stage 参数有关。
        RETURN_LSE: 是否计算并返回 LSE。
        is_P_fp8_quant: 是否对 P 进行 FP8 量化。
    """
    # --- 1. 获取当前 program instance 的 ID，确定计算任务 ---
    # 每个 program instance 处理一个 Query 块
    start_m = tl.program_id(0)
    # 获取当前处理的 head 和 batch 的索引
    off_z = tl.program_id(2) # Batch 维度
    off_h = tl.program_id(1) # Head 维度

    # --- 2. 计算内存指针 ---
    # 计算当前 Query 块中每一行的偏移量
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 计算 Key/Value 块中每一列/行的偏移量
    offs_n = tl.arange(0, BLOCK_N)
    # 计算 Head 维度内的偏移量
    offs_k = tl.arange(0, HEAD_DIM)
    
    # 根据偏移量和步长，计算指向 Q, K, V, O 具体数据块的指针
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    # 对于 K/V，需要处理 GQA，即多个 Query head 共享一个 KV head
    # (off_h // num_kv_groups) 确定当前 Query head 对应的 KV head 索引
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    # --- 3. 初始化在线 Softmax 的统计量 ---
    # m_i: 初始化为负无穷，确保第一次计算时任何数都比它大
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l_i: 初始化为 1.0，因为 exp(0)=1，这是 softmax 分母的初始值
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc: 初始化为 0，用于累加 P*V 的结果
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # --- 4. 加载当前 Query 块 ---
    # 使用掩码确保不加载超出序列长度的无效数据
    q_mask = offs_m[:, None] < qo_len
    q = tl.load(Q_ptrs, mask=q_mask)
    
    # --- 5. 调用内部循环执行分块计算 ---
    # 这里调用了两次 _attn_fwd_inner，这是一种常见的优化策略，
    # 可能是为了更好地利用指令级并行或处理特定的计算阶段。
    # 第一次调用处理对角线之前的块 (STAGE=1)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    1, offs_m, offs_n, is_P_fp8_quant) # STAGE=1 (4-3=1)

    # 第二次调用处理对角线上的块 (STAGE=2)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    2, offs_m, offs_n, is_P_fp8_quant) # STAGE=2
    
    # --- 6. 后处理和存储结果 ---
    # 将最终的累加器 acc 除以 softmax 的分母 l_i，得到最终的输出 O
    acc = acc / l_i[:, None]
    # 将计算结果写回全局内存
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    # 如果需要，计算并存储 Log-Sum-Exp (LSE)
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        # LSE(x) = log(sum(exp(x))) = log(sum(exp(x-m)*exp(m))) = log(exp(m)*sum(exp(x-m))) = m + log(sum(exp(x-m)))
        # 这里用的是 log2，所以是 m_i + log2(l_i)
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))

def attn_causal(q, k, v, tensor_layout="HND", output_dtype=torch.float16, return_lse=False, is_P_fp8_quant=False):
    """
    Attention Prefill/Prompt Processing 的 Python 封装函数。
    这个函数负责数据预处理、参数配置，并启动 Triton kernel。

    Args:
        q, k, v (torch.Tensor): 输入的 Query, Key, Value 张量。
        tensor_layout (str): 张量的维度布局，"HND" (Head, Num_tokens, Dim) 或 "NHD" (Num_tokens, Head, Dim)。
        output_dtype (torch.dtype): 输出张量的数据类型。
        return_lse (bool): 是否返回 Log-Sum-Exp。
        is_P_fp8_quant (bool): 是否在 kernel 中对 P 进行 FP8 量化。
    """
    # --- 1. 输入数据预处理和校验 ---
    head_dim_og = q.size(-1)
    # Triton kernel 通常对 head_dim 有要求（通常是 2 的幂，如 64, 128），这里进行填充以满足要求
    if head_dim_og < 64:
        pad_size = 64 - head_dim_og
        q, k, v = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q, k, v)]
    elif head_dim_og > 64 and head_dim_og < 128:
        pad_size = 128 - head_dim_og
        q, k, v = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q, k, v)]
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    seq_dim = 1 if tensor_layout == "NHD" else 2

    # --- 2. 计算 sm_scale 并应用到 Q ---
    # 这是一个有趣的优化：对 K 进行中心化 (减去均值)，可以改善数值稳定性
    km = k.mean(dim=seq_dim, keepdim=True)
    k = k - km
    
    # 标准的 attention 缩放因子
    sm_scale = 1.0 / (head_dim_og ** 0.5)
    # 关键优化：将 sm_scale 乘以 1/log(2)，为使用 exp2() 做准备
    sm_scale *= 1.44269504 
    # 将缩放因子预先乘到 q 上，这样在 kernel 中就不用再做乘法了
    q = q * sm_scale

    # --- 3. 配置 Triton Kernel 参数 ---
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3 # 这个 stage 参数似乎是传递给 kernel 的，但 kernel 内部硬编码了 STAGE=1 和 STAGE=2

    # 创建空的输出张量
    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    # --- 4. 获取张量形状和步长信息 ---
    # 根据不同的 tensor_layout，正确地解析维度和步长
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    
    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    # 如果需要，创建 LSE 张量
    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        # 传入一个空张量，kernel 内部会跳过 LSE 的计算和存储
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    # --- 5. 启动 Triton Kernel ---
    # 定义计算网格 (Grid)，决定启动多少个 program instance
    # grid = (x, y, z)
    # x: 在序列长度维度上，需要 triton.cdiv(qo_len, BLOCK_M) 个块
    # y: 在 head 维度上，需要 h_qo 个实例
    # z: 在 batch 维度上，需要 b 个实例
    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    # grid = (b,triton.cdiv(qo_len, BLOCK_M), h_qo,)
    
    _attn_fwd[grid](
        q, k, v, o, lse,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
        STAGE=stage,  
        RETURN_LSE=return_lse,
        # 根据 head_dim 选择合适的 warp 数量和 stage 数量，这是性能调优的关键
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4,
        is_P_fp8_quant=is_P_fp8_quant
    )

    # --- 6. 后处理 ---
    # 如果之前对 head_dim 进行了填充，这里需要裁剪掉填充的部分，返回原始维度的输出
    o = o[..., :head_dim_og]

    return o


import os
import math
import time
import torch
import itertools
from contextlib import nullcontext


def sdpa_reference(q, k, v, tensor_layout="HND", is_causal=True):
    seq_dim = 1 if tensor_layout == "NHD" else 2

    # 对 k 做去均值
    km = k.mean(dim=seq_dim, keepdim=True)
    k = k - km

    # 调用 PyTorch SDPA
    # 将 HND 或 NHD 转成 PyTorch 的 (B, num_heads, N, D) 期望格式
    if tensor_layout == "HND":
        # (B, H, N, D) -> (B, H, N, D)
        q_t = q
        k_t = k
        v_t = v
        attn_mask = None
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal
        )
        return out
    elif tensor_layout == "NHD":
        # (B, N, H, D) 转换为 (B, H, N, D)
        kv_repeat_n = q.size(2) // k.size(2)
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3).repeat(1,1, kv_repeat_n, 1).view_as(q_t)
        v_t = v.permute(0, 2, 1, 3).repeat(1,1, kv_repeat_n, 1).view_as(q_t)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )
        # 转回 NHD
        return out.permute(0, 2, 1, 3)
    else:
        raise ValueError("Unsupported tensor_layout")

def naive_attention_reference(q, k, v, tensor_layout="HND", is_causal=True):
    # 与 Triton/SDPA 对齐的前处理
    head_dim_og = q.size(-1)
    sm_scale = 1.0 / math.sqrt(head_dim_og)
    # sm_scale *= 1.44269504  # log2(e)
    seq_dim = 1 if tensor_layout == "NHD" else 2

    # k 去均值；q 乘缩放
    km = k.mean(dim=seq_dim, keepdim=True)
    k = k - km
    q = q * sm_scale

    if tensor_layout == "NHD":
        # 转为 (B, H, N, D)
        kv_repeat_n = q.size(2) // k.size(2)
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3).repeat(1,1, kv_repeat_n, 1).view_as(q_t)
        v_t = v.permute(0, 2, 1, 3).repeat(1,1, kv_repeat_n, 1).view_as(q_t)
    elif tensor_layout == "HND":
        q_t, k_t, v_t = q, k, v
    else:
        raise ValueError("Unsupported tensor_layout")

    B, H, N, D = q_t.shape

    # Q @ K^T
    # qk: (B, H, N, N)
    qk = torch.matmul(q_t, k_t.transpose(-1, -2))  # 按照缩放后计算

    if is_causal:
        # 下三角（含对角）为可见，其余置 -inf
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        qk = qk.masked_fill(mask, float('-inf'))

    # softmax（注意我们在 log2 缩放下，依然使用 torch 的 e-base softmax；这与 Triton 中使用 exp2+log2 只是底数不同，但等价）
    p = torch.softmax(qk, dim=-1)

    # @ V -> (B, H, N, D)
    out = torch.matmul(p, v_t)

    if tensor_layout == "NHD":
        # 转回 NHD
        out = out.permute(0, 2, 1, 3)

    return out


def _benchmark(func, warmup, iters, *args, **kwargs):
    """Helper to benchmark a function with warmup and iterations."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    # Timing
    t0 = time.perf_counter()
    for _ in range(iters):
        result = func(*args, **kwargs)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / iters
    return t, result


def benchmark_once(q, k, v, tensor_layout, dtype, warmup=10, iters=50, is_P_fp8_quant=False, return_lse=False, rtol=1e-3, atol=1e-5):
    """
    Runs a single benchmark for Triton, PyTorch SDPA, and a naive implementation,
    and returns a dictionary of performance and error metrics.
    """
    
    t_triton, o_triton = _benchmark(
        attn_causal, warmup, iters, q, k, v, 
        tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse, is_P_fp8_quant=is_P_fp8_quant
    )

    t_sdpa, o_sdpa = _benchmark(
        sdpa_reference, warmup, iters, q, k, v, 
        tensor_layout=tensor_layout, is_causal=True
    )

    t_naive, o_naive = _benchmark(
        naive_attention_reference, warmup, iters, q, k, v, 
        tensor_layout=tensor_layout, is_causal=True
    )

    # Align original head_dim and calculate error
    head_dim_og = q.size(-1)
    o_triton = o_triton[..., :head_dim_og].contiguous()
    o_sdpa = o_sdpa[..., :head_dim_og].contiguous()
    o_naive = o_naive[..., :head_dim_og].contiguous()

    def get_err_metrics(o_triton, o_ref):
        diff = (o_triton.float() - o_ref.float()).abs()
        rel_diff = diff / o_ref.abs().clamp_min(1e-6)
        return {
            "max_abs_err": diff.max().item(),
            "avg_abs_err": diff.mean().item(),
            "max_rel_err": rel_diff.max().item(),
            "avg_rel_err": rel_diff.mean().item(),
            "allclose": torch.allclose(o_triton.float(), o_ref.float(), rtol=rtol, atol=atol),
        }

    err_sdpa = get_err_metrics(o_triton, o_sdpa)
    err_naive = get_err_metrics(o_triton, o_naive)

    metrics = {
        "t_triton_ms": t_triton * 1000,
        "t_sdpa_ms": t_sdpa * 1000,
        "t_naive_ms": t_naive * 1000,
        "speedup_vs_sdpa": (t_sdpa / t_triton) if t_triton > 0 else float('inf'),
        "speedup_vs_naive": (t_naive / t_triton) if t_triton > 0 else float('inf'),
    }
    # Use a loop to add error metrics, making it more compact
    for name, err_dict in [("sdpa", err_sdpa), ("naive", err_naive)]:
        for k, v in err_dict.items():
            metrics[f"{k}_vs_{name}"] = v
            
    return metrics


def make_inputs(B, Hq, N, D, dtype, layout, kv_groups=1, device="cuda"):
    torch.manual_seed(0)
    # kv 头数
    Hk = max(1, Hq // kv_groups)
    if layout == "HND":
        q = torch.randn(B, Hq, N, D, device=device, dtype=dtype)
        k = torch.randn(B, Hk, N, D, device=device, dtype=dtype)
        v = torch.randn(B, Hk, N, D, device=device, dtype=dtype)
    elif layout == "NHD":
        q = torch.randn(B, N, Hq, D, device=device, dtype=dtype)
        k = torch.randn(B, N, Hk, D, device=device, dtype=dtype)
        v = torch.randn(B, N, Hk, D, device=device, dtype=dtype)
    else:
        raise ValueError("Unsupported layout")
    return q/10, k/10, v/10

def run_suite():
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    device = "cuda"

    # 关闭 dropout，选择可重复设置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 控制是否使用 flash/efficient 内核对比（如需纯 PyTorch 实现可关闭）
    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # torch.backends.cuda.enable_math_sdp(True)

    # 测试配置
    test_configs = [
        dict(B=B, Hq=Hq, N=N, D=D, layout=layout, dtype=dtype, g=g)
        for B, Hq, N, D, layout, dtype, g in itertools.product(
            [1, 8], [32], [1024, 4096], [128], ["NHD"], [torch.float16], [1, 8]
        )
        if Hq % g == 0 # 仅测试 Hq 可被 g 整除的情况
    ]

    print(f"Planned runs: {len(test_configs)}")

    results = []
    for i, cfg in enumerate(test_configs):
        print(f"[{i+1}/{len(test_configs)}] B={cfg['B']} Hq={cfg['Hq']} N={cfg['N']} D={cfg['D']} layout={cfg['layout']} dtype={str(cfg['dtype'])} kv_groups={cfg['g']}")

        q, k, v = make_inputs(
            cfg['B'], cfg['Hq'], cfg['N'], cfg['D'], cfg['dtype'], cfg['layout'], kv_groups=cfg['g'], device=device
        )

        rtol, atol = (1e-2, 1e-3) if cfg['dtype'] == torch.bfloat16 else (1e-3, 1e-4)
        metrics = benchmark_once(
            q, k, v, tensor_layout=cfg['layout'], dtype=cfg['dtype'], warmup=10, iters=50,
            is_P_fp8_quant=False, return_lse=False, rtol=rtol, atol=atol
        )

        results.append({**cfg, "dtype": str(cfg["dtype"]), "rtol": rtol, "atol": atol, **metrics})

        m = metrics
        print(
            f"  Times (ms): Triton={m['t_triton_ms']:.3f}, SDPA={m['t_sdpa_ms']:.3f}, Naive={m['t_naive_ms']:.3f}"
        )
        print(
            f"  Speedups: vs SDPA={m['speedup_vs_sdpa']:.2f}x, vs Naive={m['speedup_vs_naive']:.2f}x"
        )
        print(
            f"  Errors vs SDPA (allclose={m['allclose_vs_sdpa']}):\n"
            f"    abs: max={m['max_abs_err_vs_sdpa']:.3e}, avg={m['avg_abs_err_vs_sdpa']:.3e} | "
            f"    rel: max={m['max_rel_err_vs_sdpa']:.3e}, avg={m['avg_rel_err_vs_sdpa']:.3e}"
        )
        print(
            f"  Errors vs Naive (allclose={m['allclose_vs_naive']}):\n"
            f"    abs: max={m['max_abs_err_vs_naive']:.3e}, avg={m['avg_abs_err_vs_naive']:.3e} | "
            f"    rel: max={m['max_rel_err_vs_naive']:.3e}, avg={m['avg_rel_err_vs_naive']:.3e}"
        )

    # 汇总
    if results:
        avg_s_sdpa = sum(r["speedup_vs_sdpa"] for r in results) / len(results)
        avg_s_naive = sum(r["speedup_vs_naive"] for r in results) / len(results)
        all_sdpa_close = all(r["allclose_vs_sdpa"] for r in results)
        all_naive_close = all(r["allclose_vs_naive"] for r in results)
        
        print("\n--- Summary ---")
        print(f"Average speedup vs SDPA over {len(results)} runs: {avg_s_sdpa:.2f}x (allclose: {all_sdpa_close})")
        print(f"Average speedup vs Naive over {len(results)} runs: {avg_s_naive:.2f}x (allclose: {all_naive_close})")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    run_suite()