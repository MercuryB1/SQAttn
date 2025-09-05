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
    """原始的 Prefill 内部循环函数"""
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask=k_mask)
        qk = tl.dot(q, k).to(tl.float32) 

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_mask = offs_n[:, None] < (kv_len - start_n)
        v = tl.load(V_ptrs, mask=v_mask)
        
        if is_P_fp8_quant:
            p = (p*448).to(tl.float8e4nv).to(tl.float16)/448
        
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32)   
        
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
        
    return acc, l_i, m_i


@triton.jit
def _attn_prefill_fwd(Q, K, V, Out, Lse,
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
                     is_P_fp8_quant: tl.constexpr):
    """原始 Prefill Kernel"""
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q_mask = offs_m[:, None] < qo_len
    q = tl.load(Q_ptrs, mask=q_mask)
    
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    1, offs_m, offs_n, is_P_fp8_quant)

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    2, offs_m, offs_n, is_P_fp8_quant)
    
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


@triton.jit  
def _attn_decode_fwd(Q, K, V, Out, Lse,
                    stride_qz, stride_qh, stride_qn,
                    stride_kz, stride_kh, stride_kn,  
                    stride_vz, stride_vh, stride_vn,  
                    stride_oz, stride_oh, stride_on,  
                    qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr, 
                    HEAD_DIM: tl.constexpr,  
                    BLOCK_N: tl.constexpr,  
                    RETURN_LSE: tl.constexpr,
                    is_P_fp8_quant: tl.constexpr):
    """Decode 专用 Kernel"""
    off_z = tl.program_id(0)  # batch
    off_h = tl.program_id(1)  # head

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # 单个 Query
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_k
    O_ptrs = Out + (off_z * stride_oz + off_h * stride_oh) + offs_k
    
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs)
    
    K_base = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh)
    V_base = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh)
    
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k_mask = (start_n + offs_n) < kv_len
        K_ptrs = K_base + (start_n + offs_n)[None, :] * stride_kn + offs_k[:, None]
        k = tl.load(K_ptrs, mask=k_mask[None, :])
        
        qk = tl.sum(q[:, None] * k, axis=0).to(tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(qk))
        qk = qk - m_ij
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha
        
        v_mask = (start_n + offs_n) < kv_len
        V_ptrs = V_base + (start_n + offs_n)[:, None] * stride_vn + offs_k[None, :]
        v = tl.load(V_ptrs, mask=v_mask[:, None])
        
        if is_P_fp8_quant:
            p = (p*448).to(tl.float8e4nv).to(tl.float16)/448
        
        acc += tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_ij
    
    acc = acc / l_i
    tl.store(O_ptrs, acc.to(Out.type.element_ty))
    
    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len)
        lse_val = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, lse_val)


def attn_unified(q, k, v, tensor_layout="HND", output_dtype=torch.float16, 
                return_lse=False, is_P_fp8_quant=False):
    """统一的 FlashAttention 接口"""
    seq_dim = 1 if tensor_layout == "NHD" else 2
    qo_len = q.size(seq_dim)
    kv_len = k.size(seq_dim)
    is_decode = (qo_len == 1)
    
    # 输入预处理
    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        pad_size = 64 - head_dim_og
        q, k, v = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q, k, v)]
    elif head_dim_og > 64 and head_dim_og < 128:
        pad_size = 128 - head_dim_og
        q, k, v = [torch.nn.functional.pad(t, (0, pad_size)) for t in (q, k, v)]
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    
    km = k.mean(dim=seq_dim, keepdim=True)
    k = k - km
    sm_scale = 1.0 / (head_dim_og ** 0.5) * 1.44269504
    q = q * sm_scale

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, _, head_dim = q.shape
        _, h_kv, _, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, _, h_qo, head_dim = q.shape
        _, _, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)

    num_kv_groups = h_qo // h_kv
    lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device) if return_lse else torch.empty([0], dtype=torch.float32, device='cpu')

    if is_decode:
        BLOCK_N = 64
        grid = (b, h_qo)
        _attn_decode_fwd[grid](
            q, k, v, o, lse,
            stride_bz_q, stride_h_q, stride_seq_q, 
            stride_bz_k, stride_h_k, stride_seq_k,  
            stride_bz_v, stride_h_v, stride_seq_v,  
            stride_bz_o, stride_h_o, stride_seq_o,
            qo_len, kv_len, h_qo, num_kv_groups,
            BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
            RETURN_LSE=return_lse,
            num_warps=4 if head_dim <= 64 else 8,
            num_stages=3,
            is_P_fp8_quant=is_P_fp8_quant
        )
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
        _attn_prefill_fwd[grid](
            q, k, v, o, lse,
            stride_bz_q, stride_h_q, stride_seq_q, 
            stride_bz_k, stride_h_k, stride_seq_k,  
            stride_bz_v, stride_h_v, stride_seq_v,  
            stride_bz_o, stride_h_o, stride_seq_o,
            qo_len, kv_len, h_qo, num_kv_groups,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,  
            STAGE=3,  
            RETURN_LSE=return_lse,
            num_warps=4 if head_dim == 64 else 8,
            num_stages=4,
            is_P_fp8_quant=is_P_fp8_quant
        )

    o = o[..., :head_dim_og]
    return (o, lse) if return_lse else o


def attn_causal(q, k, v, tensor_layout="HND", output_dtype=torch.float16, return_lse=False, is_P_fp8_quant=False):
    return attn_unified(q, k, v, tensor_layout, output_dtype, return_lse, is_P_fp8_quant)


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
    if tensor_layout == "HND":
        # 处理 GQA
        kv_repeat_n = q.size(1) // k.size(1)
        q_t = q
        k_t = k
        v_t = v
        if kv_repeat_n > 1:
            k_t = k_t.repeat(1, kv_repeat_n, 1, 1)
            v_t = v_t.repeat(1, kv_repeat_n, 1, 1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )
        return out
    elif tensor_layout == "NHD":
        kv_repeat_n = q.size(2) // k.size(2)
        q_t = q.permute(0, 2, 1, 3)  # (B, N, H, D) -> (B, H, N, D)
        k_t = k.permute(0, 2, 1, 3)  # (B, kv_len, H_kv, D) -> (B, H_kv, kv_len, D)
        v_t = v.permute(0, 2, 1, 3)  # (B, kv_len, H_kv, D) -> (B, H_kv, kv_len, D)
        if kv_repeat_n > 1:
            k_t = k_t.repeat(1, kv_repeat_n, 1, 1)  # (B, H_kv, kv_len, D) -> (B, H, kv_len, D)
            v_t = v_t.repeat(1, kv_repeat_n, 1, 1)  # (B, H_kv, kv_len, D) -> (B, H, kv_len, D)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )
        return out.permute(0, 2, 1, 3)  # (B, H, N, D) -> (B, N, H, D)
    else:
        raise ValueError("Unsupported tensor_layout")

def naive_attention_reference(q, k, v, tensor_layout="HND", is_causal=True):
    head_dim_og = q.size(-1)
    sm_scale = 1.0 / math.sqrt(head_dim_og)
    seq_dim = 1 if tensor_layout == "NHD" else 2

    km = k.mean(dim=seq_dim, keepdim=True)
    k = k - km
    q = q * sm_scale

    if tensor_layout == "NHD":
        kv_repeat_n = q.size(2) // k.size(2)
        q_t = q.permute(0, 2, 1, 3)  # (B, N, H, D) -> (B, H, N, D)
        k_t = k.permute(0, 2, 1, 3)  # (B, kv_len, H_kv, D) -> (B, H_kv, kv_len, D)
        v_t = v.permute(0, 2, 1, 3)  # (B, kv_len, H_kv, D) -> (B, H_kv, kv_len, D)
        if kv_repeat_n > 1:
            k_t = k_t.repeat(1, kv_repeat_n, 1, 1)  # (B, H_kv, kv_len, D) -> (B, H, kv_len, D)
            v_t = v_t.repeat(1, kv_repeat_n, 1, 1)  # (B, H_kv, kv_len, D) -> (B, H, kv_len, D)
    elif tensor_layout == "HND":
        kv_repeat_n = q.size(1) // k.size(1)
        q_t, k_t, v_t = q, k, v
        if kv_repeat_n > 1:
            k_t = k_t.repeat(1, kv_repeat_n, 1, 1)
            v_t = v_t.repeat(1, kv_repeat_n, 1, 1)
    else:
        raise ValueError("Unsupported tensor_layout")

    B, H, qo_len, D = q_t.shape
    _, _, kv_len, _ = k_t.shape
    qk = torch.matmul(q_t, k_t.transpose(-1, -2))  # (B, H, qo_len, kv_len)

    if is_causal:
        # 创建适合 qo_len x kv_len 的因果掩码
        mask = torch.zeros(qo_len, kv_len, device=q.device, dtype=torch.bool)
        if qo_len == 1:  # Decode 模式：单个 Query 可以看到所有之前的 KV
            pass  # 不需要掩码
        else:  # Prefill 模式：标准的下三角掩码
            for i in range(qo_len):
                for j in range(kv_len):
                    if j > i:  # 只掩盖未来的位置
                        mask[i, j] = True
        qk = qk.masked_fill(mask, float('-inf'))

    p = torch.softmax(qk, dim=-1)
    out = torch.matmul(p, v_t)

    if tensor_layout == "NHD":
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
        attn_unified, warmup, iters, q, k, v, 
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


def make_inputs(B, Hq, N, D, dtype, layout, kv_groups=1, device="cuda", kv_len=None):
    torch.manual_seed(0)
    # kv 头数
    Hk = max(1, Hq // kv_groups)
    if kv_len is None:
        kv_len = N
        
    if layout == "HND":
        q = torch.randn(B, Hq, N, D, device=device, dtype=dtype)
        k = torch.randn(B, Hk, kv_len, D, device=device, dtype=dtype)
        v = torch.randn(B, Hk, kv_len, D, device=device, dtype=dtype)
    elif layout == "NHD":
        q = torch.randn(B, N, Hq, D, device=device, dtype=dtype)
        k = torch.randn(B, kv_len, Hk, D, device=device, dtype=dtype)
        v = torch.randn(B, kv_len, Hk, D, device=device, dtype=dtype)
    else:
        raise ValueError("Unsupported layout")
    return q/10, k/10, v/10

def run_suite():
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    device = "cuda"

    # 关闭 dropout，选择可重复设置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 测试配置
    test_configs = [
        dict(B=B, Hq=Hq, N=N, D=D, layout=layout, dtype=dtype, g=g)
        for B, Hq, N, D, layout, dtype, g in itertools.product(
            [1, 8], [32], [1024, 4096], [128], ["NHD"], [torch.float16], [1, 8]
        )
        if Hq % g == 0  # 仅测试 Hq 可被 g 整除的情况
    ]
    
    # 添加 decode 测试
    decode_configs = [
        dict(B=B, Hq=Hq, N=1, kv_len=kv_len, D=D, layout=layout, dtype=dtype, g=g)
        for B, Hq, kv_len, D, layout, dtype, g in itertools.product(
            [1, 8], [32], [1024, 4096], [128], ["NHD"], [torch.float16], [1, 8]
        )
        if Hq % g == 0
    ]
    
    all_configs = test_configs + decode_configs

    print(f"Planned runs: {len(all_configs)} (Prefill: {len(test_configs)}, Decode: {len(decode_configs)})")

    results = []
    for i, cfg in enumerate(all_configs):
        kv_len_info = f" kv_len={cfg.get('kv_len', cfg['N'])}" if 'kv_len' in cfg else ""
        print(f"[{i+1}/{len(all_configs)}] B={cfg['B']} Hq={cfg['Hq']} N={cfg['N']}{kv_len_info} D={cfg['D']} layout={cfg['layout']} dtype={str(cfg['dtype'])} kv_groups={cfg['g']}")

        q, k, v = make_inputs(
            cfg['B'], cfg['Hq'], cfg['N'], cfg['D'], cfg['dtype'], cfg['layout'], 
            kv_groups=cfg['g'], device=device, kv_len=cfg.get('kv_len')
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