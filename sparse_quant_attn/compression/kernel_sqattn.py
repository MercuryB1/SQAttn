# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial


class FlashAttentionTuneSpace:

    def __init__(
        self,
        block_sizes=(64, 128, 256),
        thread_options=(128, 256, 512),
        num_stages_range=(2, 3),
        max_shared_mem=100 * 1024,
        warp_alignment=16,
        dim=128,
        dtype_bytes=2,
    ):
        self.block_sizes = block_sizes
        self.thread_options = thread_options
        self.num_stages_range = num_stages_range
        self.max_shared_mem = max_shared_mem
        self.warp_alignment = warp_alignment
        self.dim = dim
        self.dtype_bytes = dtype_bytes


def get_configs(user_config=None):
    config = user_config or FlashAttentionTuneSpace()
    valid_configs = []

    for block_M, block_N in itertools.product(config.block_sizes, repeat=2):
        for threads in config.thread_options:
            assert threads % 32 == 0
            warp_count = threads // 32
            warp_M = block_M // warp_count
            warp_N = block_N // warp_count

            if (warp_M % config.warp_alignment != 0 or warp_N % config.warp_alignment != 0):
                continue

            shared_mem = 2 * config.dtype_bytes * config.dim * (block_M + block_N)
            if shared_mem > config.max_shared_mem:
                continue

            for num_stages in config.num_stages_range:
                valid_configs.append({
                    "block_M": block_M,
                    "block_N": block_N,
                    "num_stages": num_stages,
                    "threads": threads,
                })
    return valid_configs


def sqattn(batch, heads, seq_len, dim, is_causal, bit8_window_size=128, bit4_window_size=64, sink_window_size=128, tune=False, groups=1):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    # seq_len已经是2倍长度，左半边是8bit，右半边是4bit
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    def kernel_func(block_M, block_N, num_stages, threads):
        @T.macro
        def MMA0(
            K: T.Tensor(kv_shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    q_idx = bx * block_M + i
                    kv_idx = k * block_N + j
                    
                    # 1. 8bit部分 (左半边)
                    # Causal mask: 确保只能看到之前的位置
                    causal_mask = kv_idx <= q_idx
                    # 8bit window mask: 在window范围内的token
                    bit8_window_mask = kv_idx > q_idx - bit8_window_size
                    sink_mask = kv_idx < sink_window_size
                    # 组合8bit部分的mask
                    fp8_mask = causal_mask & (sink_mask | bit8_window_mask)
                    
                    # 2. 4bit部分 (右半边)
                    # 调整索引以处理右半边
                    kv_idx_4bit = kv_idx - seq_len // 2
                    # q_idx_4bit = q_idx - seq_len // 2
                    # 8bit window mask for 4bit part: 在8bit window外面的token
                    bit8_window_mask_4bit = kv_idx_4bit <= q_idx - bit8_window_size
                    # 4bit window mask: 在4bit window里面的token
                    bit4_window_mask = kv_idx_4bit > q_idx - bit8_window_size - bit4_window_size
                    bit4_no_sink_mask = kv_idx_4bit >= sink_window_size
                    # 组合4bit部分的mask
                    int4_mask = bit4_no_sink_mask & bit8_window_mask_4bit & bit4_window_mask
                    
                    # 3. 判断当前处理的是8bit还是4bit部分
                    is_bit8_part = kv_idx < seq_len // 2  # 前半部分是8bit
                    is_valid_q_part = q_idx < seq_len // 2
                    # 4. 组合最终的mask
                    final_mask = is_valid_q_part & ((is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask))
                    # final_mask = fp8_mask |  int4_mask
                    
                    acc_s[i, j] = T.if_then_else(final_mask, 0, -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(kv_shape, dtype),
            V_shared: T.SharedBuffer([block_M, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                V: T.Tensor(kv_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    # if bx == 1:
                    T.print(acc_s, msg="acc_s")
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    if tune:

        @autotune(
            configs=get_configs(),
            keys=["block_M", "block_N", "num_stages", "threads"],
            warmup=10,
            rep=10)
        @jit(out_idx=[3], supply_type=tilelang.TensorSupplyType.Integer, ref_prog=None)
        def kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_N, num_stages, threads)

        return kernel()
    else:

        def kernel(block_M, block_N, num_stages, threads):
            return kernel_func(block_M, block_N, num_stages, threads)

        return kernel


def ref_program(Q, K, V, is_causal, groups=1):
    # Q: [B, T, HQ, D]
    # K: [B, T, HK, D]
    # V: [B, T, HV, D]
    # HQ = HKV * groups
    assert Q.size(2) == K.size(
        2) * groups, f"Q.size(2): {Q.size(2)}, K.size(2): {K.size(2)}, groups: {groups}"
    assert Q.size(2) == V.size(
        2) * groups, f"Q.size(2): {Q.size(2)}, V.size(2): {V.size(2)}, groups: {groups}"

    dim = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=14, help='batch size')
    parser.add_argument('--heads', type=int, default=28, help='heads')
    parser.add_argument('--seq_len', type=int, default=2048, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    parser.add_argument('--groups', type=int, default=7, help='groups')
    parser.add_argument('--bit8_window_size', type=int, default=128, help='window size for 8bit computation')
    parser.add_argument('--bit4_window_size', type=int, default=64, help='window size for 4bit computation')
    parser.add_argument('--test_small', action='store_true', help='run small test case')
    args = parser.parse_args()

    if args.test_small:
        # Small test case to verify mixed-bit attention
        batch, heads, seq_len, dim = 1, 1, 16, 16 # 使用更小的维度便于观察
        is_causal = True
        groups = 1
        bit8_window_size = 3  # 使用更小的window size便于观察效果
        bit4_window_size = 1  # 4bit window比8bit window小
        sink_window_size = 1
        program = sqattn(
            batch, heads, seq_len, dim, is_causal=True, 
            bit8_window_size=bit8_window_size, bit4_window_size=bit4_window_size, 
            sink_window_size=sink_window_size,
            tune=False, groups=groups)(
                block_M=16, block_N=16, num_stages=2, threads=32)
        
        kernel = tilelang.compile(program, out_idx=[3])
        
        # 创建简单的输入张量，使用不同的值便于观察
        q = torch.ones(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.ones(batch, seq_len, heads//groups, dim, device="cuda", dtype=torch.float16)
        v = torch.ones(batch, seq_len, heads//groups, dim, device="cuda", dtype=torch.float16)
        
        # 给输入添加一些变化，便于观察
        # for i in range(seq_len // 2):  # 只处理一半长度，因为seq_len已经是2倍
        #     # 8bit部分
        #     q[0, i, 0, :] = i + 1
        #     k[0, i, 0, :] = i + 1
        #     v[0, i, 0, :] = i + 1
        #     # 4bit部分
        #     q[0, i + seq_len//2, 0, :] = i + 1
        #     k[0, i + seq_len//2, 0, :] = i + 1
        #     v[0, i + seq_len//2, 0, :] = i + 1
        
        # 运行kernel
        output = kernel(q, k, v)
        
        print("Test case parameters:")
        print(f"Sequence length: {seq_len} (already doubled)")
        print(f"8bit window size: {bit8_window_size}")
        print(f"4bit window size: {bit4_window_size}")
        print(f"Input shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
        print("\nInput values:")
        print("Q[0, :, 0, 0] (8bit part):", q[0, :seq_len//2, 0, 0].cpu().numpy())
        print("Q[0, :, 0, 0] (4bit part):", q[0, seq_len//2:, 0, 0].cpu().numpy())
        print("K[0, :, 0, 0] (8bit part):", k[0, :seq_len//2, 0, 0].cpu().numpy())
        print("K[0, :, 0, 0] (4bit part):", k[0, seq_len//2:, 0, 0].cpu().numpy())
        print("\nOutput values:")
        print("Output[0, :, 0, 0]:", output[0, :, 0, 0].cpu().numpy())
        
    else:
        batch, heads, seq_len, dim, is_causal, groups = args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups
        flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
        total_flops = 2 * flops_per_matmul
        if is_causal:
            total_flops *= 0.5

        if (not args.tune):
            program = sqattn(
                batch, heads, seq_len, dim, is_causal=True, 
                bit8_window_size=args.bit8_window_size, bit4_window_size=args.bit4_window_size,
                tune=args.tune, groups=groups)(
                    block_M=128, block_N=128, num_stages=2, threads=128)
            ref_program = partial(ref_program, is_causal=is_causal, groups=groups)
            kernel = tilelang.compile(program, out_idx=[3])
            q = torch.randn(14, 4096, 28, 128, device="cuda", dtype=torch.float16)
            k = torch.randn(14, 4096, 4, 128, device="cuda", dtype=torch.float16)
            v = torch.randn(14, 4096, 4, 128, device="cuda", dtype=torch.float16)
            kernel(q,k,v)