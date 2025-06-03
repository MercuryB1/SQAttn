import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from sparse_quant_attn.compression.sq_attn import sparsequantattn_prefill, calculate_attn_params
from loguru import logger
from sparse_quant_attn.compression.kernel import flashattn
from sparse_quant_attn.compression.kernel_sqattn import sqattn
from sparse_quant_attn.compression.fake_quant import quantize_activation_per_token_absmax, pseudo_quantize_tensor
import tilelang
import tilelang.language as T
from torch.nn.functional import scaled_dot_product_attention
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import copy


def int4_quant(x):
    # 计算张量中绝对值的最大值
    max_abs_val = torch.max(torch.abs(x))

    # 如果最大绝对值非常小 (例如，接近0)，说明张量本身接近全零
    # 此时直接返回原张量，避免 scale 为 0 或过小导致后续计算出现 NaN/Inf
    if max_abs_val < 1e-9: # 使用一个小的阈值判断是否接近零
        return torch.zeros_like(x)  # 返回全零的张量，保持原形状和数据类型

    # 计算缩放因子 (scale)
    # 对于 int4 对称量化，我们将数值映射到 [-7, 7] 这个范围 (2^(4-1) - 1 = 7)
    scale = max_abs_val / 7.0

    # 量化：将输入值除以 scale，然后四舍五入到最近的整数
    quantized_x = torch.round(x / scale)

    # 裁剪：将量化后的值限制在 int4 的对称表示范围内 [-7, 7]
    clamped_x = torch.clamp(quantized_x, -7.0, 7.0)

    # 反量化：将裁剪后的值乘以 scale，恢复到原始的数值范围（但精度已降低）
    dequantized_x = clamped_x * scale

    return dequantized_x

def fp8_quant(x):
    x_fp8_sim=x.to(torch.float8_e4m3fn)
    y=x_fp8_sim.to(torch.bfloat16)
    return y


@torch.no_grad()
def replace_flashattn_kernel_for_block(module: nn.Module, input_features: dict, blockidx: int, bit8_window_size=0, bit4_window_size=0, sink_window_size=0, args=None):
    if isinstance(module, Qwen2DecoderLayer):
        from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS
        # import pdb; pdb.set_trace()
        impl_name = f"sparsequantattn_{blockidx}"
        ALL_ATTENTION_FUNCTIONS[impl_name] = delayed_kernel_wrapper(blockidx, bit8_window_size, bit4_window_size, sink_window_size, args)
        module.self_attn.config._attn_implementation = impl_name
        # impl_name = f"sparsequantattn_{blockidx}"
        # # 替换attention函数
        # ALL_ATTENTION_FUNCTIONS[impl_name] = kernel
        # module.self_attn.config._attn_implementation = impl_name


@torch.no_grad()
def replace_attn_for_block(module: nn.Module, input_features: dict, blockidx: int):
    if isinstance(module, Qwen2DecoderLayer):
        from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS
        bit8_window_size, bit4_window_size, sink_size = calculate_attn_params(input_features)
        attn_fn = make_layer_attention_wrapper(module, bit8_window_size, bit4_window_size, sink_size)
        impl_name = f"sparsequantattn_{blockidx}"
        # 替换attention函数
        ALL_ATTENTION_FUNCTIONS[impl_name] = attn_fn
        module.self_attn.config._attn_implementation = impl_name


@torch.no_grad()
def replace_sdpa_for_block(module: nn.Module, blockidx: int, args, bit8_window_size=0, bit4_window_size=0, sink_window_size=0):
    if isinstance(module, Qwen2DecoderLayer):
        from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS
        attn_fn = delayed_sdpa_wrapper(blockidx, args=args, bit8_window_size=bit8_window_size, bit4_window_size=bit4_window_size, sink_window_size=sink_window_size)
        impl_name = f"sparsequantattn_{blockidx}"
        ALL_ATTENTION_FUNCTIONS[impl_name] = attn_fn
        module.self_attn.config = copy.deepcopy(module.self_attn.config)
        # module.self_attn.config._attn_implementation = impl_name
        module.self_attn.config._attn_implementation = impl_name


@torch.no_grad()
def make_layer_attention_wrapper(module, bit8_window_size, bit4_window_size, sink_size):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    if isinstance(module, Qwen2DecoderLayer):
        def qwen2_attention_wrapper(
            self_module,         # Qwen2Attention 实例
            q, k, v,
            attn_mask,
            dropout=0.0,
            scaling=1.0,
            sliding_window=None,
            **kwargs
        ):
            # logger.info(f"[Layer {self_module.layer_idx}] Using bit8={bit8_window_size}, bit4={bit4_window_size}, sink={sink_size}")
            
            # 只取 q, k, v -> 传给 sparsequantattn_prefill
            attn_output, _ = sparsequantattn_prefill(
                q, k, v,
                bit8_window_size=bit8_window_size,
                bit4_window_size=bit4_window_size,
                sink_size=sink_size
            )
            return attn_output, None  # 兼容Qwen2接口（不返回attn权重）
        
        return qwen2_attention_wrapper
    else:
        raise NotImplementedError(type(module))



def delayed_kernel_wrapper(layer_idx, bit8_window_size=0, bit4_window_size=0, sink_window_size=0, args=None):
    kernel_holder = {"compiled": None}

    def attention_fn(
        self_module, q, k, v, attn_mask,
        dropout=0.0, scaling=1.0, sliding_window=None, **kwargs
    ):  

        q = q.transpose(1, 2)  # (B, S, H, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if args.quant:
            # 对q, k, v进行裁剪，绝对值最大不超过1000
            q = torch.clamp(q, min=-1000, max=1000)
            k = torch.clamp(k, min=-1000, max=1000)
            v = torch.clamp(v, min=-1000, max=1000)
            # 消除nan和正负inf
            q = torch.nan_to_num(q, nan=0.0, posinf=1000.0, neginf=-1000.0)
            k = torch.nan_to_num(k, nan=0.0, posinf=1000.0, neginf=-1000.0)
            v = torch.nan_to_num(v, nan=0.0, posinf=1000.0, neginf=-1000.0)
            # TODO quantize q, k, v
            # q_int8 = quantize_activation_per_token_absmax(q, a_bits=8)
            # q_int8 = fp8_quant(q).to(torch.float16)
            q_int8, _, _ = pseudo_quantize_tensor(q, group_size=1, zero_point=True, bit_width=8)
            km = k.mean(dim=1, keepdim=True)
            k = k - km
            # k_int8 = quantize_activation_per_token_absmax(k, a_bits=8)
            # v_int8 = quantize_activation_per_token_absmax(v, a_bits=8)
            # k_int4 = quantize_activation_per_token_absmax(k, a_bits=4)
            # v_int4 = quantize_activation_per_token_absmax(v, a_bits=4)
            # k_int8 = fp8_quant(k).to(torch.float16)
            # v_int8 = fp8_quant(v).to(torch.float16) 
            # k_int4 = int4_quant(k).to(torch.float16)
            # v_int4 = int4_quant(v).to(torch.float16)
            k_int8, _, _ = pseudo_quantize_tensor(k, group_size=1, zero_point=True, bit_width=8)
            v_int8, _, _ = pseudo_quantize_tensor(v, group_size=1, zero_point=True, bit_width=8)
            k_int4, _, _ = pseudo_quantize_tensor(k, group_size=1, zero_point=True, bit_width=4)
            v_int4, _, _ = pseudo_quantize_tensor(v, group_size=1, zero_point=True, bit_width=4)
            # import pdb; pdb.set_trace()
            q_int8_int8 = torch.cat([q_int8, q_int8], dim=1)
            k_int8_int4 = torch.cat([k_int8, k_int4], dim=1)
            v_int8_int4 = torch.cat([v_int8, v_int4], dim=1)

            # q_int8_int8 = torch.cat([q, q], dim=1)
            # k_int8_int4 = torch.cat([k, k], dim=1)
            # v_int8_int4 = torch.cat([v, v], dim=1)
            

            B, S, H, D = q_int8_int8.shape
            # print(f"[Compile Kernel] Layer {layer_idx}: B={B}, H={H}, S={S}, D={D}")


            # bit8_window_size = 2000
            # bit4_window_size = 48
            # sink_size = 8
            if kernel_holder["compiled"] is None:
                logger.info(f"[Compile Kernel] Layer {layer_idx}: B={B}, H={H}, S={S}, D={D}, bit8_window_size={bit8_window_size}, bit4_window_size={bit4_window_size}, sink_window_size={sink_window_size}")
                if args.dynamic_shape:
                    program = sqattn(
                        T.symbolic("b"), H, T.symbolic("s"), D,
                        is_causal=True,
                        bit8_window_size=bit8_window_size,
                        bit4_window_size=bit4_window_size,
                        sink_size=sink_window_size,
                        tune=False,
                        groups=7
                    )(block_M=128, block_N=128, num_stages=2, threads=128)
                else:
                    program = sqattn(
                        B, H, S, D,
                        is_causal=True,
                        bit8_window_size=bit8_window_size,
                        bit4_window_size=bit4_window_size,
                        sink_window_size=sink_window_size,
                        tune=False,
                        groups=7
                    )(block_M=128, block_N=128, num_stages=2, threads=128)
                kernel_holder["compiled"] = tilelang.compile(program, out_idx=[3])
                # 调用已编译 kernel
                attn_output = kernel_holder["compiled"](q_int8_int8, k_int8_int4, v_int8_int4)[:, :S//2, :, :]
                
                
                # import pdb; pdb.set_trace()
            return kernel_holder["compiled"](q_int8_int8, k_int8_int4, v_int8_int4)[:, :S//2, :, :], None
        
        else:
            if kernel_holder["compiled"] is None:
                # 动态提取实际 shape
                B, S, H, D = q.shape
                print(f"[Compile Kernel] Layer {layer_idx}: B={B}, H={H}, S={S}, D={D}")
                # dynamic shape
                if args.dynamic_shape:
                    program = flashattn(
                        T.symbolic("b"), H, T.symbolic("s"), D,
                        is_causal=True,
                        tune=False,
                        groups=7
                    )(block_M=128, block_N=128, num_stages=2, threads=128)
                else:
                    program = flashattn(
                        B, H, S, D,
                        is_causal=True,
                        tune=False,
                        groups=7
                    )(block_M=128, block_N=128, num_stages=2, threads=128)
                
                kernel_holder["compiled"] = tilelang.compile(program, out_idx=[3])

            # 调用已编译 kernel
            return kernel_holder["compiled"](q, k, v), None


            

    return attention_fn




def delayed_sdpa_wrapper(layer_idx, bit8_window_size=0, bit4_window_size=0, sink_window_size=0, args=None):
    def construct_mix_bit_mask(seq_len, bit8_window_size, bit4_window_size, sink_window_size, device='cuda'):
        """
        返回 shape = (L, L) 的 mask，True表示可以attend，False表示mask掉。
        """
        q_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # (L, 1)
        kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, L)

        # --- FP8 左半边 ---
        causal_mask = kv_idx <= q_idx
        bit8_window_mask = kv_idx > (q_idx - bit8_window_size)
        sink_mask = kv_idx < sink_window_size
        fp8_mask = causal_mask & (sink_mask | bit8_window_mask)

        # --- INT4 右半边 ---
        kv_idx_4bit = kv_idx - seq_len // 2

        bit8_window_mask_4bit = kv_idx_4bit <= (q_idx - bit8_window_size)
        bit4_window_mask = kv_idx_4bit > (q_idx - bit8_window_size - bit4_window_size)
        bit4_no_sink_mask = kv_idx_4bit >= sink_window_size
        int4_mask = bit4_no_sink_mask & bit8_window_mask_4bit & bit4_window_mask

        # --- 组合 ---
        is_bit8_part = kv_idx < seq_len // 2
        is_valid_q_part = q_idx < seq_len // 2
        final_mask = is_valid_q_part & ((is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask))

        return final_mask  # shape = (L, L)，dtype = bool

    def attention_fn(
        module, q, k, v, attn_mask,
        dropout=0.0, scaling=1.0, sliding_window=None, **kwargs
    ):  
        if args.quant:
            seqlen, dim = q.shape[2], q.shape[3]
            km = k.mean(dim=2, keepdim=True)
            k = k - km
            # per token quantization
            q_int8, _, _ = pseudo_quantize_tensor(q, group_size=dim, zero_point=True, bit_width=8)
            k_int8, _, _ = pseudo_quantize_tensor(k, group_size=dim, zero_point=True, bit_width=8)
            v_int8, _, _ = pseudo_quantize_tensor(v, group_size=dim, zero_point=True, bit_width=8)
            k_int4, _, _ = pseudo_quantize_tensor(k, group_size=dim, zero_point=True, bit_width=4)
            v_int4, _, _ = pseudo_quantize_tensor(v, group_size=dim, zero_point=True, bit_width=4)
            q_int8_int8 = torch.cat([q_int8, q_int8], dim=2)
            k_int8_int4 = torch.cat([k_int8, k_int4], dim=2)
            v_int8_int4 = torch.cat([v_int8, v_int4], dim=2)
            
            mask = construct_mix_bit_mask(seq_len=seqlen*2,
                                bit8_window_size=bit8_window_size,
                                bit4_window_size=bit4_window_size,
                                sink_window_size=sink_window_size)
            attn_output, attn_weights = sdpa_attention_forward(
                    module, q_int8_int8, k_int8_int4, v_int8_int4, attention_mask=mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kwargs
                )
            # import pdb; pdb.set_trace()
            return attn_output[:, :seqlen, :, :], attn_weights
        else:
            return sdpa_attention_forward(
                    module, q, k, v, attention_mask=attn_mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kwargs
                )
        
    return attention_fn