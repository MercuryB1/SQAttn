import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from sparse_quant_attn.compression.sq_attn import sparsequantattn_prefill, calculate_attn_params
from loguru import logger
from sparse_quant_attn.compression.kernel import flashattn
from sparse_quant_attn.compression.kernel_sqattn import sqattn
from sparse_quant_attn.compression.fake_quant import quantize_activation_per_token_absmax
import tilelang
import tilelang.language as T


@torch.no_grad()
def replace_flashattn_kernel_for_block(module: nn.Module, input_features: dict, blockidx: int, args):
    if isinstance(module, Qwen2DecoderLayer):
        from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS
        
        impl_name = f"sparsequantattn_{blockidx}"
        ALL_ATTENTION_FUNCTIONS[impl_name] = delayed_kernel_wrapper(blockidx, args)
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



def delayed_kernel_wrapper(layer_idx, args):
    kernel_holder = {"compiled": None}

    def attention_fn(
        self_module, q, k, v, attn_mask,
        dropout=0.0, scaling=1.0, sliding_window=None, **kwargs
    ):  

        q = q.transpose(1, 2)  # (B, S, H, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # TODO quantize q, k, v
        q_int8 = quantize_activation_per_token_absmax(q, a_bits=8)
        km = k.mean(dim=1, keepdim=True)
        k = k - km
        k_int8 = quantize_activation_per_token_absmax(k, a_bits=8)
        v_int8 = quantize_activation_per_token_absmax(v, a_bits=8)
        k_int4 = quantize_activation_per_token_absmax(k, a_bits=4)
        v_int4 = quantize_activation_per_token_absmax(v, a_bits=4)
        k_int8_int4 = torch.cat([k_int8, k_int4], dim=1)
        v_int8_int4 = torch.cat([v_int8, v_int4], dim=1)
        

        if kernel_holder["compiled"] is None:
            # 动态提取实际 shape
            B, S, H, D = q.shape
            print(f"[Compile Kernel] Layer {layer_idx}: B={B}, H={H}, S={S}, D={D}")
            
            # dynamic shape
            # if args.dynamic_shape:
            #     program = flashattn(
            #         T.symbolic("b"), H, T.symbolic("s"), D,
            #         is_causal=True,
            #         tune=False,
            #         groups=7
            #     )(block_M=128, block_N=128, num_stages=2, threads=128)
            # else:
            #     program = flashattn(
            #         B, H, S, D,
            #         is_causal=True,
            #         tune=False,
            #         groups=7
            #     )(block_M=128, block_N=128, num_stages=2, threads=128)
            bit8_window_size = 2000
            bit4_window_size = 0
            sink_size = 128

            if args.dynamic_shape:
                program = sqattn(
                    T.symbolic("b"), H, T.symbolic("s"), D,
                    is_causal=True,
                    bit8_window_size=bit8_window_size,
                    bit4_window_size=bit4_window_size,
                    sink_size=sink_size,
                    tune=False,
                    groups=7
                )(block_M=128, block_N=128, num_stages=2, threads=128)
            else:
                program = sqattn(
                    B, H, S, D,
                    is_causal=True,
                    bit8_window_size=bit8_window_size,
                    bit4_window_size=bit4_window_size,
                    sink_size=sink_size,
                    tune=False,
                    groups=7
                )(block_M=128, block_N=128, num_stages=2, threads=128)
            kernel_holder["compiled"] = tilelang.compile(program, out_idx=[3])
        
        # 调用已编译 kernel
        return kernel_holder["compiled"](q_int8, k_int8_int4, v_int8_int4), None

    return attention_fn
