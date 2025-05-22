import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from sparse_quant_attn.compression.sq_attn import sparsequantattn_prefill, calculate_attn_params




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
            print(f"[Layer {self_module.layer_idx}] Using bit8={bit8_window_size}, bit4={bit4_window_size}, sink={sink_size}")
            
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
