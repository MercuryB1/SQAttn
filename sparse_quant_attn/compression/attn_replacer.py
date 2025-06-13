import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import copy
from .quant_utils import FloatQuantizer, IntegerQuantizer
import math
import os
import matplotlib.pyplot as plt


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
def replace_sdpa_for_block(module: nn.Module, blockidx: int, args, bit8_window_sizes=None, bit4_window_sizes=None, sink_window_size=0):
    if isinstance(module, Qwen2DecoderLayer):
        from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS
        attn_fn = delayed_sdpa_wrapper(blockidx, args=args, bit8_window_sizes=bit8_window_sizes, bit4_window_sizes=bit4_window_sizes, sink_window_size=sink_window_size)
        impl_name = f"sparsequantattn_{blockidx}"
        ALL_ATTENTION_FUNCTIONS[impl_name] = attn_fn
        module.self_attn.config = copy.deepcopy(module.self_attn.config)
        # module.self_attn.config._attn_implementation = impl_name
        module.self_attn.config._attn_implementation = impl_name


def delayed_sdpa_wrapper(layer_idx, bit8_window_sizes=0, bit4_window_sizes=0, sink_window_size=0, args=None):
    # Determine if we should use per-head mask based on input types
    is_per_head = isinstance(bit8_window_sizes, list) or isinstance(bit4_window_sizes, list)
    
    if is_per_head:
        # Get number of heads from the other list
        if isinstance(bit8_window_sizes, list):
            num_heads = len(bit8_window_sizes)
        else:
            num_heads = len(bit4_window_sizes)
        
        # Convert single values to lists for per-head case
        if isinstance(bit8_window_sizes, int):
            bit8_window_sizes = [bit8_window_sizes] * num_heads
        if isinstance(bit4_window_sizes, int):
            bit4_window_sizes = [bit4_window_sizes] * num_heads
        if bit8_window_sizes is None:
            bit8_window_sizes = [0] * num_heads
        if bit4_window_sizes is None:
            bit4_window_sizes = [0] * num_heads
    else:
        # For single value case, ensure we have single integers
        if isinstance(bit8_window_sizes, list):
            bit8_window_sizes = bit8_window_sizes[0] if bit8_window_sizes else 0
        if isinstance(bit4_window_sizes, list):
            bit4_window_sizes = bit4_window_sizes[0] if bit4_window_sizes else 0
        bit8_window_sizes = bit8_window_sizes or 0
        bit4_window_sizes = bit4_window_sizes or 0

    def construct_mix_bit_mask_per_head(seq_len, bit8_window_sizes, bit4_window_sizes, sink_window_size, device='cuda'):
        """
        返回 shape = (num_heads, L, L) 的 bool mask，True表示可以attend。
        """
        num_heads = len(bit8_window_sizes)
        q_idx = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(2)  # (1, L, 1)
        kv_idx = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(1)  # (1, 1, L)

        q_idx = q_idx.expand(num_heads, seq_len, 1)
        kv_idx = kv_idx.expand(num_heads, 1, seq_len)

        bit8_window_sizes = torch.tensor(bit8_window_sizes, device=device).view(-1, 1, 1)
        bit4_window_sizes = torch.tensor(bit4_window_sizes, device=device).view(-1, 1, 1)

        # --- FP8 mask ---
        causal_mask = kv_idx <= q_idx
        bit8_window_mask = kv_idx > (q_idx - bit8_window_sizes)
        sink_mask = kv_idx < sink_window_size
        fp8_mask = causal_mask & (sink_mask | bit8_window_mask)

        # --- INT4 mask ---
        kv_idx_4bit = kv_idx - seq_len // 2
        bit8_window_mask_4bit = kv_idx_4bit <= (q_idx - bit8_window_sizes)
        bit4_window_mask = kv_idx_4bit > (q_idx - bit8_window_sizes - bit4_window_sizes)
        bit4_no_sink_mask = kv_idx_4bit >= sink_window_size
        int4_mask = bit4_no_sink_mask & bit8_window_mask_4bit & bit4_window_mask

        is_bit8_part = kv_idx < seq_len // 2
        final_mask = (is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask)

        return final_mask[:, :seq_len // 2, :]  # 每个 head 一张 mask（只保留后半段）

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
        # is_valid_q_part = q_idx < seq_len // 2
        # final_mask = is_valid_q_part & ((is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask))
        final_mask = (is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask)

        # return final_mask  # shape = (L, L)，dtype = bool
        return final_mask[:seq_len // 2, :]

    def attention_fn(
        module, q, k, v, attn_mask,
        dropout=0.0, scaling=1.0, sliding_window=None, **kwargs
    ):  
        if args.quant:
            q_len, dim = q.shape[2], q.shape[3]
            kv_len = k.shape[2]
            km = k.mean(dim=2, keepdim=True)
            k = k - km
            
            # per token quantization
            if args.qk_qtype == "int":
                bit8_qk_quantizer = IntegerQuantizer(8, False, "per_token")
                bit4_qk_quantizer = IntegerQuantizer(4, False, "per_token")
            elif args.qk_qtype == "e4m3":
                bit8_qk_quantizer = FloatQuantizer("e4m3", True, "per_token")
                bit4_qk_quantizer = IntegerQuantizer(4, False, "per_token")
            elif args.qk_qtype == "e5m2":
                bit8_qk_quantizer = FloatQuantizer("e5m2", True, "per_token")
                bit4_qk_quantizer = IntegerQuantizer(4, False, "per_token")
            else:
                raise ValueError(f"Invalid quantization type: {args.qk_qtype}")
            
            if args.v_qtype == "int":
                bit8_v_quantizer = IntegerQuantizer(8, False, "per_channel")
                bit4_v_quantizer = IntegerQuantizer(4, False, "per_channel")
            elif args.v_qtype == "e4m3":
                bit8_v_quantizer = FloatQuantizer("e4m3", True, "per_channel")
                bit4_v_quantizer = IntegerQuantizer(4, False, "per_channel")
            elif args.v_qtype == "e5m2":
                bit8_v_quantizer = FloatQuantizer("e5m2", True, "per_channel")
                bit4_v_quantizer = IntegerQuantizer(4, False, "per_channel")
            else:
                raise ValueError(f"Invalid quantization type: {args.v_qtype}")

            q_bit8 = bit8_qk_quantizer.fake_quant_tensor(q)
            k_bit8 = bit8_qk_quantizer.fake_quant_tensor(k)
            k_bit4 = bit4_qk_quantizer.fake_quant_tensor(k)
            v_bit8 = bit8_v_quantizer.fake_quant_tensor(v)
            v_bit4 = bit4_v_quantizer.fake_quant_tensor(v)

            # q_bit8_bit8 = torch.cat([q_bit8, q_bit8], dim=2)
            q_bit8_bit8 = q_bit8
            k_bit8_bit4 = torch.cat([k_bit8, k_bit4], dim=2)
            v_bit8_bit4 = torch.cat([v_bit8, v_bit4], dim=2)
            
            # Choose mask construction based on input type
            if is_per_head:
                mask = construct_mix_bit_mask_per_head(
                    seq_len=kv_len*2,
                    bit8_window_sizes=bit8_window_sizes,
                    bit4_window_sizes=bit4_window_sizes,
                    sink_window_size=sink_window_size
                )
            else:
                mask = construct_mix_bit_mask(
                    seq_len=kv_len*2,
                    bit8_window_size=bit8_window_sizes,
                    bit4_window_size=bit4_window_sizes,
                    sink_window_size=sink_window_size
                )

            # mask = mask[-q_len*2:, :]
            # TODO 支持per head
            if is_per_head:
                mask = mask[:, -q_len:, :]
            else:
                mask = mask[-q_len:, :]
            
            # logger.info(f"mask: {mask.float().sum(dim=1)}")
            # import pdb; pdb.set_trace()
            # logger.info(f"mask shape: {mask.shape}")
            # logger.info(f"q_bit8_bit8 shape: {q_bit8_bit8.shape}")
            # logger.info(f"k_bit8_bit4 shape: {k_bit8_bit4.shape}")
            # logger.info(f"v_bit8_bit4 shape: {v_bit8_bit4.shape}")
            attn_output, attn_weights = sdpa_attention_forward(
                    module, q_bit8_bit8, k_bit8_bit4, v_bit8_bit4, attention_mask=mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kwargs
                )
            # import pdb; pdb.set_trace()
            return attn_output[:, :q_len, :, :], attn_weights
        
        else:
            def cal_attn_weight(query, key, is_causal=True, attn_mask=None):
                key = repeat_kv(key, 6)
                query = query.contiguous()
                key = key.contiguous()
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
                if is_causal:
                    assert attn_mask is None
                    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query.dtype)

                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias = attn_mask + attn_bias
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                import torch.nn.functional as F
                # attn_weight_pool=F.avg_pool2d(attn_weight, kernel_size=(10,10),stride=(10,10))
                attn_weight_pool=F.max_pool2d(attn_weight, (10,10), stride=(10,10))
                return attn_weight_pool
            
            attn_weights = cal_attn_weight(q, k)
            os.makedirs(f'attn_vis_softmax_max_pool/layer_{layer_idx}', exist_ok=True)
            attn_map = attn_weights.detach().to(torch.float32).cpu().mean(dim=0) # [H, Q, K]
            # import pdb; pdb.set_trace()
            for h in range(attn_map.shape[0]):
                plt.imshow(attn_map[h], cmap='coolwarm', aspect='auto')
                plt.colorbar()
                plt.title(f'Layer {layer_idx} Head {h}')
                plt.savefig(f'attn_vis_softmax_max_pool/layer_{layer_idx}/head_{h}.png')
                plt.close()
            return sdpa_attention_forward(
                    module, q, k, v, attention_mask=attn_mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kwargs
                )
        
    return attention_fn



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)