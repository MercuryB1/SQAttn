from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch
from typing import Tuple, Optional
from loguru import logger


def int4_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize input tensor to int4 format using symmetric quantization.
    
    Args:
        x: Input tensor to be quantized
        
    Returns:
        Quantized tensor in the same shape as input
    """
    max_abs_val = torch.max(torch.abs(x))
    
    # Return zeros if input is close to zero to avoid numerical issues
    if max_abs_val < 1e-9:
        return torch.zeros_like(x)
    
    # Scale factor for symmetric int4 quantization (range: [-7, 7])
    scale = max_abs_val / 7.0
    
    # Quantize and clamp to valid range
    quantized_x = torch.round(x / scale)
    clamped_x = torch.clamp(quantized_x, -7.0, 7.0)
    
    # Dequantize back to original scale
    return clamped_x * scale

def fp8_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize input tensor to FP8 format using e4m3 format.
    
    Args:
        x: Input tensor to be quantized
        
    Returns:
        Quantized tensor in bfloat16 format
    """
    x_fp8 = x.to(torch.float8_e4m3fn)
    return x_fp8.to(torch.bfloat16)

def sink_sliding_window_multi_mask(
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    bit8_window_size: int,
    bit4_window_size: int,
    sink_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create masks for FP8 and INT4 quantization based on sliding window and sink tokens.
    
    Args:
        q_idx: Query position indices
        kv_idx: Key-Value position indices
        bit8_window_size: Size of the FP8 window
        bit4_window_size: Size of the INT4 window
        sink_size: Number of sink tokens
        
    Returns:
        Tuple of (fp8_mask, int4_mask)
    """
    causal_mask = q_idx >= kv_idx
    sink_mask = kv_idx <= sink_size
    window8_mask = (q_idx - kv_idx < bit8_window_size)
    window4_mask = (q_idx - kv_idx < bit4_window_size)
    
    fp8_mask = causal_mask & (sink_mask | window4_mask)
    int4_mask = causal_mask & (~window4_mask) & window8_mask
    
    return fp8_mask, int4_mask

def create_multi_mask(
    q_len: int,
    kv_len: int,
    bit8_window_size: int,
    bit4_window_size: int,
    sink_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create position-based masks for multi-precision attention.
    
    Args:
        q_len: Length of query sequence
        kv_len: Length of key-value sequence
        bit8_window_size: Size of the FP8 window
        bit4_window_size: Size of the INT4 window
        sink_size: Number of sink tokens
        
    Returns:
        Tuple of (fp8_mask, int4_mask)
    """
    q_idx = torch.arange(q_len).unsqueeze(1).expand(q_len, kv_len)
    kv_idx = torch.arange(kv_len).unsqueeze(0).expand(q_len, kv_len)
    return sink_sliding_window_multi_mask(q_idx, kv_idx, bit8_window_size, bit4_window_size, sink_size)


@torch.no_grad()
def sparsequantattn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bit8_window_size: int,
    bit4_window_size: int,
    sink_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform sparse quantized attention during prefill phase.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        bit8_window_size: Size of the FP8 window
        bit4_window_size: Size of the INT4 window
        sink_size: Number of sink tokens
        
    Returns:
        Tuple of (output_tensor, block_mask)
        
    TODO:
        1. Implement FP8 scaling
        2. Implement smooth QKV
        3. Add head-wise quantization support
    """

    logger.info(f"q.size: {q.size()}, k.size: {k.size()}, v.size: {v.size()}")
    # import pdb; pdb.set_trace()
    kv_len = k.size(2)
    
    def sink_sliding_window_causal(b: int, h: int, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        # FP8 mask logic
        causal_mask = kv_idx <= q_idx
        bit8_window_mask = kv_idx >= q_idx - bit8_window_size
        sink_mask = kv_idx <= sink_size
        fp8_mask = causal_mask & (sink_mask | bit8_window_mask)
        
        # INT4 mask logic
        bit8_window_mask = kv_idx - kv_len <= q_idx - bit8_window_size
        bit4_window_mask = kv_idx - kv_len >= q_idx - bit4_window_size
        int4_no_sink_mask = kv_idx - kv_len > sink_size
        int4_mask = int4_no_sink_mask & bit8_window_mask & bit4_window_mask
        
        return fp8_mask | int4_mask
    
    # Quantize tensors
    dtype = q.dtype
    device = q.device
    q_fp8 = fp8_quant(q).to(dtype=dtype, device=device)
    k_fp8 = fp8_quant(k).to(dtype=dtype, device=device)
    v_fp8 = fp8_quant(v).to(dtype=dtype, device=device)
    k_int4 = int4_quant(k).to(dtype=dtype, device=device)
    v_int4 = int4_quant(v).to(dtype=dtype, device=device)
    
    # Concatenate FP8 and INT4 tensors
    k_fp8_int4 = torch.cat([k_fp8, k_int4], dim=2)
    v_fp8_int4 = torch.cat([v_fp8, v_int4], dim=2)
    
    # Create block mask and compute attention
    block_mask = create_block_mask(
        sink_sliding_window_causal,
        B=None,
        H=q.size(1),
        Q_LEN=q.size(2),
        KV_LEN=k.size(2) * 2,
        BLOCK_SIZE=16
    )
    with torch.no_grad():
        output = flex_attention(q_fp8, k_fp8_int4, v_fp8_int4, block_mask=block_mask, enable_gqa=True)
    return output, block_mask

# TODO: implement decode

@torch.no_grad()
def calculate_attn_params(input_features: dict) -> Tuple[int, int, int]:
    # TODO: calculate bit8_window_size, bit4_window_size, sink_size
    return 64, 128, 16