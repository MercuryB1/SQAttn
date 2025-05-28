import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, 1, 1, 8192, 8192, device="cuda")
query = torch.randn(14, 28, 2048, 128, device="cuda", dtype=torch.float16)
key = torch.randn(14, 28, 2048, 128, device="cuda", dtype=torch.float16)
value = torch.randn(14, 28, 2048, 128, device="cuda", dtype=torch.float16)
output = flex_attention(query, key, value, block_mask=block_mask, enable_gqa=True)