import torch.nn.functional as F
import torch


def construct_hybrid_mask(seq_len, bit8_window_size, bit4_window_size, sink_window_size, device='cuda'):
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
    q_idx_4bit = q_idx  # 这里保持不变是因为只对 q 在左侧的部分有效

    bit8_window_mask_4bit = kv_idx_4bit <= (q_idx - bit8_window_size)
    bit4_window_mask = kv_idx_4bit > (q_idx - bit8_window_size - bit4_window_size)
    bit4_no_sink_mask = kv_idx_4bit >= sink_window_size
    int4_mask = bit4_no_sink_mask & bit8_window_mask_4bit & bit4_window_mask

    # --- 组合 ---
    is_bit8_part = kv_idx < seq_len // 2
    is_valid_q_part = q_idx < seq_len // 2
    final_mask = is_valid_q_part & ((is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask))

    return final_mask  # shape = (L, L)，dtype = bool



mask = construct_hybrid_mask(seq_len=8,
                             bit8_window_size=0,
                             bit4_window_size=0,
                             sink_window_size=0)
print(mask)
# 模拟 Q, K, V
Q = torch.ones(1, 1, 8, 8, device='cuda')
K = torch.ones(1, 1, 8, 8, device='cuda')
# V = torch.ones(1, 1, 8, 8, device='cuda')
S = 8  # 序列长度
V = torch.eye(S, device='cuda')  # shape = (8, 8)
V = V.unsqueeze(0).unsqueeze(0)

output = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V, attn_mask=mask, is_causal=False, dropout_p=0.0
)
import pdb; pdb.set_trace()
print(output)