import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sparse_quant_attn.utils.model_utils import get_blocks, move_embed
from sparse_quant_attn.compression.calibration import get_calib_dataset
from sparse_quant_attn.compression.attn_replacer import replace_sdpa_for_block
import gc
from tqdm import tqdm
from sparse_quant_attn.compression.window_search import grid_search_block_window_size_per_head, grid_search_block_window_size_per_head_v2

@torch.no_grad()
def compress_model(model, tokenizer, device, args):
    layers = get_blocks(model)

    logger.info(f"loading calibdation data: {args.calib_dataset}")
    samples, padding_mask = get_calib_dataset(
        data=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.nsamples,
        seq_len=args.seqlen,
        device=device
    )
    logger.info("dataset loading complete")
    max_window_size = samples.shape[1]
    inps = []
    layer_kwargs = {}
    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            layer_kwargs['use_cache'] = False
            raise ValueError  # early exit to break later inference
        
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    
    gc.collect()
    torch.cuda.empty_cache()
    bits_alloc = dict()

    for i in tqdm(range(len(layers)), desc="Running SQAttn..."):
        layer = layers[i]
        layer.cuda()
        
        #TODO: quantize attn
        # logger.info(f"{layer.self_attn.config._attn_implementation}")
        # if not (i == 0 or i == len(layers) - 1): 
        # replace_sdpa_for_block(layer, i, args)
        if i !=0 and i != len(layers) - 1:
            # bit8_window_sizes, bit4_window_sizes = grid_search_block_window_size_8bit_only_per_head(layer, i, inps, ori_outputs, layer_kwargs, max_window_size, args)
            bit8_window_sizes, bit4_window_sizes = grid_search_block_window_size_per_head_v2(layers, i, inps, layer_kwargs, max_window_size, args)
            bits_alloc[i] = {
                "bit8": bit8_window_sizes,
                "bit4": bit4_window_sizes,
                "sink": 16  # 如需支持 per-layer sink window，可改为 list
            }
            #TODO per head support
            replace_sdpa_for_block(layer, i, args, bit8_window_sizes=bit8_window_sizes, bit4_window_sizes=bit4_window_sizes, sink_window_size=16)
        
        # update output after compression
        inps = layer(inps, **layer_kwargs)[0]
        
        # del input_feat
        layer.cpu()
        torch.cuda.empty_cache()
    return compute_avg_bits(bits_alloc, max_window_size)
    # return 0



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
    # final_mask = (is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask)

    return final_mask  # shape = (L, L)，dtype = bool



def construct_mix_bit_mask_per_head(seq_len, bit8_window_sizes, bit4_window_sizes, sink_window_size, device='cuda'):
    num_heads = len(bit8_window_sizes)
    q_idx = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(2)  # (1, L, 1)
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(1)  # (1, 1, L)

    q_idx = q_idx.expand(num_heads, seq_len, 1)
    kv_idx = kv_idx.expand(num_heads, 1, seq_len)

    bit8_window_sizes = torch.tensor(bit8_window_sizes, device=device).view(-1, 1, 1)
    bit4_window_sizes = torch.tensor(bit4_window_sizes, device=device).view(-1, 1, 1)

    causal_mask = kv_idx <= q_idx
    bit8_window_mask = kv_idx > (q_idx - bit8_window_sizes)
    sink_mask = kv_idx < sink_window_size
    fp8_mask = causal_mask & (sink_mask | bit8_window_mask)

    kv_idx_4bit = kv_idx - seq_len // 2
    bit8_window_mask_4bit = kv_idx_4bit <= (q_idx - bit8_window_sizes)
    bit4_window_mask = kv_idx_4bit > (q_idx - bit8_window_sizes - bit4_window_sizes)
    bit4_no_sink_mask = kv_idx_4bit >= sink_window_size
    int4_mask = bit4_no_sink_mask & bit8_window_mask_4bit & bit4_window_mask

    is_bit8_part = kv_idx < seq_len // 2
    is_valid_q_part = q_idx < seq_len // 2
    final_mask = is_valid_q_part & ((is_bit8_part & fp8_mask) | (~is_bit8_part & int4_mask))

    return final_mask  # shape: (num_heads, L, L)



def compute_avg_bits(bits_alloc: dict, max_window_size: int, sink_window_size: int = 32):
    """
    计算所有层所有 head 的平均比特数（使用 per-head 掩码）

    参数：
        bits_alloc: dict[int, dict]，如：
            {
                1: {"bit8": [256, 128, ...], "bit4": [128, 64, ...], "sink": 16},
                ...
            }
        max_window_size: 半序列长度（即原始序列长度为 2 * max_window_size）
        sink_window_size: 若未在 bits_alloc 中单独指定，使用该默认值

    返回：
        avg_bits_overall: 所有层所有 head 的平均比特数（float）
        avg_bits_per_layer: 每层的平均比特数组（List[float]）
    """
    all_head_bits = []
    avg_bits_per_layer = []

    for layer_idx, layer_cfg in bits_alloc.items():
        bit8_windows = layer_cfg["bit8"]
        bit4_windows = layer_cfg["bit4"]
        sink_window = layer_cfg.get("sink", sink_window_size)

        num_heads = len(bit8_windows)
        # assert len(bit4_windows) == num_heads, f"bit4 config mismatch at layer {layer_idx}"

        # 构造 mask：shape = (H, L, 2L)
        mask = construct_mix_bit_mask_per_head(
            seq_len=max_window_size * 2,
            bit8_window_sizes=bit8_windows,
            bit4_window_sizes=bit4_windows,
            sink_window_size=sink_window,
            device='cuda'
        )  # shape = (H, L, 2L)

        # 截取有效部分 (只看 q 的前半部分)
        #TODO need to support for INT4
        bit8_mask = mask[:, :max_window_size, :max_window_size]  # (H, L, L)
        bit4_mask = mask[:, :max_window_size, max_window_size:]  # (H, L, L)
        numel = bit8_mask.shape[1] * bit8_mask.shape[2] // 2
        # import pdb; pdb.set_trace()
        layer_head_bits = []
        for h in range(num_heads):
            bit8_attend_cnt = bit8_mask[h].sum().item()
            bit4_attend_cnt = bit4_mask[h].sum().item()
            # 假设所有 attend token 都来自 int8 区域，等价于原版实现
            avg_bits = (bit8_attend_cnt * 8 + bit4_attend_cnt * 4) / numel
            layer_head_bits.append(avg_bits)
            all_head_bits.append(avg_bits)

        layer_avg = sum(layer_head_bits) / num_heads
        avg_bits_per_layer.append(layer_avg)

    overall_avg = sum(all_head_bits) / len(all_head_bits)
    return overall_avg, avg_bits_per_layer
