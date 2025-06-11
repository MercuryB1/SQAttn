from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sparse_quant_attn.utils.model_utils import get_blocks, move_embed, get_named_linears
from sparse_quant_attn.compression.calibration import get_calib_dataset
from sparse_quant_attn.compression.attn_replacer import replace_sdpa_for_block
import gc
from tqdm import tqdm
import functools
import numpy as np


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
        # named_linears = get_named_linears(layer)
        
        #  # firstly, get input features of all linear layers
        # def cache_input_hook(m, x, y, name, feat_dict):
        #     x = x[0]
        #     x = x.detach().cpu()
        #     feat_dict[name].append(x)

        # input_feat = defaultdict(list)
        
        # handles = []
        # for name in named_linears:
        #     handles.append(
        #         named_linears[name].register_forward_hook(
        #             functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
        #         )
        #     )
        # inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # # get output as next layer's input
        
        
        # layer(inps, **layer_kwargs)[0]
        
        # for h in handles:
        #     h.remove()
        # input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        #TODO: quantize attn
        # logger.info(f"{layer.self_attn.config._attn_implementation}")
        # if not (i == 0 or i == len(layers) - 1): 
        # replace_sdpa_for_block(layer, i, args)
        if i !=0 and i != len(layers) - 1:
            ori_outputs = layer(inps, **layer_kwargs)[0]
            # bit8_window_size, bit4_window_size = grid_search_block_window_size_8bit_only(layer, i, inps, ori_outputs, layer_kwargs, max_window_size, args)
            bit8_window_sizes, bit4_window_sizes = grid_search_block_window_size_8bit_only_per_head(layer, i, inps, ori_outputs, layer_kwargs, max_window_size, args)
            bits_alloc[i] = {
                "bit8": bit8_window_sizes,
                "bit4": bit4_window_sizes,
                "sink": 16  # 如需支持 per-layer sink window，可改为 list
            }
            # import pdb; pdb.set_trace()
            #TODO per head support
            replace_sdpa_for_block(layer, i, args, bit8_window_sizes=bit8_window_sizes, bit4_window_sizes=bit4_window_sizes, sink_window_size=16)
        # replace_sdpa_for_block(layer, i, args, bit8_window_size=0, bit4_window_size=0, sink_window_size=0)
        # update output after compression
        inps = layer(inps, **layer_kwargs)[0]
        
        # del input_feat
        layer.cpu()
        torch.cuda.empty_cache()
    return compute_avg_bits(bits_alloc, max_window_size)
    # return 0


@torch.no_grad()
def search_bit8_window_size_for_head(layer, layer_idx, head_id, inps, ori_outputs, bit8_window_candidate_sizes, thres_cos, thres_rmse, layer_kwargs, args):
    for w in bit8_window_candidate_sizes:
        # 替换指定 head 的注意力 kernel（你要确保 replace_sdpa_for_block 支持 per-head）
        # Construct a list where only head_id has window size w, others use max_window_size
        bit8_window_sizes = [bit8_window_candidate_sizes[-1]] * layer.self_attn.config.num_attention_heads
        bit8_window_sizes[head_id] = w
        bit4_window_sizes = [0] * layer.self_attn.config.num_attention_heads
        replace_sdpa_for_block(
            layer, layer_idx, args,
            bit8_window_sizes=bit8_window_sizes,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=32,
        )
        quant_outputs = layer(inps, **layer_kwargs)[0]
        sim, rmse = compute_cos_rmse(ori_outputs, quant_outputs)
        # logger.info(f"[Layer {layer_idx} | Head {head_id}] Bit8 window size: {w}, similarity: {sim:.5f}, rmse: {rmse:.5f}")
        if sim >= thres_cos and rmse <= thres_rmse:
            return w
    return bit8_window_candidate_sizes[-1]


@torch.no_grad()
def grid_search_block_window_size_8bit_only_per_head(layer, layer_idx, inps, ori_outputs, layer_kwargs, max_window_size, args):
    # logger.info(f"Starting per-head grid search for layer {layer_idx}")
    bit8_window_candidate_sizes = list(range(32, max_window_size + 1, 32))
    if max_window_size not in bit8_window_candidate_sizes:
        bit8_window_candidate_sizes.append(max_window_size)
    
    # per_head_windows = []
    bit8_windows = []
    for h in range(layer.self_attn.config.num_attention_heads):  # 当前模型 num_heads
        best_w = search_bit8_window_size_for_head(
            layer, layer_idx, h,
            inps, ori_outputs,
            bit8_window_candidate_sizes,
            args.bit8_thres_cos,
            args.bit8_thres_rmse,
            layer_kwargs,
            args
        )
        bit8_windows.append(best_w)
        logger.info(f"layer {layer_idx} head {h} bit8 window size: {best_w}")
        # per_head_windows.append((best_w, 0))  # 目前只支持 8bit，4bit=0
    return bit8_windows, None  # List[(bit8, bit4)] × num_heads


@torch.no_grad()
def grid_search_block_window_size_8bit_only(layer, layer_idx, inps, ori_outputs, layer_kwargs, max_window_size, args):
    logger.info(f"Starting grid search for optimal window sizes for layer {layer_idx}")
    # bit8_window_candidate_sizes = list(range(128, args.seqlen + 1, 128))
    # Evenly distribute candidate sizes up to max_window_size
    bit8_window_candidate_sizes = list(range(32, max_window_size + 1, 32))
    if max_window_size not in bit8_window_candidate_sizes:
        bit8_window_candidate_sizes.append(max_window_size)
    bit8_thres_cos = args.bit8_thres_cos
    bit8_thres_rmse = args.bit8_thres_rmse
    bit8_window_size = search_bit8_window_size(layer, layer_idx, inps, ori_outputs, bit8_window_candidate_sizes, bit8_thres_cos, bit8_thres_rmse, layer_kwargs, args)
    return bit8_window_size, 0

@torch.no_grad()
def grid_search_block_window_size(layer, layer_idx, inps, ori_outputs, layer_kwargs, args):

    logger.info(f"Starting grid search for optimal window sizes for layer {layer_idx}")
    bit8_window_candidate_sizes = list(range(16, args.seqlen + 1, 16))
    bit8_thres_cos = 0.9999
    bit8_thres_rmse = 0.05
    bit4_thres_cos = 0.9999
    bit4_thres_rmse = 0.01

    bit8_window_size = search_bit8_window_size(layer, layer_idx, inps, ori_outputs, bit8_window_candidate_sizes, bit8_thres_cos, bit8_thres_rmse, layer_kwargs, args)
    bit4_window_candidate_sizes = bit8_window_candidate_sizes[:(args.seqlen-bit8_window_size)//16]
    bit4_window_size = search_bit4_window_size(layer, layer_idx, inps, ori_outputs, bit8_window_size, bit4_window_candidate_sizes, bit4_thres_cos, bit4_thres_rmse, layer_kwargs, args)
    logger.info(f"Best bit8 window size: {bit8_window_size}, best bit4 window size: {bit4_window_size}")

    return bit8_window_size, bit4_window_size

    
def compute_cos_rmse(a: torch.Tensor, b: torch.Tensor):
    a = a.view(-1).float()
    b = b.view(-1).float()
    cos_sim = F.cosine_similarity(a, b, dim=0).item()
    rmse = torch.sqrt(F.mse_loss(a, b)).item()
    return cos_sim, rmse

@torch.no_grad()
def search_bit8_window_size(layer, layer_idx, inps, ori_outputs, bit8_window_candidate_sizes, thres_cos, thres_rmse, layer_kwargs, args):
    for w in bit8_window_candidate_sizes:
        replace_sdpa_for_block(layer, layer_idx, args, bit8_window_size=w, bit4_window_size=0, sink_window_size=32)
        # replace
        quant_outputs = layer(inps, **layer_kwargs)[0]
        sim, rmse = compute_cos_rmse(ori_outputs, quant_outputs)
        logger.info(f"Bit8 window size: {w}, similarity: {sim}, rmse: {rmse}")
        if sim >= thres_cos and rmse <= thres_rmse:
            return w
    return bit8_window_candidate_sizes[-1]


@torch.no_grad()
def search_bit4_window_size(layer, layer_idx, inps, ori_outputs, bit8_window_size, bit4_window_candidate_sizes, thres_cos, thres_rmse, layer_kwargs, args):
    for w in bit4_window_candidate_sizes:
        replace_sdpa_for_block(layer, layer_idx, args, bit8_window_size=bit8_window_size, bit4_window_size=w, sink_window_size=32)
        quant_outputs = layer(inps, **layer_kwargs)[0]
        sim, rmse = compute_cos_rmse(ori_outputs, quant_outputs)
        logger.info(f"Bit4 window size: {w}, similarity: {sim}, rmse: {rmse}")
        if sim >= thres_cos and rmse <= thres_rmse:
            return w
    return bit4_window_candidate_sizes[-1]


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


# #TODO 增加int4的计算
# def compute_avg_bits(bits_alloc, max_window_size):
#     total_bits = 0
#     total_tokens = 0
#     # import pdb; pdb.set_trace()
#     for window_sizes in bits_alloc.values():
#         mask = construct_mix_bit_mask(max_window_size*2, window_sizes[0], window_sizes[1], 32)
#         mask = mask[:max_window_size, :max_window_size]
#         total_bits += torch.sum(mask.float()) * 8 # 8 bits per token
#         total_tokens += mask.numel()
#     avg_bits = total_bits / (total_tokens / 2)
#     return avg_bits


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
            bit4_window_sizes=0,
            sink_window_size=sink_window,
            device='cuda'
        )  # shape = (H, L, 2L)

        # 截取有效部分 (只看 q 的前半部分)
        #TODO need to support for INT4
        mask = mask[:, :max_window_size, :max_window_size]  # (H, L, L)
        numel = mask.shape[1] * mask.shape[2] // 2

        layer_head_bits = []
        for h in range(num_heads):
            attend_cnt = mask[h].sum().item()
            # 假设所有 attend token 都来自 int8 区域，等价于原版实现
            avg_bits = (attend_cnt * 8) / numel
            layer_head_bits.append(avg_bits)
            all_head_bits.append(avg_bits)

        layer_avg = sum(layer_head_bits) / num_heads
        avg_bits_per_layer.append(layer_avg)

    overall_avg = sum(all_head_bits) / len(all_head_bits)
    return overall_avg, avg_bits_per_layer
