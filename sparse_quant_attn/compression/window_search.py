import torch
import torch.nn.functional as F
from loguru import logger
from sparse_quant_attn.compression.attn_replacer import replace_sdpa_for_block

@torch.no_grad()
def search_bit4_window_size_for_head(layers, layer_idx, head_id, inps, ori_outputs, bit8_windows, bit4_window_candidate_sizes, layer_kwargs, args):
    thres_cos = args.bit4_thres_cos
    thres_rmse = args.bit4_thres_rmse
    for w in bit4_window_candidate_sizes:
        bit4_window_sizes = [0] * layers[layer_idx].self_attn.config.num_attention_heads
        bit4_window_sizes[head_id] = w
        replace_sdpa_for_block(
            layers[layer_idx], layer_idx, args,
            bit8_window_sizes=bit8_windows,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=32,
        )
        # quant_outputs = layer(inps, **layer_kwargs)[0]
        quant_outputs = layers_infer(layers, layer_idx, inps, layer_kwargs, args)
        sim, rmse = compute_cos_rmse(ori_outputs, quant_outputs)
        # logger.info(f"Bit4 window size: {w}, similarity: {sim}, rmse: {rmse}")
        if sim >= thres_cos and rmse <= thres_rmse:
            return w
    return bit4_window_candidate_sizes[-1]


@torch.no_grad()
def search_bit8_window_size_for_head(layers, layer_idx, head_id, inps, ori_outputs, bit8_window_candidate_sizes, layer_kwargs, args):
    thres_cos = args.bit8_thres_cos
    thres_rmse = args.bit8_thres_rmse
    for w in bit8_window_candidate_sizes:
        # 替换指定 head 的注意力 kernel（你要确保 replace_sdpa_for_block 支持 per-head）
        # Construct a list where only head_id has window size w, others use max_window_size
        bit8_window_sizes = [bit8_window_candidate_sizes[-1]] * layers[layer_idx].self_attn.config.num_attention_heads
        bit8_window_sizes[head_id] = w
        bit4_window_sizes = [0] * layers[layer_idx].self_attn.config.num_attention_heads
        # replace_sdpa_for_block(
        #     layer, layer_idx, args,
        #     bit8_window_sizes=bit8_window_sizes,
        #     bit4_window_sizes=bit4_window_sizes,
        #     sink_window_size=32,
        # )
        replace_sdpa_for_block(layers[layer_idx], layer_idx, args,bit8_window_sizes=bit8_window_sizes,bit4_window_sizes=bit4_window_sizes,sink_window_size=32)
        # quant_outputs = layer(inps, **layer_kwargs)[0]
        quant_outputs = layers_infer(layers, layer_idx, inps, layer_kwargs, args)
        sim, rmse = compute_cos_rmse(ori_outputs, quant_outputs)
        # logger.info(f"[Layer {layer_idx} | Head {head_id}] Bit8 window size: {w}, similarity: {sim:.5f}, rmse: {rmse:.5f}")
        if sim >= thres_cos and rmse <= thres_rmse:
            return w
    return bit8_window_candidate_sizes[-1]


@torch.no_grad()
def layers_infer(layers, layer_idx, inps, layer_kwargs, args):
    # Infer through remaining layers starting from layer_idx
    outputs = inps
    for i in range(layer_idx, len(layers)):
        layer = layers[i]
        layer = layer.cuda()
        outputs = layer(outputs, **layer_kwargs)[0]
        # layer = layer.cpu()
        torch.cuda.empty_cache()
    return outputs

@torch.no_grad()
def grid_search_block_window_size_per_head_v2(layers, layer_idx, inps, layer_kwargs, max_window_size, args):
    ori_outputs = layers_infer(layers, layer_idx, inps, layer_kwargs, args)
    bit8_window_candidate_sizes = list(range(32, max_window_size + 1, 32))
    if max_window_size not in bit8_window_candidate_sizes:
        bit8_window_candidate_sizes.append(max_window_size)
    # per_head_windows = []
    bit8_windows = []
    bit4_windows = []
    for h in range(layers[layer_idx].self_attn.config.num_attention_heads):  # 当前模型 num_heads
        bit8_best_w = search_bit8_window_size_for_head(
            layers, layer_idx, h,
            inps, ori_outputs,
            bit8_window_candidate_sizes,
            layer_kwargs,
            args
        )
        bit8_windows.append(bit8_best_w)
        logger.info(f"layer {layer_idx} head {h} bit8 window size: {bit8_best_w}")
        bit4_window_candidate_sizes = list(range(0, max_window_size - bit8_best_w + 1, 32))
        if max_window_size - bit8_best_w not in bit4_window_candidate_sizes:
            bit4_window_candidate_sizes.append(max_window_size - bit8_best_w)
        bit8_window_sizes = [bit8_window_candidate_sizes[-1]] * layers[layer_idx].self_attn.config.num_attention_heads
        bit8_window_sizes[h] = bit8_best_w
        bit4_best_w = search_bit4_window_size_for_head(
            layers, layer_idx, h,
            inps, ori_outputs,
            bit8_window_sizes, bit4_window_candidate_sizes,
            layer_kwargs,
            args
        )
        bit4_windows.append(bit4_best_w)
        logger.info(f"layer {layer_idx} head {h} bit4 window size: {bit4_best_w}")
    return bit8_windows, bit4_windows

@torch.no_grad()
def grid_search_block_window_size_per_head(layer, layer_idx, inps, ori_outputs, layer_kwargs, max_window_size, args):
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
            layer_kwargs,
            args
        )
        bit8_windows.append(best_w)
        logger.info(f"layer {layer_idx} head {h} bit8 window size: {best_w}")
    bit4_windows = []
    
    for h in range(layer.self_attn.config.num_attention_heads):
        bit4_window_candidate_sizes = list(range(32, max_window_size - bit8_windows[h] + 1, 32))
        if max_window_size - bit8_windows[h] not in bit4_window_candidate_sizes:
            bit4_window_candidate_sizes.append(max_window_size - bit8_windows[h])
        best_w = search_bit4_window_size_for_head(
            layer, layer_idx, h,
            inps, ori_outputs,
            bit8_windows, bit4_window_candidate_sizes,
            layer_kwargs,
            args
        )
        bit4_windows.append(best_w)
        logger.info(f"layer {layer_idx} head {h} bit4 window size: {best_w}")
    return bit8_windows, bit4_windows


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
        # logger.info(f"Bit8 window size: {w}, similarity: {sim}, rmse: {rmse}")
        if sim >= thres_cos and rmse <= thres_rmse:
            return w
    return bit8_window_candidate_sizes[-1]


@torch.no_grad()
def search_bit4_window_size(layer, layer_idx, inps, ori_outputs, bit8_window_size, bit4_window_candidate_sizes, thres_cos, thres_rmse, layer_kwargs, args):
    for w in bit4_window_candidate_sizes:
        replace_sdpa_for_block(layer, layer_idx, args, bit8_window_size=bit8_window_size, bit4_window_size=w, sink_window_size=32)
        quant_outputs = layer(inps, **layer_kwargs)[0]
        sim, rmse = compute_cos_rmse(ori_outputs, quant_outputs)
        # logger.info(f"Bit4 window size: {w}, similarity: {sim}, rmse: {rmse}")
        if sim >= thres_cos and rmse <= thres_rmse:
            return w
    return bit4_window_candidate_sizes[-1]