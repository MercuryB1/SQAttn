from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sparse_quant_attn.utils.model_utils import get_blocks, move_embed, get_named_linears
from sparse_quant_attn.compression.calibration import get_calib_dataset
from sparse_quant_attn.compression.attn_fake_quant import replace_attn_for_block, replace_flashattn_kernel_for_block, replace_sdpa_for_block
import gc
from tqdm import tqdm
import functools
import numpy as np


@torch.no_grad()
def compress_model(model, tokenizer, device, args):
    layers = get_blocks(model)

    logger.info(f"loading calibdation data: {args.calib_dataset}")
    samples = get_calib_dataset(
        data=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.nsamples,
        seq_len=args.seqlen,
    )

    samples = torch.cat(samples, dim=0)
    samples = samples[0:1]
    logger.info("dataset loading complete")

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
    bits_alloc = defaultdict(list)

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
        ori_outputs = layer(inps, **layer_kwargs)[0]
        bit8_window_size, bit4_window_size = grid_search_block_window_size(layer, i, inps, ori_outputs, layer_kwargs, args)
        bits_alloc[i] = [bit8_window_size, bit4_window_size]
        replace_sdpa_for_block(layer, i, args, bit8_window_size=bit8_window_size, bit4_window_size=bit4_window_size, sink_window_size=32)
        # replace_sdpa_for_block(layer, i, args, bit8_window_size=0, bit4_window_size=0, sink_window_size=0)
        # update output after compression
        inps = layer(inps, **layer_kwargs)[0]
        
        # del input_feat
        layer.cpu()
        torch.cuda.empty_cache()
    return compute_avg_bits(bits_alloc)


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


def compute_avg_bits(bits_alloc):
    avg_bits = sum(bits_alloc.values()) / len(bits_alloc)
    return avg_bits
