from collections import defaultdict
import torch
import torch.nn as nn
from loguru import logger
from sparse_quant_attn.utils.model_utils import get_blocks, move_embed, get_named_linears
from sparse_quant_attn.compression.calibration import get_calib_dataset
from sparse_quant_attn.compression.attn_fake_quant import replace_attn_for_block
import gc
from tqdm import tqdm
import functools

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

    for i in tqdm(range(len(layers)), desc="Running SQAttn..."):
        layer = layers[i]
        layer.cuda()
        named_linears = get_named_linears(layer)
        
         # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        
        
        layer(inps, **layer_kwargs)[0]
        
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        
        logger.info(f"quantizer attn for layer {i}")
        #TODO: quantize attn
        replace_attn_for_block(layer, input_feat, i)

        # update output after compression
        inps = layer(inps, **layer_kwargs)[0]
        
        del input_feat
        layer.cpu()
        torch.cuda.empty_cache()