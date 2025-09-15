import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import gc
from typing import Dict
from dataclasses import dataclass
from loguru import logger
from sparse_quant_attn.utils.model_utils import get_blocks, move_embed
from sparse_quant_attn.utils.model_utils import batch_layer_infer
from sparse_quant_attn.compression.calibration import get_calib_dataset
from sparse_quant_attn.compression.attn_replacer import replace_mp_triton_for_block
from sparse_quant_attn.compression.window_search import calibrate_layer_windows

# ============= Configuration =============

@dataclass
class CalibrationConfig:
    """Calibration configuration"""
    use_calibration: bool = True
    energy_threshold: float = 0.95
    save_checkpoints: bool = True
    checkpoint_interval: int = 4
    default_bit8_window: int = 128
    default_bit4_window: int = 256
    default_sink_window: int = 16


# ============= Phase 1: Data Preparation =============

def get_layer_inputs(model, layers, sample, device):
    """
    Get inputs to first layer using Catcher mechanism
    """
    inps = []
    layer_kwargs = {}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            layer_kwargs['use_cache'] = False
            raise ValueError
    
    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(sample.to(device))
        else:
            model(sample.to(device))
    except ValueError:
        pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    gc.collect()
    torch.cuda.empty_cache()
    
    return inps[0], layer_kwargs


# ============= Phase 3: Results Management =============

def save_checkpoint(layer_idx: int, windows_dict: Dict, bits_alloc: Dict):
    """Save checkpoint for recovery"""
    checkpoint = {
        'layer_idx': layer_idx,
        'windows': {f"{k[0]}_{k[1]}": v for k, v in windows_dict.items()},
        'bits_alloc': bits_alloc
    }
    
    path = f"checkpoint_layer_{layer_idx}.json"
    with open(path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    logger.info(f"Checkpoint saved: {path}")


def save_final_results(windows_dict: Dict, bits_alloc: Dict, config: CalibrationConfig, output_path: str):
    """Save final calibration results with statistics"""
    # Compute statistics
    all_bit8 = []
    all_bit4 = []
    
    for layer_config in bits_alloc.values():
        if isinstance(layer_config, dict):
            all_bit8.append(layer_config.get('bit8', 0))
            all_bit4.append(layer_config.get('bit4', 0))
    
    results = {
        'configuration': {
            'energy_threshold': config.energy_threshold,
            'use_calibration': config.use_calibration
        },
        'bits_allocation': bits_alloc,
        'statistics': {
            'bit8': {
                'mean': np.mean(all_bit8) if all_bit8 else 0,
                'std': np.std(all_bit8) if all_bit8 else 0,
                'min': np.min(all_bit8) if all_bit8 else 0,
                'max': np.max(all_bit8) if all_bit8 else 0,
            },
            'bit4': {
                'mean': np.mean(all_bit4) if all_bit4 else 0,
                'std': np.std(all_bit4) if all_bit4 else 0,
                'min': np.min(all_bit4) if all_bit4 else 0,
                'max': np.max(all_bit4) if all_bit4 else 0,
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved: {output_path}")


# ============= Main Compression Function =============

def compress_model(model, tokenizer, device, args):
    """
    Compress model with integrated layer-by-layer calibration
    """
    # Initialize configuration
    
    layers = get_blocks(model)
    logger.info(f"Starting compression with {len(layers)} layers")
    
    # ========== Phase 1: Data Preparation ==========
    logger.info("Phase 1: Preparing calibration data...")
    
    # Load raw samples
    raw_samples, _ = get_calib_dataset(
        data=args.calib_dataset,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.nsamples,
        seq_len=args.seqlen,
        device=device,
        args=args
    )

    # Generate complete sequences if using calibration
    # import pdb; pdb.set_trace()
    if not isinstance(raw_samples, list):
        raw_samples = [raw_samples]

    max_len = max([sample.size(1) for sample in raw_samples])
    
    # Get layer inputs
    samples_inps, samples_layer_kwargs = [], []
    for sample in raw_samples:
        inps, layer_kwargs = get_layer_inputs(model, layers, sample, device)
        samples_inps.append(inps)
        samples_layer_kwargs.append(layer_kwargs)

    del raw_samples
    gc.collect()
    
    # ========== Phase 2: Layer Processing ==========
    logger.info("Phase 2: Processing layers...")
    
    bits_alloc = {}
    # Get original outputs if needed
    # if args.mse_output == "full":
    #     ori_model_outputs = model_infer(model, inps, layer_kwargs, args)
    
    # Process each layer
    for layer_idx in tqdm(range(len(layers)), desc="Compressing"):
        layer = layers[layer_idx].cuda()
          
        # Apply calibration and compression (skip first/last layers)
        if layer_idx not in [0, len(layers) - 1]:
            # Calibrate current layer
            bit8_window, bit4_window, head_windows = calibrate_layer_windows(
                layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args
            )
            
            # Store detailed results
            bit8_windows, bit4_windows = [], []
            for hw in head_windows:
                bit8_windows.append(hw['bit8_relative'])
                bit4_windows.append(hw['bit4_relative'])

            # logger.info(f"L{layer_idx}: bit8={bit8_windows}, bit4={bit4_windows}")
            
            # Store configuration
            bits_alloc[layer_idx] = {
                "bit8": bit8_windows,
                "bit4": bit4_windows,
                "sink": 256
            }
            
            # Apply compression

            replace_mp_triton_for_block(
                layer, layer_idx, args,
                bit8_window_sizes=bit8_windows,
                bit4_window_sizes=bit4_windows,
                sink_window_size=256
            )
        
        # Update inputs for next layer
        # inps = ori_output if args.mse_output == "full" else layer(inps, **layer_kwargs)[0]

        samples_inps = batch_layer_infer(layer, samples_inps, samples_layer_kwargs, args)
        layer.cpu()
        torch.cuda.empty_cache()
        # Save checkpoint if needed
        # if config.save_checkpoints and (layer_idx + 1) % config.checkpoint_interval == 0:
        #     save_checkpoint(layer_idx, windows_dict, bits_alloc)
    
    # ========== Phase 3: Save Results ==========
    # if config.use_calibration:
    #     output_path = getattr(args, 'window_config_path', 'window_config.json')
    #     save_final_results(windows_dict, bits_alloc, config, output_path)
    del args.current_attention
    logger.info("Compression complete!")
    return bits_alloc