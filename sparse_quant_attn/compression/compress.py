import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from sparse_quant_attn.utils.model_utils import get_blocks, move_embed
from sparse_quant_attn.compression.window_search import model_infer
from sparse_quant_attn.utils.model_utils import batch_layer_infer
from sparse_quant_attn.compression.calibration import get_calib_dataset
from sparse_quant_attn.compression.attn_replacer import replace_sdpa_for_block_with_attn_weights, replace_mp_triton_for_block

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


# ============= Phase 2: Calibration Functions =============

# too slow, recommend using gpu version
def compute_attention_histogram(attention_map: np.ndarray, max_len: int) -> np.ndarray:
    """
    Compute energy distribution histogram for attention map
    """
    histogram = np.zeros(max_len)
    seq_len = attention_map.shape[0]
    
    for i in range(seq_len):
        for j in range(min(i + 1, seq_len)):
            distance = i - j + 1
            if distance < max_len:
                energy = attention_map[i, j] * distance
                histogram[distance] += energy

    
    return histogram


def compute_attention_histogram_gpu(attention_map: torch.Tensor, max_len: int) -> np.ndarray:
   """
   Ultra-fast attention histogram computation using torch.bincount
   
   Args:
       attention_map: [seq_len, seq_len] attention weights on GPU
       max_len: Maximum distance to consider
   
   Returns:
       histogram: Energy distribution histogram as numpy array, padded to max_len
   """
   seq_len = attention_map.shape[0]
   device = attention_map.device
   
   # Step 1: Create distance matrix D[i,j] = i - j
   row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
   col_indices = torch.arange(seq_len, device=device).unsqueeze(0)
   distance_matrix = row_indices - col_indices
   
   # Step 2: Create weight matrix (using distance as weight: weight = d)
   weight_matrix = distance_matrix.float()
   
   # Step 3: Compute weighted energy matrix E = A * W
   weighted_energy_matrix = attention_map * weight_matrix
   
   # Step 4: Use bincount on lower triangular part
   tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
   distances_flat = distance_matrix[tril_mask]
   weighted_energies_flat = weighted_energy_matrix[tril_mask]
   
   # Compute histogram using bincount, ensure it has length max_len
   histogram = torch.bincount(
       distances_flat,
       weights=weighted_energies_flat,
       minlength=max_len  # This ensures output has at least max_len elements
   )
   
   # If histogram is longer than max_len, truncate; if shorter, it's already padded with zeros
   if len(histogram) > max_len:
       histogram = histogram[:max_len]
   
   return histogram.cpu().numpy()


def find_optimal_window(histogram: np.ndarray, threshold: float) -> int:
    """
    Find minimum window size that retains threshold of total energy
    """
    total_energy = np.sum(histogram)
    if total_energy == 0:
        return 1
    
    cumulative_energy = 0
    for d in range(0, len(histogram), 128):
        cumulative_energy += sum(histogram[d:d+128])
        if cumulative_energy >= threshold * total_energy:
            return d + 128
    
    return len(histogram)


def calibrate_layer_windows(layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args):
    """
    Calibrate window sizes for a single layer
    """
    assert len(samples_inps) == len(samples_layer_kwargs)
    num_heads = layer.self_attn.config.num_attention_heads
    
    # Initialize histograms
    histograms = [np.zeros(max_len) for _ in range(num_heads)]
    
    # Process all samples
    # for sample_data in tqdm(analysis_dataset, desc=f"L{layer_idx} calibration", leave=False):
    for i in tqdm(range(len(samples_inps)), desc=f"L{layer_idx} calibration"):
        inps = samples_inps[i]
        layer_kwargs = samples_layer_kwargs[i]
        # get attn weights
        bit8_window_sizes = [max_len * 2] * num_heads  # 使用大窗口获取完整attention
        bit4_window_sizes = [0] * num_heads
        replace_sdpa_for_block_with_attn_weights(layer, i, args,
            bit8_window_sizes=bit8_window_sizes,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=32
        )
        _ = layer(inps, **layer_kwargs)[0]
        attention_maps = args.current_attention.detach()
     
        # Accumulate energy for each head
        for head_idx, attn_map in enumerate(attention_maps.squeeze(0)):
            
            if attn_map is not None:
                hist = compute_attention_histogram_gpu(attn_map, max_len)
                histograms[head_idx] += hist

    # Find optimal windows for each head
    windows = []
    for head_idx, histogram in enumerate(histograms):
        bit8_window = find_optimal_window(histogram, args.bit8_thres_cos)
        # bit4_window = min(bit8_window * 2, max_len)
        bit4_window = find_optimal_window(histogram, args.bit4_thres_cos)
        windows.append({
            'head_idx': head_idx,
            'bit8': bit8_window,
            'bit4': bit4_window
        })
    
    # Return max windows for layer (or could return per-head)
    bit8_max = max(w['bit8'] for w in windows)
    bit4_max = max(w['bit4'] for w in windows)
    
    return bit8_max, bit4_max, windows

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
    windows_dict = {}
    
    # Get original outputs if needed
    # if args.mse_output == "full":
    #     ori_model_outputs = model_infer(model, inps, layer_kwargs, args)
    
    # Process each layer
    for layer_idx in tqdm(range(len(layers)), desc="Compressing"):
        layer = layers[layer_idx].cuda()
        
        # Store original output if needed
        # if args.mse_output == "full":
        #     ori_output = layer(inps, **layer_kwargs)[0]
        
        # Apply calibration and compression (skip first/last layers)
        if layer_idx not in [0, len(layers) - 1]:
            # Calibrate current layer
            bit8_window, bit4_window, head_windows = calibrate_layer_windows(
                layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args
            )
            
            # Store detailed results
            for hw in head_windows:
                windows_dict[(layer_idx, hw['head_idx'])] = {
                    'bit8': hw['bit8'],
                    'bit4': hw['bit4']
                }
            
            logger.info(f"L{layer_idx}: bit8={bit8_window}, bit4={bit4_window}")
            
            
            # Store configuration
            bits_alloc[layer_idx] = {
                "bit8": bit8_window,
                "bit4": bit4_window,
                "sink": 128
            }
            
            # Apply compression
            replace_mp_triton_for_block(
                layer, layer_idx, args,
                bit8_window_sizes=bit8_window,
                bit4_window_sizes=bit4_window,
                sink_window_size=128
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
    
    logger.info("Compression complete!")
    return bits_alloc