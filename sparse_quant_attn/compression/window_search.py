import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sparse_quant_attn.compression.attn_replacer import replace_sdpa_for_block_with_attn_weights


# ============= Phase 2: Calibration Functions =============

# too slow, recommend using gpu version
def compute_attention_histogram(attention_map: np.ndarray, max_len: int):
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


def compute_attention_histogram_gpu(attention_map: torch.Tensor, max_len: int):
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


def compute_relative_attention_histogram_gpu(
    attention_map: torch.Tensor, 
    num_bins: int = 100
):
    """
    Ultra-fast relative attention histogram computation using torch.bincount

    Args:
        attention_map: [seq_len, seq_len] attention weights on GPU
        num_bins: Number of bins for relative distance [0, 1] interval
    
    Returns:
        histogram: Energy distribution histogram with relative distance bins
    """
    seq_len = attention_map.shape[0]
    device = attention_map.device
    
    # Step 1: Create absolute distance matrix D[i,j] = i - j
    row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
    col_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    distance_matrix = row_indices - col_indices
    
    # Risk weight matrix W[i,j] = i - j (using pure distance as weight)
    weight_matrix = distance_matrix.float()
    
    # Step 2: Create relative distance matrix D_rel[i,j] = (i-j) / i
    # Handle division by zero for i=0
    row_indices_safe = row_indices.float().clone()
    row_indices_safe[0] = 1  # Avoid division by zero
    relative_distance_matrix = distance_matrix.float() / row_indices_safe
    relative_distance_matrix[0, 0] = 0  # Fix the (0,0) position
    
    # Step 3: Create bin index matrix B[i,j] = floor(d_rel * num_bins)
    # Clamp to [0, num_bins-1] to handle edge cases
    bin_index_matrix = (relative_distance_matrix * num_bins).floor().long()
    bin_index_matrix = torch.clamp(bin_index_matrix, 0, num_bins - 1)
    
    # Step 4: Compute weighted energy matrix E = A * W
    weighted_energy_matrix = attention_map * weight_matrix
    
    # Step 5: Use bincount on lower triangular part
    tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    bin_indices_flat = bin_index_matrix[tril_mask]
    weighted_energies_flat = weighted_energy_matrix[tril_mask]
    
    # Compute histogram using bincount
    histogram = torch.bincount(
        bin_indices_flat,
        weights=weighted_energies_flat,
        minlength=num_bins
    )
    
    return histogram.cpu().numpy()

def find_optimal_relative_window(
    histogram: np.ndarray, 
    threshold: float, 
    num_bins: int = 100
):
    """
    Find optimal relative window size that retains threshold of total energy
    
    Args:
        histogram: Relative distance energy distribution histogram
        threshold: Energy retention threshold (e.g., 0.95)
        num_bins: Number of bins used in histogram
    
    Returns:
        Optimal relative window size (float between 0 and 1)
    """
    total_energy = np.sum(histogram)
    if total_energy == 0:
        return 0.1  # Default to 10% relative window
    
    cumulative_energy = 0
    optimal_bin = 0
    
    for bin_idx in range(len(histogram)):
        cumulative_energy += histogram[bin_idx]
        if cumulative_energy >= threshold * total_energy:
            optimal_bin = bin_idx
            break
    
    # Convert bin index to relative distance
    # Add 1 to include the entire bin range
    relative_window = (optimal_bin + 1) / num_bins
    
    return min(relative_window, 1.0)

def find_optimal_window(histogram: np.ndarray, threshold: float):
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


def calibrate_layer_windows_absolute(layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args):
    """
    Calibrate absolute window sizes for a single layer
    Returns fixed window sizes as integers
    """
    assert len(samples_inps) == len(samples_layer_kwargs)
    num_heads = layer.self_attn.config.num_attention_heads
    
    # Initialize histograms
    histograms = [np.zeros(max_len) for _ in range(num_heads)]
    
    # Process all samples
    for i in tqdm(range(len(samples_inps)), desc=f"L{layer_idx} absolute calibration"):
        inps = samples_inps[i]
        layer_kwargs = samples_layer_kwargs[i]
        
        # Get attention weights
        bit8_window_sizes = [max_len * 2] * num_heads
        bit4_window_sizes = [0] * num_heads
        
        replace_sdpa_for_block_with_attn_weights(
            layer, i, args,
            bit8_window_sizes=bit8_window_sizes,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=128
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
        bit8_window = find_optimal_window(histogram, args.bit8_thres)
        bit4_window = find_optimal_window(histogram, args.bit4_thres)
        
        windows.append({
            'head_idx': head_idx,
            'bit8': bit8_window,
            'bit4': bit4_window
        })
    
    # Return max windows for layer
    bit8_max = max(w['bit8'] for w in windows)
    bit4_max = max(w['bit4'] for w in windows)
    
    return bit8_max, bit4_max, windows


def calibrate_layer_windows_relative(
    layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args
):
    """
    Calibrate relative window sizes excluding sink tokens
    """
    assert len(samples_inps) == len(samples_layer_kwargs)
    num_heads = layer.self_attn.config.num_attention_heads
    num_bins = getattr(args, 'relative_bins', 200)
    sink_len = getattr(args, 'sink_window_size', 256)
    
    # Initialize histograms
    histograms = [np.zeros(num_bins) for _ in range(num_heads)]
    
    # Process all samples
    for i in tqdm(range(len(samples_inps)), desc=f"L{layer_idx} relative calibration (sink-aware)"):
        inps = samples_inps[i]
        layer_kwargs = samples_layer_kwargs[i]
        
        # Get attention weights
        bit8_window_sizes = [max_len * 2] * num_heads
        bit4_window_sizes = [0] * num_heads
        
        replace_sdpa_for_block_with_attn_weights(
            layer, i, args,
            bit8_window_sizes=bit8_window_sizes,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=sink_len
        )
        
        _ = layer(inps, **layer_kwargs)[0]
        attention_maps = args.current_attention.detach()
        
        # Accumulate energy for each head (excluding sink)
        for head_idx, attn_map in enumerate(attention_maps.squeeze(0)):
            if attn_map is not None:
                hist = compute_relative_attention_histogram_gpu_with_sink(
                    attn_map, num_bins, sink_len
                )
                histograms[head_idx] += hist
    
    # Find optimal relative windows
    windows = []
    for head_idx, histogram in enumerate(histograms):
        bit8_relative = find_optimal_relative_window(histogram, args.bit8_thres, num_bins)
        bit4_relative = find_optimal_relative_window(histogram, args.bit4_thres, num_bins)
        
        windows.append({
            'head_idx': head_idx,
            'bit8_relative': bit8_relative,
            'bit4_relative': bit4_relative,
            'sink_len': sink_len
        })
    
    bit8_max_relative = max(w['bit8_relative'] for w in windows)
    bit4_max_relative = max(w['bit4_relative'] for w in windows)
    
    return bit8_max_relative, bit4_max_relative, windows


# Wrapper function to choose which one to use
def calibrate_layer_windows(layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args):
    """
    Wrapper function that calls appropriate calibration based on args
    """
    if getattr(args, 'use_relative_distance', False):
        return calibrate_layer_windows_relative(
            layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args
        )
    else:
        return calibrate_layer_windows_absolute(
            layer, layer_idx, samples_inps, samples_layer_kwargs, max_len, args
        )

def compute_relative_attention_histogram_gpu_with_sink(
    attention_map: torch.Tensor, 
    num_bins: int = 200,
    sink_len: int = 128
):
    """
    Ultra-fast relative attention histogram computation excluding sink tokens
    
    Args:
        attention_map: [seq_len, seq_len] attention weights on GPU
        num_bins: Number of bins for relative distance [0, 1] interval
        sink_len: Number of sink tokens (default: 128)
    
    Returns:
        histogram: Energy distribution histogram for sliding window world only
    """
    seq_len = attention_map.shape[0]
    device = attention_map.device
    
    # Step 0: Create non-sink mask - exclude all attention to sink tokens
    non_sink_mask = torch.ones_like(attention_map)
    non_sink_mask[:, :sink_len] = 0
    
    # Apply mask to remove sink attention
    A_masked = attention_map * non_sink_mask
    
    # Step 1: Create absolute distance matrix D[i,j] = i - j
    row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
    col_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    distance_matrix_abs = row_indices - col_indices
    
    # Step 2: Create relative distance matrix with sink-aware context length
    # Context length is the sliding window world size: i - sink_len
    context_len = row_indices.float() - sink_len
    context_len_safe = torch.clamp(context_len, min=1)  # Avoid division by zero
    
    # Relative distance: (i - j) / (i - sink_len)
    relative_distance_matrix = distance_matrix_abs.float() / context_len_safe
    
    # For positions before or at sink boundary, set relative distance to 0
    relative_distance_matrix[row_indices.squeeze() <= sink_len] = 0
    
    # Step 3: Create bin index matrix
    bin_index_matrix = (relative_distance_matrix * num_bins).floor().long()
    bin_index_matrix = torch.clamp(bin_index_matrix, 0, num_bins - 1)
    
    # Step 4: Risk weight matrix based on relative distance
    # Risk grows with relative jump in the sliding window world
    risk_weight_matrix = relative_distance_matrix * num_bins + 1
    
    # Step 5: Compute weighted energy matrix using masked attention
    weighted_energy_matrix = A_masked * risk_weight_matrix
    
    # Step 6: Final mask combining lower triangular and non-sink
    tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    final_mask = tril_mask & non_sink_mask.bool()
    
    # Extract valid elements
    bin_indices_flat = bin_index_matrix[final_mask]
    weighted_energies_flat = weighted_energy_matrix[final_mask]
    
    # Compute histogram using bincount
    histogram = torch.bincount(
        bin_indices_flat,
        weights=weighted_energies_flat,
        minlength=num_bins
    )
    
    return histogram.cpu().numpy()


def compute_absolute_attention_histogram_gpu_with_sink(
    attention_map: torch.Tensor, 
    max_len: int,
    sink_len: int = 128
):
    """
    Compute absolute distance histogram excluding sink tokens
    
    Args:
        attention_map: [seq_len, seq_len] attention weights on GPU
        max_len: Maximum distance to consider
        sink_len: Number of sink tokens (default: 128)
    
    Returns:
        histogram: Energy distribution for sliding window world only
    """
    seq_len = attention_map.shape[0]
    device = attention_map.device
    
    # Step 0: Create non-sink mask
    non_sink_mask = torch.ones_like(attention_map)
    non_sink_mask[:, :sink_len] = 0
    
    # Apply mask to remove sink attention
    A_masked = attention_map * non_sink_mask
    
    # Create distance matrix
    row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
    col_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    distance_matrix = row_indices - col_indices
    
    # Weight matrix (pure distance)
    weight_matrix = distance_matrix.float()
    
    # Weighted energy
    weighted_energy_matrix = A_masked * weight_matrix
    
    # Final mask: lower triangular AND non-sink
    tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    final_mask = tril_mask & non_sink_mask.bool()
    
    distances_flat = distance_matrix[final_mask]
    weighted_energies_flat = weighted_energy_matrix[final_mask]
    
    # Compute histogram
    histogram = torch.bincount(
        distances_flat,
        weights=weighted_energies_flat,
        minlength=max_len
    )
    
    if len(histogram) > max_len:
        histogram = histogram[:max_len]
    
    return histogram.cpu().numpy()