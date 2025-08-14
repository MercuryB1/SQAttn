import torch
import torch.nn.functional as F
from loguru import logger
import math
from typing import Tuple, Optional, List


class TokenAwareMetrics:
    """
    Token-aware metrics for evaluating attention pattern preservation.
    Measures both local energy retention and hub energy retention.
    """
    
    def __init__(self, 
                 local_window_size: int = 128,
                 hub_top_k: int = 32,
                 alpha: float = 0.5,
                 beta: float = 0.5):
        """
        Args:
            local_window_size: Size of local window for measuring local energy (D_local)
            hub_top_k: Number of top hub tokens to consider
            alpha: Weight for local energy in combined metric
            beta: Weight for hub energy in combined metric
        """
        self.local_window_size = local_window_size
        self.hub_top_k = hub_top_k
        self.alpha = alpha
        self.beta = beta
    
    @torch.no_grad()
    def compute_attention_weights(self, query, key, value, is_causal=True):
        """
        Compute attention weights without actually applying them.
        This mimics the attention computation to get the softmax weights.
        """
        # Handle potential shape differences
        batch_size, num_heads, seq_len_q, head_dim = query.shape
        _, _, seq_len_k, _ = key.shape
        
        # Scale factor
        scale_factor = 1 / math.sqrt(head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device),
                diagonal=1
            )
            attn_scores.masked_fill_(causal_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights
    
    @torch.no_grad()
    def compute_local_energy_retention(self, 
                                      attn_weights: torch.Tensor,
                                      mask: torch.Tensor) -> float:
        """
        Compute Local Energy Retention (LER) rate.
        
        Args:
            attn_weights: Original attention weights [B, H, Q, K] or [H, Q, K]
            mask: Binary mask indicating which positions are retained (True = retained)
        
        Returns:
            LER: Local Energy Retention rate
        """
        device = attn_weights.device
        
        # Handle different input dimensions
        if attn_weights.dim() == 4:
            # [B, H, Q, K] -> average over batch
            attn_weights = attn_weights.mean(dim=0)  # [H, Q, K]
        
        if mask.dim() == 2:
            # Expand mask to match number of heads if needed
            mask = mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        
        num_heads, seq_len_q, seq_len_k = attn_weights.shape
        
        # Create local window mask
        q_indices = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_indices = torch.arange(seq_len_k, device=device).unsqueeze(0)
        distance = torch.abs(q_indices - k_indices)
        local_mask = distance < self.local_window_size
        
        # Expand local_mask for all heads
        local_mask = local_mask.unsqueeze(0).expand(num_heads, -1, -1)
        
        # Calculate total local energy per head
        total_local_energy = (attn_weights * local_mask.float()).sum(dim=[1, 2])  # [H]
        
        # Calculate retained local energy per head
        retained_mask = mask & local_mask
        retained_local_energy = (attn_weights * retained_mask.float()).sum(dim=[1, 2])  # [H]
        
        # Calculate LER per head (avoid division by zero)
        ler_per_head = torch.where(
            total_local_energy > 1e-10,
            retained_local_energy / total_local_energy,
            torch.ones_like(total_local_energy)
        )
        
        return ler_per_head  # [H]
    
    @torch.no_grad()
    def compute_hub_energy_retention(self,
                                    attn_weights: torch.Tensor,
                                    mask: torch.Tensor,
                                    sink_size: int = 0) -> float:
        """
        Compute Hub Energy Retention (HER) rate.
        
        Args:
            attn_weights: Original attention weights [B, H, Q, K] or [H, Q, K]
            mask: Binary mask indicating which positions are retained
            sink_size: Size of sink tokens (always considered as hubs)
        
        Returns:
            HER: Hub Energy Retention rate per head
        """
        device = attn_weights.device
        
        # Handle different input dimensions
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=0)  # [H, Q, K]
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        
        num_heads, seq_len_q, seq_len_k = attn_weights.shape
        her_per_head = []
        
        for h in range(num_heads):
            head_attn = attn_weights[h]  # [Q, K]
            head_mask = mask[h]  # [Q, K]
            
            # Compute column sums to identify hubs
            column_sums = head_attn.sum(dim=0)  # [K]
            
            # Always include sink tokens as hubs
            hub_scores = column_sums.clone()
            if sink_size > 0:
                hub_scores[:sink_size] = float('inf')  # Ensure sink tokens are selected
            
            # Find top-k hub indices
            k = min(self.hub_top_k, seq_len_k)
            _, hub_indices = torch.topk(hub_scores, k)
            
            # Create hub mask
            hub_mask = torch.zeros(seq_len_k, dtype=torch.bool, device=device)
            hub_mask[hub_indices] = True
            
            # Calculate total hub energy
            total_hub_energy = head_attn[:, hub_mask].sum()
            
            # Calculate retained hub energy
            retained_hub_mask = head_mask[:, hub_mask]
            retained_hub_energy = (head_attn[:, hub_mask] * retained_hub_mask.float()).sum()
            
            # Calculate HER (avoid division by zero)
            if total_hub_energy > 1e-10:
                her = retained_hub_energy / total_hub_energy
            else:
                her = 1.0
            
            her_per_head.append(her)
        
        return torch.tensor(her_per_head, device=device)  # [H]
    
    @torch.no_grad()
    def compute_combined_metric(self,
                               attn_weights: torch.Tensor,
                               mask: torch.Tensor,
                               sink_size: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined token-aware metric.
        
        Returns:
            combined_score: Alpha * LER + Beta * HER per head
            ler_per_head: Local Energy Retention per head
            her_per_head: Hub Energy Retention per head
        """
        ler_per_head = self.compute_local_energy_retention(attn_weights, mask)
        her_per_head = self.compute_hub_energy_retention(attn_weights, mask, sink_size)
        
        combined_score = self.alpha * ler_per_head + self.beta * her_per_head
        
        return combined_score, ler_per_head, her_per_head


@torch.no_grad()
def search_bit8_window_size_with_token_metrics(
    layers, 
    layer_idx, 
    head_id,
    inps, 
    bit8_window_candidate_sizes,
    layer_kwargs,
    args,
    metric_calculator: TokenAwareMetrics,
    ler_threshold: float = 0.95,
    her_threshold: float = 0.95
):
    """
    Search for optimal bit8 window size using token-aware metrics.
    
    Args:
        layers: Model layers
        layer_idx: Current layer index
        head_id: Current head index
        inps: Input tensors
        bit8_window_candidate_sizes: Candidate window sizes to search
        layer_kwargs: Layer forward kwargs
        args: Global arguments
        metric_calculator: TokenAwareMetrics instance
        ler_threshold: Minimum Local Energy Retention threshold
        her_threshold: Minimum Hub Energy Retention threshold
    
    Returns:
        Optimal window size
    """
    from sparse_quant_attn.compression.attn_replacer import replace_sdpa_for_block
    
    # Get original attention weights (without quantization)
    args_orig = type('obj', (object,), {'quant': False, 'vis_attn': False})()
    replace_sdpa_for_block(layers[layer_idx], layer_idx, args_orig,
                          bit8_window_sizes=bit8_window_candidate_sizes[-1],
                          bit4_window_sizes=0,
                          sink_window_size=32)
    
    # Get Q, K, V from the layer
    with torch.no_grad():
        # We need to hook into the attention computation to get Q, K, V
        # This is a simplified approach - you may need to adjust based on your model
        layer = layers[layer_idx].cuda()
        hidden_states = inps
        
        # Get attention module
        attn_module = layer.self_attn
        
        # Compute Q, K, V (this depends on your model architecture)
        # For Qwen2, it's typically like this:
        hidden_states_shape = hidden_states.shape
        bsz, q_len, _ = hidden_states_shape
        
        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
        
        # Compute original attention weights
        orig_attn_weights = metric_calculator.compute_attention_weights(
            query_states, key_states, value_states, is_causal=True
        )
    
    # Search for optimal window size
    for w in bit8_window_candidate_sizes:
        # Create mask for this window size
        seq_len = orig_attn_weights.shape[-1]
        
        # Construct per-head mask
        bit8_window_sizes = [bit8_window_candidate_sizes[-1]] * layers[layer_idx].self_attn.config.num_attention_heads
        bit8_window_sizes[head_id] = w
        bit4_window_sizes = [0] * layers[layer_idx].self_attn.config.num_attention_heads
        
        # Create the attention mask for evaluation
        mask = construct_mix_bit_mask_per_head(
            seq_len=seq_len,
            bit8_window_sizes=bit8_window_sizes,
            bit4_window_sizes=bit4_window_sizes,
            sink_window_size=32,
            device=orig_attn_weights.device
        )
        
        # Compute metrics
        combined_score, ler_per_head, her_per_head = metric_calculator.compute_combined_metric(
            orig_attn_weights, mask, sink_size=32
        )
        
        # Check if this head meets both thresholds
        head_ler = ler_per_head[head_id].item()
        head_her = her_per_head[head_id].item()
        
        logger.info(f"[Layer {layer_idx} | Head {head_id}] Window {w}: LER={head_ler:.4f}, HER={head_her:.4f}")
        
        if head_ler >= ler_threshold and head_her >= her_threshold:
            return w
    
    # If no window size meets the criteria, return the largest
    return bit8_window_candidate_sizes[-1]


def construct_mix_bit_mask_per_head(seq_len, bit8_window_sizes, bit4_window_sizes, sink_window_size, device='cuda'):
    """
    Helper function to construct per-head attention masks.
    (Reused from original code with minor modifications)
    """
    num_heads = len(bit8_window_sizes)
    q_idx = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(2)  # (1, L, 1)
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(1)  # (1, 1, L)
    
    q_idx = q_idx.expand(num_heads, seq_len, 1)
    kv_idx = kv_idx.expand(num_heads, 1, seq_len)
    
    bit8_window_sizes_t = torch.tensor(bit8_window_sizes, device=device).view(-1, 1, 1)
    bit4_window_sizes_t = torch.tensor(bit4_window_sizes, device=device).view(-1, 1, 1)
    
    # Causal mask
    causal_mask = kv_idx <= q_idx
    
    # 8-bit window mask
    bit8_window_mask = kv_idx > (q_idx - bit8_window_sizes_t)
    
    # Sink mask
    sink_mask = kv_idx < sink_window_size
    
    # Final mask: causal AND (sink OR window)
    final_mask = causal_mask & (sink_mask | bit8_window_mask)
    
    return final_mask  # [num_heads, seq_len, seq_len]