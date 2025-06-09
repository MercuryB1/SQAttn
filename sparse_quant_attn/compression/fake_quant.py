import torch
from torch import nn
from functools import partial
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, w_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (w_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, w_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (w_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(x, a_bits=8):
    x_shape = x.shape
    x.view(-1, x_shape[-1])
    scales = x.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (a_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x


@torch.no_grad()
def quantize_activation_per_tensor_absmax(x, a_bits=8):
    x_shape = x.shape
    x.view(-1, x_shape[-1])
    scales = x.abs().max()
    q_max = 2 ** (a_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    x.div_(scales).round_().mul_(scales)
    return x


@torch.no_grad()
def pseudo_quantize_tensor(w: torch.Tensor, group_size=0, zero_point=False, bit_width=8):
        org_w_shape = w.shape
        if group_size > 0:
            assert org_w_shape[-1] % group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({group_size})!"
            w = w.reshape(-1, group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**bit_width - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (bit_width - 1) - 1
            min_int = -(2 ** (bit_width - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

@torch.no_grad()
def pseudo_fp_quantize_tensor(tensor: torch.Tensor, group_size=0, zero_point=False, bit="e4m3"):
    from qtorch.quant import float_quantize
    if bit == "e4m3":
        e_bits = 4
        m_bits = 3
        fp_dtype = torch.float8_e4m3fn
    elif bit == "e5m2":
        e_bits = 5
        m_bits = 2
        fp_dtype = torch.float8_e5m2
    finfo = torch.finfo(fp_dtype)
    qmin, qmax = finfo.min, finfo.max

    qmax = torch.tensor(qmax)
    qmin = torch.tensor(qmin)

    def quant(tensor, scales, zeros, qmax, qmin):
        scaled_tensor = tensor / scales + zeros
        scaled_tensor = torch.clip(scaled_tensor, qmin.cuda(), qmax.cuda())
        org_dtype = scaled_tensor.dtype
        q_tensor = float_quantize(scaled_tensor.float(), e_bits, m_bits, rounding="nearest")
        q_tensor.to(org_dtype)
        return q_tensor

    def dequant(tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor
    
    def quant_dequant(tensor, scales, zeros, qmax, qmin):
        tensor = quant(tensor, scales, zeros, qmax, qmin)
        tensor = dequant(tensor, scales, zeros)
        return tensor
    
    def get_qparams(tensor_range, device, sym=False):
        min_val, max_val = tensor_range[0], tensor_range[1]
        qmin = qmin.to(device)
        qmax = qmax.to(device)
        if sym:
            abs_max = torch.max(max_val.abs(), min_val.abs())
            abs_max = abs_max.clamp(min=1e-5)
            scales = abs_max / qmax
            zeros = torch.tensor(0.0)
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / (qmax - qmin)
            zeros = (qmin - torch.round(min_val / scales)).clamp(qmin, qmax)
        return scales, zeros, qmax, qmin

    def reshape_tensor(tensor, granularity="per_group", group_size=0, allow_padding=False):
        if granularity == "per_group":
            t = tensor.reshape(-1, group_size)
        else:
            t = tensor
        return t

    def get_tensor_qparams(tensor):
        tensor = reshape_tensor(tensor)
        tensor_range = get_tensor_range(tensor)
        scales, zeros, qmax, qmin = get_qparams(tensor_range, tensor.device)
        return tensor, scales, zeros, qmax, qmin
    
    def restore_tensor(tensor, shape):
        if tensor.shape == shape:
            t = tensor
        else:
            t = tensor.reshape(shape)
        return t
    
    def fake_quant_tensor(tensor):
        org_shape = tensor.shape
        org_dtype = tensor.dtype
        tensor, scales, zeros, qmax, qmin = get_tensor_qparams(tensor)
        tensor = quant_dequant(tensor, scales, zeros, qmax, qmin)
        tensor = restore_tensor(tensor, org_shape).to(org_dtype)
        return tensor
    
    return fake_quant_tensor(tensor)
