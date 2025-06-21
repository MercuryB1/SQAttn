import torch
from loguru import logger
import math
import torch.nn.functional as F
from sparse_quant_attn.compression.attn_replacer import repeat_kv
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter


def cal_attn_weight(query, key, is_causal=True, attn_mask=None):
                    key = repeat_kv(key, 6)
                    query = query.contiguous()
                    key = key.contiguous()
                    L, S = query.size(-2), key.size(-2)
                    scale_factor = 1 / math.sqrt(query.size(-1))
                    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
                    if is_causal:
                        assert attn_mask is None
                        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
                        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                        attn_bias.to(query.dtype)

                    if attn_mask is not None:
                        if attn_mask.dtype == torch.bool:
                            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                        else:
                            attn_bias = attn_mask + attn_bias
                    attn_weight = query @ key.transpose(-2, -1) * scale_factor
                    attn_weight += attn_bias
                    attn_weight = torch.softmax(attn_weight, dim=-1)
                    import torch.nn.functional as F
                    # attn_weight_pool=F.avg_pool2d(attn_weight, kernel_size=(10,10),stride=(10,10))
                    attn_weight_pool=F.max_pool2d(attn_weight, (10,10), stride=(10,10))
                    return attn_weight_pool


@torch.no_grad()
def get_static_important_token_per_head(layers, inps, layer_kwargs, args):
    layer_kwargs["output_attentions"] = True
    token_list = dict()
    threshold = 0.001  # softmax阈值
    freq_threshold = 0.1  # q被attend到的频率阈值（可调）
    num_batches = inps.shape[0]
    seq_len = inps.shape[1]
    for layer_idx, layer in enumerate(layers):
        layer = layer.cuda()
        outputs, attn_weights = layer(inps, **layer_kwargs)  # attn_weights: [B, H, Q, K]
        # 统计每个head的k被q attend到的频率
        layer_token_list = dict()
        # TODO 要不要加mask？
        for head_idx in range(attn_weights.shape[1]):
            token_counter = Counter()
            # 对每个batch分别统计k被attend的频率
            for batch_idx in range(attn_weights.shape[0]):
                attn = attn_weights[batch_idx, head_idx]  # [Q, K]
                # 对每个k，统计有多少q使得softmax>threshold
                attended = (attn > threshold).sum(dim=0)  # [K]
                # 计算该batch中每个k被attend的频率
                freq = attended.float() / attn_weights.shape[2]  # [K]
                # 找出该batch中频率大于阈值的k idx
                important_k_idx = (freq > freq_threshold).nonzero(as_tuple=True)[0]
                # import pdb; pdb.set_trace()
                # 将该batch的重要k idx映射到token id
                for k_idx in important_k_idx.tolist():
                    token_id = layer.self_attn.kv_token_ids[batch_idx, k_idx].item()
                    token_counter[token_id] += 1
            # 只保留在多个batch中都出现的token id
            min_count = int(attn_weights.shape[0] * 0.5)  # 例如至少在50% batch中出现
            important_token_ids = [tid for tid, cnt in token_counter.items() if cnt >= min_count]
            layer_token_list[head_idx] = important_token_ids
        token_list[layer_idx] = layer_token_list
    # 可选：保存或打印结果
    logger.info(f"Important token list per layer/head: {token_list}")
    # import pdb; pdb.set_trace()
    return token_list


# @torch.no_grad()
# def get_static_important_token_per_head(layers, inps, layer_kwargs, args):
#     layer_kwargs["output_attentions"] = True
#     token_list = dict()
#     for layer_idx, layer in enumerate(layers):
#         layer = layer.cuda()
#         outputs, attn_weights = layer(inps, **layer_kwargs)
#         for batch_idx in range(attn_weights.shape[0]):
              
        # attn_weights=F.max_pool2d(attn_weights, (10,10), stride=(10,10))
        # attn_map = attn_weights[0].detach().to(torch.float32).cpu() # [H, Q, K]
        # os.makedirs(f'attn_eager/layer_{layer_idx}', exist_ok=True)
        # for h in range(attn_weights.shape[1]):
            # plt.imshow(attn_map[h], cmap='coolwarm', aspect='auto')
            # sns.heatmap(
            #     attn_weights[0][h].detach().to(torch.float32).cpu().numpy(),
            #     # xticklabels=tokens,
            #     # yticklabels=tokens,
            #     cmap="magma",  # 类似你提供图片的配色
            #     square=True,
            #     cbar=True
            # )
            # # plt.colorbar()
            # plt.title(f'Layer {layer_idx} Head {h}')
            # plt.savefig(f'attn_eager/layer_{layer_idx}/head_{h}.png')
            # plt.close()
    # import pdb; pdb.set_trace()