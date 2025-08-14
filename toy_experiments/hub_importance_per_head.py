# -*- coding: utf-8 -*-
import os
import random
import types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================== 基本配置 ==================
# MODEL_ID = "/mnt/disk3/hg/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
MODEL_ID = "/mnt/disk3/hg/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 实验参数
K = 8                        # 每个 head 要消融的列（Top-K）
CALIB_SAMPLES = 128         # 校准样本数（累计列能量）
TEST_SAMPLES  = 200         # 评估样本数（计算 PPL）
MAX_TOKENS = 128            # 让校准与评估一致，降低偏移
SEED = 42

# 只动一个 head（可修改）
TARGET_LAYER = 12           # 例：第 13 层
TARGET_HEAD  = 3            # 例：该层的第 4 个头

# 本地数据（离线可用）
USE_LOCAL_DATA = False
CALIB_TXT = "./calib.txt"   # 每行一条文本
TEST_TXT  = "./test.txt"

# 随机性/性能
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ================== 加载模型与分词器（单卡） ==================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    output_attentions=True,        # 校准阶段需要 attentions
    attn_implementation="eager",   # 兼容 output_attentions
    device_map=None,               # 单卡避免设备不一致
).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

NUM_LAYERS   = model.config.num_hidden_layers
NUM_HEADS    = model.config.num_attention_heads
NUM_KV_HEADS = getattr(model.config, "num_key_value_heads", NUM_HEADS)
HIDDEN_SIZE  = model.config.hidden_size
HEAD_DIM     = HIDDEN_SIZE // NUM_HEADS
MAX_POS      = getattr(model.config, "max_position_embeddings", 4096)

# ================== 数据加载与清洗 ==================
def load_lines_from_txt(path, limit=None):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
                if limit and len(lines) >= limit:
                    break
    return lines

def token_len(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=MAX_TOKENS, padding=False, add_special_tokens=True)
    return int(enc["input_ids"].size(1))

def filter_texts(texts, min_tokens=2, max_keep=None):
    kept = []
    for t in texts:
        if t and token_len(t) >= min_tokens:
            kept.append(t)
            if max_keep and len(kept) >= max_keep:
                break
    return kept

if USE_LOCAL_DATA:
    raw_train = load_lines_from_txt(CALIB_TXT)
    raw_test  = load_lines_from_txt(TEST_TXT)
else:
    from datasets import load_dataset
    raw_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
    raw_test  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]

train_split = filter_texts(raw_train, min_tokens=2, max_keep=CALIB_SAMPLES*3)
test_split  = filter_texts(raw_test,  min_tokens=2, max_keep=TEST_SAMPLES*3)

# ================== 编码与评估工具 ==================
def pack(text, max_len):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_len, padding=False, add_special_tokens=True)
    if enc["input_ids"].size(1) < 2:
        return None
    return {
        k: (v.to(DEVICE, dtype=torch.long) if k == "input_ids" else v.to(DEVICE))
        for k, v in enc.items()
    }

@torch.no_grad()
def evaluate_ppl(model, dataset_texts, max_len=128, samples=200):
    total_loss, total_tokens = 0.0, 0
    used, i = 0, 0
    while used < samples and i < len(dataset_texts):
        enc = pack(dataset_texts[i], max_len); i += 1
        if enc is None:
            continue
        labels = enc["input_ids"].clone()
        if labels.size(1) < 2:
            continue
        loss = model(**enc, labels=labels).loss
        tok = enc["input_ids"].size(1)
        total_loss  += float(loss.item()) * tok
        total_tokens += tok
        used += 1
    if total_tokens == 0:
        raise RuntimeError("评测集未产生有效样本，请检查过滤条件。")
    return float(np.exp(total_loss / total_tokens))

# ================== 校准：统计所有 (layer, head) 的 Top-K 列 ==================
@torch.no_grad()
def get_all_hub_columns(model, dataset_texts, k, sample_size=128, max_len=128):
    """
    返回 dict[(layer, head)] -> LongTensor[K]（在 DEVICE 上）。
    对每个 (layer, head) 累计列能量（batch+query 求和），变长 K 用“右扩容/右pad”对齐。
    """
    n = min(sample_size, len(dataset_texts))
    energy = [[None for _ in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]
    used, i = 0, 0
    while used < n and i < len(dataset_texts):
        enc = pack(dataset_texts[i], max_len); i += 1
        if enc is None:
            continue
        outputs = model(**enc)  # attentions: list(len=NUM_LAYERS) of [B,H,Q,K]
        for l, A_l in enumerate(outputs.attentions):
            if A_l.size(2) == 0:
                continue
            sum_BQ = A_l.sum(dim=0).sum(dim=1)  # [H, K]
            for h in range(sum_BQ.size(0)):
                vec = sum_BQ[h]                # [K]
                cur = energy[l][h]
                if cur is None:
                    energy[l][h] = vec
                else:
                    if vec.size(0) > cur.size(0):
                        pad = vec.size(0) - cur.size(0)
                        cur = torch.cat([cur, torch.zeros(pad, device=DEVICE, dtype=cur.dtype)], dim=0)
                        energy[l][h] = cur + vec
                    elif vec.size(0) < cur.size(0):
                        vec = F.pad(vec, (0, cur.size(0) - vec.size(0)))
                        energy[l][h] = cur + vec
                    else:
                        energy[l][h] = cur + vec
        used += 1

    hub_cols = {}
    for l in range(NUM_LAYERS):
        for h in range(NUM_HEADS):
            vec = energy[l][h]
            if vec is None or vec.numel() == 0:
                hub_cols[(l, h)] = torch.arange(k, device=DEVICE, dtype=torch.long)
            else:
                hub_cols[(l, h)] = torch.topk(vec, k).indices.to(DEVICE, dtype=torch.long)
    return hub_cols

# ================== 概率消融 + 打补丁/还原（支持 GQA） ==================
def ablate_probs_inplace(attn_probs_qk, cols):
    """
    attn_probs_qk: [Q, K]（单个 batch/head 的视图）
    cols: LongTensor（列索引，需同设备）
    """
    valid = cols[cols < attn_probs_qk.size(-1)]
    if valid.numel() == 0:
        return attn_probs_qk
    attn_probs_qk[..., valid] = 0.0
    row_sum = attn_probs_qk.sum(dim=-1, keepdim=True)
    zero_rows = row_sum <= 1e-12
    if zero_rows.any():
        eps = 1e-8
        attn_probs_qk = torch.where(zero_rows, eps * torch.ones_like(attn_probs_qk), attn_probs_qk)
        row_sum = attn_probs_qk.sum(dim=-1, keepdim=True)
    attn_probs_qk /= (row_sum + 1e-8)
    return attn_probs_qk

def make_surgical_forward(attn_module, head_to_cols, num_heads, num_kv_heads, head_dim):
    """
    手术版 self-attn.forward（支持 GQA）：
      - Q: [B,H,Q,D]
      - K,V: [B,H_kv,Q,D] -> repeat 到 [B,H,Q,D]
      - softmax 后按 head 做列消融（置零+逐行归一）
    只返回 (attn_output, attn_probs) 两个值，匹配 Qwen2 层期望。
    """
    orig_forward = attn_module.forward  # 备份

    def repeat_kv(x, n_rep: int):
        # x: [B,H_kv,Q,D] -> [B,H_kv*n_rep,Q,D]
        if n_rep == 1:
            return x
        b, h_kv, q, d = x.shape
        return x.unsqueeze(2).expand(b, h_kv, n_rep, q, d).reshape(b, h_kv * n_rep, q, d)

    def surgical_forward(self, *args, **kwargs):
        # ---- 兼容位置/关键字参数 ----
        if len(args) >= 1:
            hidden_states = args[0]
            rem_args = args[1:]
            kwargs.pop("hidden_states", None)
        else:
            hidden_states = kwargs.pop("hidden_states")
            rem_args = ()

        def pick(name, idx, default=None):
            if name in kwargs:
                return kwargs.pop(name)
            if len(rem_args) > idx:
                return rem_args[idx]
            return default

        attention_mask = pick("attention_mask", 0, None)
        position_ids   = pick("position_ids", 1, None)
        # past/use_cache/output_attentions 不使用，但保持兼容
        kwargs.pop("past_key_value", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("output_attentions", None)

        bsz, q_len, _ = hidden_states.size()

        # qkv 投影
        q = self.q_proj(hidden_states)                # [B,Q,H*D]
        k = self.k_proj(hidden_states)                # [B,Q,H_kv*D]
        v = self.v_proj(hidden_states)                # [B,Q,H_kv*D]

        # 变形
        q = q.view(bsz, q_len, num_heads,    head_dim).transpose(1, 2)  # [B,H,Q,D]
        k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [B,H_kv,Q,D]
        v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [B,H_kv,Q,D]

        # RoPE
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat K/V 到 H 个头
        n_rep = num_heads // num_kv_heads
        k = repeat_kv(k, n_rep)   # [B,H,Q,D]
        v = repeat_kv(v, n_rep)   # [B,H,Q,D]

        # 注意力权重 + 掩码
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [B,H,Q,K]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # softmax + dropout（兼容 float 或模块）
        attn_probs = torch.softmax(attn_weights, dim=-1)
        drop_layer_or_p = getattr(self, "attention_dropout", 0.0)
        if isinstance(drop_layer_or_p, (float, int)):
            attn_probs = F.dropout(attn_probs, p=float(drop_layer_or_p), training=self.training)
        else:
            attn_probs = drop_layer_or_p(attn_probs)

        # === 列消融（仅改 head_to_cols 指定的那些 head）===
        B, H, Q, Klen = attn_probs.shape
        for h, cols in head_to_cols.items():
            if h >= H or cols is None:
                continue
            cols = cols.to(attn_probs.device, dtype=torch.long)
            valid = cols[cols < Klen]
            if valid.numel() == 0:
                continue
            for b in range(B):
                A = attn_probs[b, h]            # [Q,K]
                A[..., valid] = 0.0
                row_sum = A.sum(dim=-1, keepdim=True)
                zero_rows = row_sum <= 1e-12
                if zero_rows.any():
                    eps = 1e-8
                    A = torch.where(zero_rows, eps * torch.ones_like(A), A)
                    row_sum = A.sum(dim=-1, keepdim=True)
                A /= (row_sum + 1e-8)
                attn_probs[b, h] = A

        # 与 V 相乘 -> 输出投影
        attn_output = torch.matmul(attn_probs, v)                      # [B,H,Q,D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_probs

    surgical_forward._orig = orig_forward
    return surgical_forward

def patch_one_head(model, layer_idx: int, head_idx: int, cols_tensor: torch.Tensor):
    """
    仅打补丁到 layer_idx 这层，并且只改 head_idx 这个头。
    其余层/头完全不变。
    """
    layer = model.model.layers[layer_idx]
    attn  = layer.self_attn
    mapping = {head_idx: cols_tensor}  # 只传一个 head 的列索引
    new_forward = types.MethodType(
        make_surgical_forward(attn, mapping, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM),
        attn
    )
    handle = (attn, attn.forward)
    attn.forward = new_forward
    return [handle]  # 与 unpatch 兼容

def unpatch(handles):
    for attn_module, orig_forward in handles:
        attn_module.forward = orig_forward

# ================== 随机列：从大空间采样并显式排除 hub ==================
def sample_random_cols_excluding(hubs: torch.Tensor, k: int) -> torch.Tensor:
    hubs_cpu = set(hubs.detach().to("cpu").tolist())
    pool = [i for i in range(MAX_POS) if i not in hubs_cpu]
    if len(pool) < k:
        choice = np.random.choice(pool, size=k, replace=True)
    else:
        choice = np.random.choice(pool, size=k, replace=False)
    return torch.tensor(choice, device=DEVICE, dtype=torch.long)

# ================== 1) 校准：拿全头 Top-K 枢纽列 ==================
print(f"[Info] Collecting hub columns for ALL heads... layers={NUM_LAYERS}, heads={NUM_HEADS}, K={K}")
all_hub_cols = get_all_hub_columns(
    model=model,
    dataset_texts=train_split,
    k=K,
    sample_size=CALIB_SAMPLES,
    max_len=MAX_TOKENS
)
print("[Info] Done collecting hub columns.")

# 取目标 head 的 hub / rand
hub_cols_target  = all_hub_cols[(TARGET_LAYER, TARGET_HEAD)]
rand_cols_target = sample_random_cols_excluding(hub_cols_target, K)

# ================== 2) 三次评估：Baseline / Random(1 head) / Hub(1 head) ==================
PPL_baseline = evaluate_ppl(model, test_split, max_len=MAX_TOKENS, samples=TEST_SAMPLES)
print(f"[L{TARGET_LAYER} H{TARGET_HEAD}] PPL_baseline: {PPL_baseline:.4f}")

rand_handle = patch_one_head(model, TARGET_LAYER, TARGET_HEAD, rand_cols_target)
PPL_rand = evaluate_ppl(model, test_split, max_len=MAX_TOKENS, samples=TEST_SAMPLES)
unpatch(rand_handle)
print(f"[L{TARGET_LAYER} H{TARGET_HEAD}] PPL_random:  {PPL_rand:.4f}  (Δ {100*(PPL_rand-PPL_baseline)/PPL_baseline:.2f}%)")

hub_handle = patch_one_head(model, TARGET_LAYER, TARGET_HEAD, hub_cols_target)
PPL_hub = evaluate_ppl(model, test_split, max_len=MAX_TOKENS, samples=TEST_SAMPLES)
unpatch(hub_handle)
print(f"[L{TARGET_LAYER} H{TARGET_HEAD}] PPL_hub:     {PPL_hub:.4f}  (Δ {100*(PPL_hub-PPL_baseline)/PPL_baseline:.2f}%)")

# ================== 3) 结果表 ==================
print("\n结果表（单个 head）：")
print(f"{'类型':<18}{'PPL':<15}{'相对增幅(%)':<15}")
print(f"{'Baseline':<18}{PPL_baseline:<15.4f}{0.0:<15.2f}")
print(f"{'Random (1 head)':<18}{PPL_rand:<15.4f}{100*(PPL_rand-PPL_baseline)/PPL_baseline:<15.2f}")
print(f"{'Hub (1 head)':<18}{PPL_hub:<15.4f}{100*(PPL_hub-PPL_baseline)/PPL_baseline:<15.2f}")

# ================== （可选）扫该层所有 head：逐个 head 单独消融 ==================
# 取消注释即可运行（可能较慢）
print(f"\n[Info] Sweep one-by-one per head in layer {TARGET_LAYER} ...")
baseline = PPL_baseline
rows = []
for h in range(NUM_HEADS):
    hubs = all_hub_cols[(TARGET_LAYER, h)]
    rands = sample_random_cols_excluding(hubs, K)

    hdl = patch_one_head(model, TARGET_LAYER, h, rands)
    ppl_r = evaluate_ppl(model, test_split, max_len=MAX_TOKENS, samples=TEST_SAMPLES)
    unpatch(hdl)

    hdl = patch_one_head(model, TARGET_LAYER, h, hubs)
    ppl_h = evaluate_ppl(model, test_split, max_len=MAX_TOKENS, samples=TEST_SAMPLES)
    unpatch(hdl)

    rows.append((h, float(ppl_r), float(ppl_h),
                 100*(ppl_r-baseline)/baseline, 100*(ppl_h-baseline)/baseline))
    print(f"[L{TARGET_LAYER:02d} H{h:02d}] "
          f"PPL_rand={ppl_r:.4f} (Δ{100*(ppl_r-baseline)/baseline:.2f}%), "
          f"PPL_hub={ppl_h:.4f} (Δ{100*(ppl_h-baseline)/baseline:.2f}%)")

rows = sorted(rows, key=lambda x: x[4], reverse=True)
print("\n==== 单层逐头结果（按Hub增幅降序）====")
print(f"{'Head':<6}{'PPL_rand':<12}{'Δrand%':<10}{'PPL_hub':<12}{'Δhub%':<10}")
for h, ppl_r, ppl_h, inc_r, inc_h in rows:
    print(f"{h:<6}{ppl_r:<12.4f}{inc_r:<10.2f}{ppl_h:<12.4f}{inc_h:<10.2f}")
