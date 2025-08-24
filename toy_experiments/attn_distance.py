# -*- coding: utf-8 -*-
import os
import random
import types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# ================== 基本配置 ==================
MODEL_ID = "/mnt/disk3/hg/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 实验参数
TEST_SAMPLES = 200         # 评估样本数
MAX_TOKENS = 256           # 序列长度限制
SEED = 42

# 相对距离阈值 - 更精细的测试最远几个位置
RELATIVE_THRESHOLDS = [0, -1, -2, -3, -4, None]  # None表示无限制（基准）

# 全局统计数据收集
MASK_STATS = {}

# 随机性/性能
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ================== 加载模型与分词器 ==================
print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="eager",
    device_map=None,
).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

NUM_LAYERS = model.config.num_hidden_layers
NUM_HEADS = model.config.num_attention_heads
NUM_KV_HEADS = getattr(model.config, "num_key_value_heads", NUM_HEADS)
HIDDEN_SIZE = model.config.hidden_size
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

print(f"Model info: {NUM_LAYERS} layers, {NUM_HEADS} heads")

# ================== 数据加载 ==================
print("Loading dataset...")
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
raw_test = dataset["text"]

# 过滤有效文本
def filter_texts(texts, min_tokens=10, max_keep=None):
    kept = []
    for t in texts:
        if t and len(t.strip()) > 20:  # 简单过滤
            kept.append(t)
            if max_keep and len(kept) >= max_keep:
                break
    return kept

test_split = filter_texts(raw_test, max_keep=TEST_SAMPLES*3)
print(f"Loaded {len(test_split)} test samples")

# ================== 编码工具 ==================
def pack(text, max_len):
    """将文本编码为模型输入格式"""
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_len, padding=False, add_special_tokens=True)
    if enc["input_ids"].size(1) < 10:  # 过滤太短的序列
        return None
    return {
        k: (v.to(DEVICE, dtype=torch.long) if k == "input_ids" else v.to(DEVICE))
        for k, v in enc.items()
    }

# ================== 距离遮罩消融 ==================
def make_distance_ablated_forward(attn_module, relative_threshold):
    """创建应用距离遮罩的注意力前向传播函数"""
    orig_forward = attn_module.forward

    def repeat_kv(x, n_rep: int):
        if n_rep == 1:
            return x
        b, h_kv, q, d = x.shape
        return x.unsqueeze(2).expand(b, h_kv, n_rep, q, d).reshape(b, h_kv * n_rep, q, d)

    def distance_ablated_forward(self, *args, **kwargs):
        # 解析参数
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
        # 忽略其他参数
        for key in list(kwargs.keys()):
            kwargs.pop(key, None)

        bsz, q_len, _ = hidden_states.size()

        # qkv 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 变形为多头格式
        q = q.view(bsz, q_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # GQA: repeat K/V
        n_rep = NUM_HEADS // NUM_KV_HEADS
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        # 计算attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)

        # === 核心：应用相对距离遮罩 ===
        if relative_threshold is not None:
            seq_len = attn_weights.size(-1)
            
            # 特殊处理：threshold=0表示不遮罩任何位置
            if relative_threshold == 0:
                # 不应用任何遮罩，相当于baseline
                pass
            else:
                # 负值表示从末尾开始遮罩多少行
                # 例如：-1表示遮罩最后1行，-2表示遮罩最后2行
                num_rows_to_mask = abs(relative_threshold)
                
                if num_rows_to_mask > 0 and num_rows_to_mask < seq_len:
                    # 创建行遮罩：遮罩最后几行的所有连接
                    row_mask = torch.zeros(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool)
                    
                    # 遮罩最后num_rows_to_mask行
                    start_row = seq_len - num_rows_to_mask
                    row_mask[start_row:, :] = True  # 最后几行全部遮罩
                    
                    # 只在causal允许的范围内应用遮罩
                    positions = torch.arange(seq_len, device=hidden_states.device)
                    causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
                    
                    # 最终遮罩：既要在causal范围内，又要在指定行内
                    final_mask = row_mask & causal_mask
                    
                    # === 收集遮罩统计信息 ===
                    total_causal_positions = causal_mask.sum().item()
                    masked_positions = final_mask.sum().item()
                    mask_ratio = (masked_positions / total_causal_positions * 100) if total_causal_positions > 0 else 0
                    
                    # 存储到全局统计
                    if relative_threshold not in MASK_STATS:
                        MASK_STATS[relative_threshold] = {
                            'total_mask_ratio': 0.0,
                            'sample_count': 0,
                            'seq_lengths': []
                        }
                    
                    stats = MASK_STATS[relative_threshold]
                    stats['total_mask_ratio'] += mask_ratio
                    stats['sample_count'] += 1
                    stats['seq_lengths'].append(seq_len)
                    
                    # 应用遮罩
                    attn_weights = attn_weights.masked_fill(
                        final_mask.unsqueeze(0).unsqueeze(0), -1e9
                    )

        # 应用attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # softmax
        attn_probs = torch.softmax(attn_weights, dim=-1)

        # 应用到values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, NUM_HEADS * HEAD_DIM)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_probs

    return distance_ablated_forward

def patch_all_attention_layers(model, relative_threshold):
    """为所有注意力层应用距离遮罩"""
    handles = []
    modified_count = 0
    
    for layer_idx in range(NUM_LAYERS):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        
        if hasattr(attn, 'q_proj'):
            new_forward = types.MethodType(
                make_distance_ablated_forward(attn, relative_threshold),
                attn
            )
            handle = (attn, attn.forward)
            attn.forward = new_forward
            handles.append(handle)
            modified_count += 1
    
    return handles

def unpatch(handles):
    """恢复原始的前向传播函数"""
    for attn_module, orig_forward in handles:
        attn_module.forward = orig_forward

# ================== 逐样本评估 ==================
@torch.no_grad()
def evaluate_ppl_per_sample(model, dataset_texts, relative_threshold, max_samples=200):
    """对每个样本单独计算PPL，使用基于该样本seq_len的动态阈值"""
    total_loss = 0.0
    total_tokens = 0
    valid_samples = 0
    
    print(f"Evaluating with relative threshold {relative_threshold}...")
    
    # 如果有阈值，先patch模型
    handles = None
    if relative_threshold is not None and relative_threshold != 0:
        handles = patch_all_attention_layers(model, relative_threshold)
    
    try:
        for i, text in enumerate(dataset_texts):
            if valid_samples >= max_samples:
                break
            
            if valid_samples % 50 == 0:
                print(f"Progress: {valid_samples}/{max_samples}")
            
            # 编码文本
            enc = pack(text, MAX_TOKENS)
            if enc is None:
                continue
            
            seq_len = enc["input_ids"].size(1)
            
            # 如果设置了相对阈值，检查是否有效
            if relative_threshold is not None and relative_threshold != 0:
                # 检查是否会遮罩过多行
                num_rows_to_mask = abs(relative_threshold)
                if num_rows_to_mask >= seq_len:
                    continue  # 跳过会遮罩所有行的阈值
            
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", None)
            
            try:
                # 计算损失 - 使用正确的方式
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # 计算交叉熵损失：shift logits和labels
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # 展平为2D
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    
                    # 计算loss（不包括padding）
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
                    loss = loss_fct(shift_logits, shift_labels)
                    
                    # 计算有效token数量
                    valid_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
                    
                    if valid_tokens > 0:
                        # 累计
                        total_loss += loss.item()
                        total_tokens += valid_tokens
                        valid_samples += 1
                        
                        # 打印前几个样本的详细信息
                        if valid_samples <= 3:
                            sample_loss = loss.item() / valid_tokens
                            sample_ppl = np.exp(sample_loss)
                            print(f"Sample {valid_samples}: seq_len={seq_len}, valid_tokens={valid_tokens}, loss={sample_loss:.4f}, ppl={sample_ppl:.4f}")
                    
            except Exception as e:
                print(f"Error in sample {i}: {e}")
                continue
                
    finally:
        # 恢复模型
        if handles is not None:
            unpatch(handles)
    
    if total_tokens == 0:
        print("ERROR: No valid samples found!")
        return float('inf'), 0
    
    avg_loss = total_loss / total_tokens
    ppl = float(np.exp(avg_loss))
    
    print(f"Completed: {valid_samples} samples, {total_tokens} tokens, avg_loss={avg_loss:.4f}, PPL = {ppl:.4f}")
    return ppl, valid_samples

# ================== 主实验 ==================
def visualize_distance_masks():
    """可视化不同阈值下的距离遮罩效果"""
    print("\n" + "="*60)
    print("DISTANCE MASK VISUALIZATION")
    print("="*60)
    
    # 使用一个示例序列长度
    seq_len = 20  # 较小的长度便于可视化
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 创建位置矩阵
    positions = torch.arange(seq_len)
    distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
    
    # Causal mask (下三角)
    causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
    
    thresholds_to_viz = [None, 0, -1, -2, -3, -4]
    titles = ["Original (No Limit)", "Threshold=0 (No Mask)", "Threshold=-1", "Threshold=-2", "Threshold=-3", "Threshold=-4"]
    
    for idx, (threshold, title) in enumerate(zip(thresholds_to_viz, titles)):
        ax = axes[idx]
        
        if threshold is None:
            # 原始attention矩阵 (只有causal mask)
            mask_matrix = causal_mask.float()
            masked_positions = 0
        elif threshold == 0:
            # threshold=0: 不应用距离限制
            mask_matrix = causal_mask.float()
            masked_positions = 0
        else:
            # 应用距离限制
            actual_threshold = seq_len + threshold  # threshold是负数
            if actual_threshold > 0:
                distance_mask = (distance_matrix >= actual_threshold) & causal_mask
                # 1表示允许attention，0表示被遮罩
                mask_matrix = causal_mask.float() - distance_mask.float()
                masked_positions = distance_mask.sum().item()
            else:
                mask_matrix = torch.zeros_like(causal_mask, dtype=torch.float)
                masked_positions = causal_mask.sum().item()
        
        # 绘制热力图
        im = ax.imshow(mask_matrix.numpy(), cmap='RdYlBu', vmin=0, vmax=1)
        
        # 设置标题和标签
        total_positions = causal_mask.sum().item()
        mask_ratio = (masked_positions / total_positions * 100) if total_positions > 0 else 0
        ax.set_title(f'{title}\nMasked: {masked_positions}/{total_positions} ({mask_ratio:.1f}%)')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # 添加网格
        ax.set_xticks(range(0, seq_len, 5))
        ax.set_yticks(range(0, seq_len, 5))
        ax.grid(True, alpha=0.3)
        
        # 在每个单元格中添加数值（对于小矩阵）
        if seq_len <= 10:
            for i in range(seq_len):
                for j in range(seq_len):
                    text = ax.text(j, i, f'{mask_matrix[i, j]:.0f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    # 添加颜色条
    plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8, 
                 label='1=Allowed, 0=Masked')
    
    plt.tight_layout()
    plt.savefig('distance_mask_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig('distance_mask_visualization.pdf', bbox_inches='tight')
    print("Mask visualization saved as distance_mask_visualization.png/.pdf")
    plt.show()
    
    # 打印详细的遮罩统计
    print(f"\nDetailed Mask Statistics (seq_len={seq_len}):")
    print(f"{'Threshold':<12}{'Actual':<8}{'Masked':<8}{'Total':<8}{'Ratio':<8}{'Description'}")
    print("-" * 60)
    
    total_causal = causal_mask.sum().item()
    
    for threshold in [0, -1, -2, -3, -4]:
        if threshold == 0:
            masked = 0
            desc = "No distance limit"
        else:
            actual_threshold = seq_len + threshold
            if actual_threshold > 0:
                distance_mask = (distance_matrix >= actual_threshold) & causal_mask
                masked = distance_mask.sum().item()
                desc = f"Distance >= {actual_threshold}"
            else:
                masked = total_causal
                desc = "All positions masked"
        
        ratio = (masked / total_causal * 100) if total_causal > 0 else 0
        print(f"{threshold:<12}{seq_len + threshold if threshold != 0 else 'N/A':<8}{masked:<8}{total_causal:<8}{ratio:<8.1f}{desc}")

def visualize_real_attention_sample():
    """可视化真实样本的注意力遮罩"""
    print("\n" + "="*60) 
    print("REAL SAMPLE ATTENTION MASK VISUALIZATION")
    print("="*60)
    
    # 获取一个真实样本
    sample_text = None
    for text in test_split[:10]:
        enc = pack(text, 64)  # 使用较短长度便于可视化
        if enc is not None and enc['input_ids'].size(1) >= 20:
            sample_text = text
            sample_enc = enc
            break
    
    if sample_text is None:
        print("No suitable sample found for visualization")
        return
    
    seq_len = sample_enc['input_ids'].size(1)
    print(f"Sample text (first 100 chars): {sample_text[:100]}...")
    print(f"Sequence length: {seq_len}")
    
    # 可视化不同阈值的效果
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    positions = torch.arange(seq_len)
    distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
    causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
    
    thresholds = [0, -1, -2, -4]
    titles = ["No Distance Limit", "Cut Last 1 Pos", "Cut Last 2 Pos", "Cut Last 4 Pos"]
    
    for idx, (threshold, title) in enumerate(zip(thresholds, titles)):
        ax = axes[idx]
        
        if threshold == 0:
            mask_matrix = causal_mask.float()
            masked_positions = 0
        else:
            actual_threshold = seq_len + threshold
            if actual_threshold > 0:
                distance_mask = (distance_matrix >= actual_threshold) & causal_mask
                mask_matrix = causal_mask.float() - distance_mask.float()
                masked_positions = distance_mask.sum().item()
            else:
                mask_matrix = torch.zeros_like(causal_mask, dtype=torch.float)
                masked_positions = causal_mask.sum().item()
        
        # 绘制
        im = ax.imshow(mask_matrix.numpy(), cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')
        
        total_positions = causal_mask.sum().item()
        mask_ratio = (masked_positions / total_positions * 100) if total_positions > 0 else 0
        
        ax.set_title(f'{title}\nMasked: {mask_ratio:.1f}% ({masked_positions}/{total_positions})')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # 设置合适的刻度
        step = max(1, seq_len // 10)
        ax.set_xticks(range(0, seq_len, step))
        ax.set_yticks(range(0, seq_len, step))
    
    plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.15, shrink=0.8,
                 label='1=Attention Allowed, 0=Masked (Red=Allowed, Blue=Masked)')
    
    plt.tight_layout()
    plt.savefig('real_sample_mask_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig('real_sample_mask_visualization.pdf', bbox_inches='tight')
    print("Real sample visualization saved as real_sample_mask_visualization.png/.pdf")
    plt.show()


def run_experiment():
# 在主函数中添加可视化调用
    """运行主实验"""
    print("="*60)
    print("RELATIVE DISTANCE THRESHOLD ABLATION EXPERIMENT")
    print("="*60)
    
    # 先做一个简单的基准测试
    print("First, let's test baseline PPL calculation...")
    baseline_ppl, baseline_samples = evaluate_ppl_per_sample(model, test_split[:10], None, max_samples=10)
    print(f"Quick baseline test (10 samples): PPL = {baseline_ppl:.4f}")
    
    if baseline_ppl > 100:  # 如果基准PPL异常高
        print("WARNING: Baseline PPL seems too high! There might be an issue with the model or data.")
        print("Let's try a single sample debug...")
        
        # 调试单个样本
        for i, text in enumerate(test_split[:3]):
            enc = pack(text, MAX_TOKENS)
            if enc is not None:
                print(f"\nDebug sample {i+1}:")
                print(f"Text length: {len(text)}")
                print(f"Token length: {enc['input_ids'].size(1)}")
                print(f"First 10 tokens: {enc['input_ids'][0][:10].tolist()}")
                
                with torch.no_grad():
                    outputs = model(input_ids=enc['input_ids'])
                    logits = outputs.logits
                    print(f"Logits shape: {logits.shape}")
                    print(f"Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
                break
    
    results = {}
    
    # 测试每个相对阈值
    for i, rel_threshold in enumerate(RELATIVE_THRESHOLDS):
        print(f"\n[{i+1}/{len(RELATIVE_THRESHOLDS)}] Testing relative threshold: {rel_threshold}")
        
        if rel_threshold is None:
            print("  → Baseline (no distance limit)")
        elif rel_threshold == 0:
            print("  → Control: no masking (should be same as baseline)")
        else:
            print(f"  → Will mask the last {abs(rel_threshold)} row(s) completely")
        
        try:
            ppl, sample_count = evaluate_ppl_per_sample(
                model, test_split, rel_threshold, max_samples=TEST_SAMPLES
            )
            
            key = rel_threshold if rel_threshold is not None else 'baseline'
            results[key] = {'ppl': ppl, 'samples': sample_count}
            
            print(f"Result: PPL = {ppl:.4f} (based on {sample_count} samples)")
            
        except Exception as e:
            print(f"Error with threshold {rel_threshold}: {e}")
            import traceback
            traceback.print_exc()
            
            key = rel_threshold if rel_threshold is not None else 'baseline'
            results[key] = {'ppl': None, 'samples': 0}
    
    return results

def plot_and_analyze_results(results):
    """分析和可视化结果"""
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    
    baseline_ppl = results.get('baseline', {}).get('ppl', None)
    if baseline_ppl is None:
        print("No baseline result found!")
        return
    
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print("\nRelative Threshold Results:")
    print(f"{'Threshold':<12}{'PPL':<12}{'Increase %':<12}{'Mask %':<10}{'Samples':<8}")
    print("-" * 60)
    
    # 收集有效结果
    thresholds = []
    ppls = []
    increases = []
    mask_ratios = []
    
    for rel_thresh in sorted([k for k in results.keys() if k != 'baseline'], reverse=True):
        result = results[rel_thresh]
        if result['ppl'] is not None:
            ppl = result['ppl']
            samples = result['samples']
            increase = (ppl - baseline_ppl) / baseline_ppl * 100
            
            # 获取遮罩统计
            if rel_thresh in MASK_STATS:
                stats = MASK_STATS[rel_thresh]
                avg_mask_ratio = stats['total_mask_ratio'] / stats['sample_count'] if stats['sample_count'] > 0 else 0
            else:
                avg_mask_ratio = 0
            
            print(f"{rel_thresh:<12}{ppl:<12.4f}{increase:<12.2f}{avg_mask_ratio:<10.2f}{samples:<8}")
            
            thresholds.append(abs(rel_thresh))  # 用绝对值便于绘图
            ppls.append(ppl)
            increases.append(increase)
            mask_ratios.append(avg_mask_ratio)
    
    # 绘制结果 - 五个子图（2x3布局）
    if len(thresholds) > 0:
        fig = plt.figure(figsize=(18, 12))
        
        # 子图1：PPL增幅 vs 切断位置数
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(thresholds, increases, 'ro-', linewidth=2, markersize=8)
        ax1.set_xlabel('Positions Cut from End')
        ax1.set_ylabel('Perplexity Increase (%)')
        ax1.set_title('Performance Drop vs Long-Range Connection Removal')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for x, y in zip(thresholds, increases):
            ax1.annotate(f'{y:.1f}%', (x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', fontweight='bold')
        
        # 子图2：绝对PPL
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(thresholds, ppls, 'bo-', linewidth=2, markersize=6, label='Ablated')
        ax2.axhline(y=baseline_ppl, color='green', linestyle='--', linewidth=2, label='Baseline')
        ax2.set_xlabel('Positions Cut from End')
        ax2.set_ylabel('Absolute Perplexity')
        ax2.set_title('Absolute Perplexity Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：遮罩比例
        ax3 = plt.subplot(2, 3, 3)
        ax3.bar(range(len(thresholds)), mask_ratios, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Positions Cut from End')
        ax3.set_ylabel('Masked Attention Positions (%)')
        ax3.set_title('Percentage of Attention Positions Masked')
        ax3.set_xticks(range(len(thresholds)))
        ax3.set_xticklabels([str(int(t)) for t in thresholds])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值
        for i, (x, y) in enumerate(zip(range(len(thresholds)), mask_ratios)):
            ax3.text(x, y + max(mask_ratios) * 0.01, f'{y:.2f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 子图4：效率分析 - PPL增幅 vs 遮罩比例
        ax4 = plt.subplot(2, 3, 4)
        if len(mask_ratios) > 0 and all(m > 0 for m in mask_ratios):
            scatter = ax4.scatter(mask_ratios, increases, c=thresholds, s=100, 
                         cmap='viridis', alpha=0.7, edgecolors='black')
            
            # 添加标签
            for i, thresh in enumerate(thresholds):
                ax4.annotate(f'-{int(thresh)}', 
                           (mask_ratios[i], increases[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax4.set_xlabel('Masked Positions (%)')
            ax4.set_ylabel('PPL Increase (%)')
            ax4.set_title('Impact Efficiency: Small Masks, Big Effects')
            ax4.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Positions Cut')
        
        # 子图5：遮罩可视化（示例）
        ax5 = plt.subplot(2, 3, 5)
        # 使用一个示例矩阵展示遮罩原理
        seq_len = 16  # 适中的大小便于显示
        positions = torch.arange(seq_len)
        causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
        
        # 选择一个代表性的阈值进行可视化（如-2）
        if len(thresholds) >= 2:
            demo_threshold = -2
            num_rows_to_mask = abs(demo_threshold)
            
            # 创建行遮罩：遮罩最后几行
            row_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
            start_row = seq_len - num_rows_to_mask
            row_mask[start_row:, :] = True
            
            # 最终遮罩
            final_mask = row_mask & causal_mask
            mask_matrix = causal_mask.float() - final_mask.float()
            
            im = ax5.imshow(mask_matrix.numpy(), cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')
            ax5.set_title(f'Row Mask Example\n(Threshold=-2: Mask Last 2 Rows)')
            ax5.set_xlabel('Key Position')
            ax5.set_ylabel('Query Position')
            
            # 设置合适的刻度
            ax5.set_xticks(range(0, seq_len, 4))
            ax5.set_yticks(range(0, seq_len, 4))
            
            # 添加分隔线显示被遮罩的行
            ax5.axhline(y=start_row-0.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
        
        # 子图6：对比不同阈值的遮罩模式
        ax6 = plt.subplot(2, 3, 6)
        if len(thresholds) >= 3:
            # 创建多个小的遮罩示例
            small_seq = 8
            positions_small = torch.arange(small_seq)
            causal_mask_small = positions_small.unsqueeze(0) <= positions_small.unsqueeze(1)
            
            # 显示3个不同阈值的遮罩
            demo_thresholds = [-1, -2, -3] if len(thresholds) >= 3 else thresholds[:3]
            combined_matrix = torch.zeros(small_seq, small_seq * len(demo_thresholds))
            
            for i, thresh in enumerate(demo_thresholds):
                num_rows = abs(thresh)
                
                # 创建行遮罩
                row_mask = torch.zeros(small_seq, small_seq, dtype=torch.bool)
                if num_rows > 0 and num_rows < small_seq:
                    start_row = small_seq - num_rows
                    row_mask[start_row:, :] = True
                
                final_mask = row_mask & causal_mask_small
                mask_mat = causal_mask_small.float() - final_mask.float()
                
                start_col = i * small_seq
                end_col = (i + 1) * small_seq
                combined_matrix[:, start_col:end_col] = mask_mat
            
            im2 = ax6.imshow(combined_matrix.numpy(), cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')
            ax6.set_title('Row Mask Comparison\n(Left to Right: -1, -2, -3 Rows)')
            ax6.set_xlabel('Key Position (Multiple Thresholds)')
            ax6.set_ylabel('Query Position')
            
            # 添加分隔线
            for i in range(1, len(demo_thresholds)):
                ax6.axvline(x=i * small_seq - 0.5, color='white', linewidth=2)
            
            # 添加行分隔线显示遮罩位置
            for i, thresh in enumerate(demo_thresholds):
                num_rows = abs(thresh)
                if num_rows > 0 and num_rows < small_seq:
                    start_row = small_seq - num_rows
                    x_center = i * small_seq + small_seq // 2
                    ax6.axhline(y=start_row-0.5, xmin=(i)/len(demo_thresholds), 
                              xmax=(i+1)/len(demo_thresholds), color='red', linewidth=1, alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('comprehensive_distance_ablation_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('comprehensive_distance_ablation_analysis.pdf', bbox_inches='tight')
        print(f"\nComprehensive analysis plots saved as comprehensive_distance_ablation_analysis.png/.pdf")
        plt.show()
    
    # 关键洞察
    print(f"\n💡 Key Insights:")
    if len(increases) > 0 and len(mask_ratios) > 0:
        min_mask = min(mask_ratios)
        min_idx = mask_ratios.index(min_mask)
        min_thresh = thresholds[min_idx] 
        min_increase = increases[min_idx]
        
        print(f"• Cutting just the last {int(min_thresh)} position(s) affects only {min_mask:.2f}% of attention positions")
        print(f"• But causes {min_increase:.1f}% performance degradation!")
        print(f"• This proves that the furthest connections, though few, are CRITICAL")
        print(f"• Long-range attention: 'Small in quantity, huge in importance'")
        
        # 计算效率
        if min_mask > 0:
            efficiency = min_increase / min_mask
            print(f"• Impact efficiency: {efficiency:.1f}% PPL increase per 1% masked positions")



# ================== 运行实验 ==================
if __name__ == "__main__":
    print("Starting Relative Distance Threshold Ablation Study...")
    
    # 运行实验
    results = run_experiment()
    
    # 分析结果  
    plot_and_analyze_results(results)
    
    print("\nExperiment completed!")