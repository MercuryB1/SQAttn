import torch
import datasets
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, seq_len=512, device="cuda"):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == "gsm8k":
        return get_calib_dataset_gsm8k(tokenizer, device)
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    samples = torch.cat(samples, dim=1)
    n_split = samples.shape[1] // seq_len
    samples = [samples[:, i * seq_len: (i + 1) * seq_len] for i in range(n_split)]
    samples = torch.cat(samples, dim=0)
    samples = samples[0:1]
    return samples, None

def doc_to_text(doc, fewshot_prompt):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )

def get_calib_dataset_gsm8k(tokenizer=None, device="cuda"):
    fewshot_prompt = open("/mnt/disk3/wzn/SQAttn/sparse_quant_attn/eval/gsm8k_prompt.txt").read()
    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = load_dataset("gsm8k", "main", download_config=config)
    dataset = dataset["train"].select(range(10))
    texts = []
    for doc in dataset:
        context = doc_to_text(doc, fewshot_prompt)
        texts.append(context)
    tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,           
        truncation=True        
    )
    input_ids = encodings["input_ids"].to(device)           # shape (B, L_max)
    attention_mask = encodings["attention_mask"].to(device)     

    return input_ids, attention_mask