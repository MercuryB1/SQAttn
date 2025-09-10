import torch
import datasets
from datasets import load_dataset
import json
import pickle
from sparse_quant_attn.eval.longbench.pred import build_chat, post_process


def get_calib_dataset(data="pileval", model=None, tokenizer=None, n_samples=512, seq_len=512, device="cuda", args=None):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == "gsm8k":
        return get_calib_dataset_gsm8k(tokenizer, device, args.gsm8k_prompt)
    elif data == 'longbench':
        return get_calib_dataset_longbench(model, tokenizer, device, True)
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

def get_calib_dataset_gsm8k(tokenizer=None, device="cuda", gsm8k_prompt=None):
    fewshot_prompt = open(gsm8k_prompt).read()
    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = load_dataset("gsm8k", "main", download_config=config)
    dataset = dataset["train"].select(range(1))
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

def get_calib_dataset_longbench(model=None, tokenizer=None, device="cuda", e=False, static=True):
    if static:
        with open("longbench_samples.pkl", "rb") as f:
            samples = []
            loaded_data = pickle.load(f)
            for data in loaded_data:
                tokenized_data = tokenizer(data, truncation=False, return_tensors="pt")
                samples.append(tokenized_data['input_ids'])
            # import pdb; pdb.set_trace()
            return samples, None
    max_length = model.config.max_position_embeddings - 500
    if e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        # datasets = ["qasper"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    dataset2prompt = json.load(open("sparse_quant_attn/eval/longbench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("sparse_quant_attn/eval/longbench/config/dataset2maxlen.json", "r"))

    samples = []
    model.cuda()
    # dataset = "qasper"
    for dataset in datasets:
        if e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
        # data = data.shuffle(seed=42)
        data = data.select(range(1))
        data = [data_sample for data_sample in data][0]
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        prompt = prompt_format.format(**data)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensor='pt').input_ids
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, None)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        # samples.append(input)
        
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0].detach()
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0].detach()
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        # import pdb; pdb.set_trace()
        samples.append(prompt + pred)
        # pred = post_process(pred, model_name)
    with open("longbench_samples.pkl", "wb") as f:
        pickle.dump(samples, f)
    model.cpu()
    return samples, None