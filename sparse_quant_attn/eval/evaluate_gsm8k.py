import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# from .sageattn_wrapper import QwenSageAttnForward
from types import MethodType
from loguru import logger
from tqdm import tqdm


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def doc_to_text(doc, fewshot_prompt):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer(input_txt)['input_ids']
    raw_text_len = len(input_ids)
    logger.info(f"raw_text_len: {raw_text_len}")
    context_enc = torch.tensor([input_ids]).to(model.device)
    # print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    # print(f"\nOutput text: {output_text}\n")
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold



def evaluate_gsm8k(model, tokenizer, args):
    fewshot_prompt = open("/mnt/disk3/wzn/SQAttn/sparse_quant_attn/eval/gsm8k_prompt.txt").read()
    
    config = datasets.DownloadConfig(resume_download=True, max_retries=100)
    dataset = load_dataset("gsm8k", "main", download_config=config)
    
    test = dataset["test"].select(range(50))
    # test = dataset["test"]

    # sample_output_file = "gsm8k_res.jsonl"
    sample_output_file = args.sample_output_file
    # print("Loading tokenizer ...")
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.checkpoint_path, trust_remote_code=True
    # )

    # print("Loading model ...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.checkpoint_path, device_map="auto", trust_remote_code=True
    # ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.model, trust_remote_code=True
    )
    model.generation_config.do_sample = False
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    f_output = jsonlines.Writer(open(sample_output_file, "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []
    for doc in tqdm(test):
        context = doc_to_text(doc, fewshot_prompt)
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print("Acc: ", np.mean(acc_res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="/mnt/disk3/hg/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
        # default="Qwen/Qwen2.5-Math-7B"
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )

    args = parser.parse_args()

    fewshot_prompt = open("gsm8k_prompt.txt").read()
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    # test = dataset["test"].select(range(50))
    test = dataset["test"]


    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print("Loading model ...")
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="auto", trust_remote_code=True, **kwargs
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False
    # for layer in model.model.layers:
    #     layer.self_attn.forward = MethodType(QwenSageAttnForward, layer.self_attn)
    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
    tot_length = test.num_rows
    acc_res = []
    for doc in tqdm(test):
        context = doc_to_text(doc, fewshot_prompt)
        completion = generate_sample(model, tokenizer, context)
        answer = doc["answer"]
        acc = is_correct(completion, answer)
        doc["completion"] = completion
        doc["acc"] = acc
        f_output.write(doc)
        acc_res.append(acc)

    f_output.close()
    print("Acc: ", np.mean(acc_res))




