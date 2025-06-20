import random
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from loguru import logger
from sparse_quant_attn.utils.eval_utils import evaluate 
from sparse_quant_attn.compression.entrance import compress_model
from sparse_quant_attn.plot.window_size_alloc import plot_window_size_alloc


def seed_everything(seed: int):
    random.seed(seed)  # Python built-in random module
    np.random.seed(seed)  # NumPy

    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed (for consistent hashing)

    # PyTorch
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU (single card)
    torch.cuda.manual_seed_all(seed)  # GPU (multiple cards)

    # Ensure deterministic behavior in PyTorch (if necessary)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_and_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.use_cache = False
    enc = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True
    )
    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, trust_remote_code=True, **kwargs
    )
    model.eval()
    return model, enc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name or model path")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--calib_dataset",type=str,default="pileval",
        choices=["wikitext2", "ptb", "c4", "mix","pileval", "gsm8k"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seqlen", type=int, default=2048, help="seqlen.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", help="eval wikitext2 ppl")
    parser.add_argument("--eval_gsm8k", action="store_true", help="eval gsm8k")
    parser.add_argument("--multigpu", action="store_true", help="use multigpu for eval")
    parser.add_argument("--tasks", default=None, type=str)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--dynamic_shape", action="store_true", help="use dynamic shape for flashattn")
    parser.add_argument("--quant", action="store_true", help="use causal mask for flashattn")
    parser.add_argument("--qk_qtype", type=str, default="int", choices=["int", "e4m3", "e5m2"], help="quantization type for qk")
    parser.add_argument("--v_qtype", type=str, default="int", choices=["int", "e4m3", "e5m2"], help="quantization type for v")
    parser.add_argument("--bit8_thres_cos", type=float, default=0.9999, help="threshold for cosine similarity for bit8")
    parser.add_argument("--bit8_thres_rmse", type=float, default=0.01, help="threshold for rmse for bit8")
    parser.add_argument("--bit4_thres_cos", type=float, default=0.9999, help="threshold for cosine similarity for bit4")
    parser.add_argument("--bit4_thres_rmse", type=float, default=0.01, help="threshold for rmse for bit4")
    parser.add_argument("--sample_output_file", type=str, default="gsm8k_res.jsonl", help="file for saving sample output")
    parser.add_argument("--vis_attn", action="store_true", help="visualize attention")
    parser.add_argument("--plot_window_size_alloc", action="store_true", help="plot window size allocation")
    args = parser.parse_args()
    seed_everything(args.seed)
        
    logger.info(f"loading llm model {args.model}")
    model, tokenizer = build_model_and_tokenizer(args)

    device = torch.device("cuda:0")
    # TODO: add device_map for multi-GPUs
    # if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    logger.info(f"use device: {device}")
    
    bits_per_head, avg_bits_per_layer, overall_avg = compress_model(model, tokenizer, device, args)  
    # import pdb; pdb.set_trace()
    logger.info(f"avg bits: {overall_avg}")
    for layer_idx in range(len(avg_bits_per_layer)):
        logger.info(f"layer {layer_idx} avg bits: {avg_bits_per_layer[layer_idx]}")
    if args.plot_window_size_alloc:
        save_path = os.path.join("/mnt/disk3/wzn/SQAttn/scripts", f"{args.model.split('/')[-1]}_{args.bit8_thres_cos}_{args.bit8_thres_rmse}_{args.bit4_thres_cos}_{args.bit4_thres_rmse}_window_size_alloc.png")
        plot_window_size_alloc(bits_per_head, save_path)
    logger.info("*"*30)
    model.cuda()
    evaluate(model, tokenizer, args)
    

if __name__ == "__main__":
    main()