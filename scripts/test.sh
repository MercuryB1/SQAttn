#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
sqattn=/mnt/disk3/wzn/SQAttn
export PYTHONPATH=$jsq:$PYTHONPATH


task_name=test

python ${sqattn}/main.py \
--model /mnt/disk3/hg/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2 \
--calib_dataset gsm8k \
--quant \
--qk_qtype int \
--v_qtype e4m3 \
--eval_ppl \
--eval_gsm8k \
--bit8_thres_cos 0.3 \
--bit8_thres_rmse 3 \
--bit4_thres_cos 0.998 \
--bit4_thres_rmse 0.25 \
--plot_window_size_alloc \
--plot_window_size_alloc_save_dir /mnt/disk3/wzn/SQAttn/results \
--mse_output block \
--gsm8k_prompt /mnt/disk3/wzn/SQAttn/sparse_quant_attn/eval/gsm8k_prompt.txt \
# --tasks wikitext \
# --batch_size 1 \
# --dynamic_shape \


# python ${jsq}/main.py \


# --model /mnt/nvme1/models/llama2/llama2-7b \
# --pruning_method wanda \
# --rho 2.1 \
# --sparsity_ratio 0.5 \
# --sparsity_type unstructured \
# --tasks wikitext \
