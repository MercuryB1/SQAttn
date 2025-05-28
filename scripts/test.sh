#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
sqattn=/mnt/disk3/wzn/SQAttn
export PYTHONPATH=$jsq:$PYTHONPATH


task_name=test

python ${sqattn}/main.py \
--model Qwen/Qwen2.5-7B \
--eval_ppl \
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
