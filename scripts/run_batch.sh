#!/bin/bash

export sqattn=/opt/tiger/LLaMA-Factory/SQAttn
export PYTHONPATH=$sqattn:$PYTHONPATH

model_path=Qwen/Qwen2.5-7B-Instruct
calib_dataset=longbench


GPUS=(0 1 2 3 4 5 6 7)
BIT8_THRES_LIST=(0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 )
BIT4_THRES_LIST=(0.98 0.95 0.80 0.85 0.80 0.75 0.70 0.65)

for i in "${!GPUS[@]}"; do
  GPU=${GPUS[$i]}
  BIT8_THRES=${BIT8_THRES_LIST[$i]}
  BIT4_THRES=${BIT4_THRES_LIST[$i]}
  sample_output_file=longbench_res_bit8_cos${BIT8_THRES}_bit4_cos${BIT4_THRES}_gpu${GPU}.jsonl
  LOG_FILE=logs/bit8_thres_${BIT8_THRES}_bit4_thres_${BIT4_THRES}.log

  echo "Launching: GPU=${GPU}, BIT8_THRES=${BIT8_THRES}, BIT4_THRES=${BIT4_THRES} -> $LOG_FILE"

  nohup bash -c "
    CUDA_VISIBLE_DEVICES=$GPU python ${sqattn}/main.py \
      --model $model_path \
      --calib_dataset $calib_dataset \
      --quant \
      --qk_qtype int \
      --v_qtype e4m3 \
      --bit8_thres $BIT8_THRES \
      --bit4_thres $BIT4_THRES \
      --sample_output_file $sample_output_file \
      --use_relative_distance \
      --method ours \
  " > "$LOG_FILE" 2>&1 &
done

echo "🚀 All 8 quant jobs launched from strict → loose setting."