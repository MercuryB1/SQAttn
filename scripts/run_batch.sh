#!/bin/bash

export sqattn=/mnt/disk3/wzn/SQAttn
export PYTHONPATH=$sqattn:$PYTHONPATH

model_path=/mnt/disk3/hg/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
calib_dataset=gsm8k


GPUS=(3 4 5 7)
BIT8_COS_THRES=(0.9988 0.9990 0.9992 0.9994 )
BIT8_RMSE_THRES=(0.4  0.36  0.32 0.28 )
BIT4_COS_THRES=(0.9988 0.9990 0.9992 0.9994 )
BIT4_RMSE_THRES=(0.36  0.32  0.28  0.24 )

for i in "${!GPUS[@]}"; do
  GPU=${GPUS[$i]}
  BIT8_COS=${BIT8_COS_THRES[$i]}
  BIT8_RMSE=${BIT8_RMSE_THRES[$i]}
  BIT4_COS=${BIT4_COS_THRES[$i]}
  BIT4_RMSE=${BIT4_RMSE_THRES[$i]}
  sample_output_file=gsm8k_res_bit8_cos${BIT8_COS}_bit8_rmse${BIT8_RMSE}_bit4_cos${BIT4_COS}_bit4_rmse${BIT4_RMSE}_gpu${GPU}.jsonl
  LOG_FILE=log_bit8_cos${BIT8_COS}_bit8_rmse${BIT8_RMSE}_bit4_cos${BIT4_COS}_bit4_rmse${BIT4_RMSE}_gpu${GPU}.log

  echo "Launching: GPU=${GPU}, BIT8_COS=${BIT8_COS}, BIT8_RMSE=${BIT8_RMSE}, BIT4_COS=${BIT4_COS}, BIT4_RMSE=${BIT4_RMSE} -> $LOG_FILE"

  nohup bash -c "
    CUDA_VISIBLE_DEVICES=$GPU python ${sqattn}/main.py \
      --model $model_path \
      --calib_dataset $calib_dataset \
      --quant \
      --qk_qtype int \
      --v_qtype e4m3 \
      --eval_gsm8k \
      --bit8_thres_cos $BIT8_COS \
      --bit8_thres_rmse $BIT8_RMSE \
      --bit4_thres_cos $BIT4_COS \
      --bit4_thres_rmse $BIT4_RMSE \
      --sample_output_file $sample_output_file
  " > "$LOG_FILE" 2>&1 &
done

echo "🚀 All 8 quant jobs launched from strict → loose setting."