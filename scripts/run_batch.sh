#!/bin/bash

export sqattn=/mnt/disk3/wzn/SQAttn
export PYTHONPATH=$sqattn:$PYTHONPATH

model_path=/mnt/disk3/hg/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
calib_dataset=gsm8k

# 分别定义 GPU, cos 阈值, rmse 阈值的对应关系
GPUS=(3 5 6 7)
COS_THRES=(0.9990 0.9992 0.9994 0.9996)
RMSE_THRES=(0.010 0.012 0.014 0.016 )

for i in "${!GPUS[@]}"; do
  GPU=${GPUS[$i]}
  COS=${COS_THRES[$i]}
  RMSE=${RMSE_THRES[$i]}
  sample_output_file=gsm8k_res_cos${COS}_rmse${RMSE}_gpu${GPU}.jsonl
  LOG_FILE=log_cos${COS}_rmse${RMSE}_gpu${GPU}.log

  echo "Launching: GPU=${GPU}, COS=${COS}, RMSE=${RMSE} -> $LOG_FILE"

  nohup bash -c "
    CUDA_VISIBLE_DEVICES=$GPU python ${sqattn}/main.py \
      --model $model_path \
      --calib_dataset $calib_dataset \
      --quant \
      --qk_qtype int \
      --v_qtype e4m3 \
      --eval_gsm8k \
      --bit8_thres_cos $COS \
      --bit8_thres_rmse $RMSE \
      --sample_output_file $sample_output_file
  " > "$LOG_FILE" 2>&1 &
done

echo "All jobs launched in background with matched cos-rmse pairs."
