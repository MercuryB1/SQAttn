model="Qwen/Qwen2.5-7B-Instruct"
# change model to the path of your model if needed
basedir=./results/qwen
export CUDA_VISIBLE_DEVICES=0
bash run_seer.sh \
    $model \
    hf \
    $basedir