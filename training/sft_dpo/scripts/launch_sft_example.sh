#!/bin/bash -l
#SBATCH --job-name=sft_example
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:8
#SBATCH --mem=240G
#SBATCH --time=2:45:00
#SBATCH --cpus-per-task=1

# >>> Set your conda environment here
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rlmt

mode="qwen" # "llama" or "qwen"

if [ "${mode}" == "qwen" ]; then
    base_model="Qwen/Qwen2.5-7B" # >>> Replace with your Qwen model path
    base_tokenizer="Qwen/Qwen2.5-7B"
    lr=4e-6 
    warmup_ratio=0.1 
    gradient_accumulation_steps=1
    train_path="/home/bml_job/custom_workspace/job-gguryn8pkf7k/zhurui/RLMT/data/sft_dataset" # >>> Replace with your training dataset path
    epochs=2
else
    base_model="models/Llama-3.1-8B" # >>> Replace with your Llama model path
    base_tokenizer="Llama-3.1-8B-Instruct"
    lr=4e-6
    warmup_ratio=0.1 
    gradient_accumulation_steps=2 
    train_path="/home/bml_job/custom_workspace/job-gguryn8pkf7k/zhurui/RLMT/data/sft_dataset" # >>> Replace with your training dataset path
    epochs=2
fi

micro_batch_size=1
max_steps=-1
push_to_hub=false
max_length=20000

model_basename=$(basename ${base_model})
data_basename="gemini_2.5_flash_0417_sft-data" # >>> Replace with your dataset name
out_path="checkpoints/sft__m_${model_basename}__d_${data_basename}"
weight_decay=1e-4 

# --add_longcot_config: only set if the dataset is longcot!!!
extra="--use_liger=True --use_load_from_disk=True --add_longcot_config" 

echo "当前可见 GPU 编号：$CUDA_VISIBLE_DEVICES"

# 动态计算 CUDA_VISIBLE_DEVICES 中指定的 GPU 数量
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # 分割逗号，统计数组长度（处理可能的空格）
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES// /}"
    gpu_count=${#GPU_IDS[@]}
else
    # 未设置 CUDA_VISIBLE_DEVICES 时，使用所有物理 GPU
    gpu_count=$(nvidia-smi -L | wc -l)
fi

echo "使用的 GPU 数量：$gpu_count"


# 打印 GPU 详情
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits

WANDB_MODE=online torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    src/sft.py \
    --block_size=${max_length} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path=${train_path} \
    --model_name=${base_model} \
    --tokenizer_name=${base_tokenizer} \
    --warmup_ratio=${warmup_ratio} \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="configs/fsdp_config_${mode}.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir=${out_path} \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}' \
    ${extra}
