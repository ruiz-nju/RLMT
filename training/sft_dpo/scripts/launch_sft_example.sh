#!/bin/bash -l
#SBATCH --job-name=sft_example
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:8
#SBATCH --mem=240G
#SBATCH --time=2:45:00
#SBATCH --cpus-per-task=1

# >>> Set your conda environment here
source /path/to/conda/profile.d/conda.sh
conda activate <your-env>

mode="llama" # "llama" or "qwen"

if [ "${mode}" == "qwen" ]; then
    base_model="models/Qwen2.5-7B" # >>> Replace with your Qwen model path
    base_tokenizer="models/Qwen2.5-7B"
    lr=4e-6 
    warmup_ratio=0.1 
    gradient_accumulation_steps=1
    train_path="/path/to/your/sft/dataset" # >>> Replace with your training dataset path
    epochs=2
else
    base_model="models/Llama-3.1-8B" # >>> Replace with your Llama model path
    base_tokenizer="Llama-3.1-8B-Instruct"
    lr=4e-6
    warmup_ratio=0.1 
    gradient_accumulation_steps=2 
    train_path="/path/to/your/sft/dataset" # >>> Replace with your training dataset path
    epochs=2
fi

micro_batch_size=1
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false
max_length=20000

model_basename=$(basename ${base_model})
data_basename="example_dataset" # >>> Replace with your dataset name
out_path="checkpoints/sft__m_${model_basename}__d_${data_basename}"
weight_decay=1e-4 

# --add_longcot_config: only set if the dataset is longcot!!!
extra="--use_liger=True --use_load_from_disk=True --add_longcot_config" 

WANDB_MODE=offline torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    ../src/sft.py \
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
