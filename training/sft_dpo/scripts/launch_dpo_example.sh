#!/bin/bash -l
#SBATCH --job-name=dpo_example
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out
#SBATCH --gres=gpu:8
#SBATCH --mem=310G
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12

# >>> Set your conda environment here
source /path/to/conda/profile.d/conda.sh
conda activate <your-env>

# Model configuration
mode="llama" # "llama" or "qwen"

base_model="/path/to/your/sft_or_zero/model" # >>> Replace with your SFT-trained model path
train_path="/path/to/your/dpo/dataset" # >>> Replace with your DPO dataset path

# Hyperparameters
lr=3e-7
min_lr=0
epochs=2
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=16
max_steps=-1
beta=0.1
push_to_hub=false
max_length=4096

# Output configuration
model_basename=$(basename ${base_model})
data_basename=$(basename ${train_path})
out_path="checkpoints/dpo/${data_basename}"

# Environment and extra settings
ENV="<something>" # >>> Set your environment name
extra="--use_load_from_disk=True"

# Setup distributed training
num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
if [ -z "$SLURM_GPUS_PER_NODE" ]; then
    export SLURM_GPUS_PER_NODE=8
fi
export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))
export NUM_NODES=$num_nodes

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

gpu_count=$(nvidia-smi -L | wc -l)

# Run DPO training
torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    src/dpo.py \
    --block_size=${max_length} \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path=${train_path} \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --beta=${beta} \
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
    ${extra}
