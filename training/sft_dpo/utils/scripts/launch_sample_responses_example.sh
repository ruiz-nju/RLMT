#!/bin/bash -l

# >>> Set your conda environment here
ENV="<your-env>" # >>> Replace with your environment name

# Configuration
DATASET="/path/to/your/dataset.json" # >>> Replace with your dataset path
MODEL="/path/to/your/model" # >>> Replace with your model path (SFT or base model)
OUTPUT_DIR="/path/to/your/outputs" # >>> Replace with your output directory

# Generation parameters
DATASET_SIZE=10000
SHARD_SIZE=1000
TENSOR_PARALLEL_SIZE=1
MAX_TOKENS=4096
MAX_MODEL_LEN=8192
BATCH_SIZE=8
TEMPERATURE=0.7
NUM_SAMPLES=8

# Key configuration
PROMPT_KEY="prompt" # For longcot models
QUESTION_KEY="question" # For standard models
SOLUTION_KEY="solution" # Optional, for evaluation datasets

# Dataset loading
USE_LOAD_FROM_DISK=true # Set to true if loading from disk, false for HuggingFace datasets

# Output path
MODEL_BASENAME=$(basename $MODEL)
DATASET_BASENAME=$(basename $DATASET .json)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build base command
CMD="python ../src/sample_responses.py --dataset ${DATASET} --model ${MODEL} --n ${NUM_SAMPLES} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} --max_tokens ${MAX_TOKENS} --max_model_length ${MAX_MODEL_LEN} --batch_size ${BATCH_SIZE} --temperature ${TEMPERATURE} --prompt_key ${PROMPT_KEY} --question_key ${QUESTION_KEY}"

if [ -n "${SOLUTION_KEY}" ]; then
    CMD="${CMD} --solution_key ${SOLUTION_KEY}"
fi

if [ "${USE_LOAD_FROM_DISK}" = true ]; then
    CMD="${CMD} --use_load_from_disk"
fi

# Optional: specify custom longcot config (if not auto-detected from model directory)
# CMD="${CMD} --longcot_config /path/to/custom/longcot_config.json"

# Process dataset in shards
for START in $(seq 0 $SHARD_SIZE $((DATASET_SIZE - 1))); do
    END=$((START + SHARD_SIZE - 1))
    if [ $END -ge $DATASET_SIZE ]; then
        END=$((DATASET_SIZE - 1))
    fi
    
    OUT_PATH="${OUTPUT_DIR}/m_${MODEL_BASENAME}__d_${DATASET_BASENAME}__s_${START}__e_${END}.json"
    SHARD_CMD="${CMD} --out_path ${OUT_PATH} --start ${START} --end ${END}"
    
    JOB_NAME="sample_${MODEL_BASENAME}_${START}_${END}"
    
    echo "Submitting job for shard ${START}-${END}:"
    echo "${SHARD_CMD}"
    echo ""
    
    sbatch<<EOT
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --output=./joblog/%x-%A_%a.out                          
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --time=3:00:00

source /path/to/conda/profile.d/conda.sh
conda activate ${ENV}

echo "Running: ${SHARD_CMD}"
${SHARD_CMD}

echo "Shard ${START}-${END} complete! Output saved to: ${OUT_PATH}"
EOT

done

echo "All shard jobs submitted! Check joblog/ for progress."
