#!/bin/bash -l

# Build DPO dataset from scored responses.
# This script demonstrates different configurations for the build_dpo_dataset.py script.

# >>> Set your conda environment here
ENV="<your-env>" # >>> Replace with your environment name

# Configuration
SCORED_DATASET="/path/to/your/scored_responses.json" # >>> Replace with your scored responses dataset
OUTPUT_BASE_DIR="/path/to/your/dpo_datasets" # >>> Replace with your output directory

# Model configurations
MODEL_TYPE="llama" # "llama" or "qwen"
TOKENIZER_PATH="/path/to/your/custom/tokenizer" # >>> Optional: custom tokenizer for prompted models

# Dataset parameters
MIN_RESPONSES=4
MAX_LENGTH=8192
TRAIN_SPLIT=0.97
SEED=42

# Create output directory
mkdir -p "${OUTPUT_BASE_DIR}"

# Commands will be built as single strings below

# Guard rails: ensure variables are customized
for var in SCORED_DATASET OUTPUT_BASE_DIR; do
  if [[ "${!var}" == "/path/to"* ]]; then
    echo "Error: Please set $var to a real path before running."
    exit 1
  fi
done

case "$MODEL_TYPE" in
  llama|qwen) ;;
  *) echo "Error: MODEL_TYPE must be 'llama' or 'qwen'"; exit 1;;
esac

echo "Building DPO datasets with the following configurations:"
echo "  Scored dataset: ${SCORED_DATASET}"
echo "  Output directory: ${OUTPUT_BASE_DIR}"
echo "  Model type: ${MODEL_TYPE}"
echo "  Min responses: ${MIN_RESPONSES}"
echo "  Max length: ${MAX_LENGTH}"
echo ""

# Configuration 1: Standard model, non-longcot
echo "=== Configuration 1: Standard ${MODEL_TYPE}, non-longcot ==="
OUTPUT_PATH_1="${OUTPUT_BASE_DIR}/dpo_standard_${MODEL_TYPE}_nothink"
CMD_1="python src/build_dpo_dataset.py --dataset ${SCORED_DATASET} --model_type ${MODEL_TYPE} --min_responses ${MIN_RESPONSES} --max_length ${MAX_LENGTH} --train_split ${TRAIN_SPLIT} --seed ${SEED} --out_path ${OUTPUT_PATH_1}"

echo "Command: ${CMD_1}"
echo "Output: ${OUTPUT_PATH_1}"
echo ""

# Configuration 2: Standard model, longcot
echo "=== Configuration 2: Standard ${MODEL_TYPE}, longcot ==="
OUTPUT_PATH_2="${OUTPUT_BASE_DIR}/dpo_standard_${MODEL_TYPE}_think"
CMD_2="python src/build_dpo_dataset.py --dataset ${SCORED_DATASET} --model_type ${MODEL_TYPE} --min_responses ${MIN_RESPONSES} --max_length ${MAX_LENGTH} --train_split ${TRAIN_SPLIT} --seed ${SEED} --out_path ${OUTPUT_PATH_2} --use_longcot"

echo "Command: ${CMD_2}"
echo "Output: ${OUTPUT_PATH_2}"
echo ""

# Configuration 3: Prompted model, non-longcot (if tokenizer path provided)
if [[ "${TOKENIZER_PATH}" != "/path/to"* ]]; then
    echo "=== Configuration 3: Prompted ${MODEL_TYPE}, non-longcot ==="
    OUTPUT_PATH_3="${OUTPUT_BASE_DIR}/dpo_prompted_${MODEL_TYPE}_nothink"
    CMD_3="python src/build_dpo_dataset.py --dataset ${SCORED_DATASET} --model_type ${MODEL_TYPE} --min_responses ${MIN_RESPONSES} --max_length ${MAX_LENGTH} --train_split ${TRAIN_SPLIT} --seed ${SEED} --out_path ${OUTPUT_PATH_3} --tokenizer_path ${TOKENIZER_PATH}"
    
    echo "Command: ${CMD_3}"
    echo "Output: ${OUTPUT_PATH_3}"
    echo ""
    
    # Configuration 4: Prompted model, longcot
    echo "=== Configuration 4: Prompted ${MODEL_TYPE}, longcot ==="
    OUTPUT_PATH_4="${OUTPUT_BASE_DIR}/dpo_prompted_${MODEL_TYPE}_think"
    CMD_4="python src/build_dpo_dataset.py --dataset ${SCORED_DATASET} --model_type ${MODEL_TYPE} --min_responses ${MIN_RESPONSES} --max_length ${MAX_LENGTH} --train_split ${TRAIN_SPLIT} --seed ${SEED} --out_path ${OUTPUT_PATH_4} --tokenizer_path ${TOKENIZER_PATH} --use_longcot"
    
    echo "Command: ${CMD_4}"
    echo "Output: ${OUTPUT_PATH_4}"
    echo ""
else
    echo "=== Skipping prompted configurations (no custom tokenizer path provided) ==="
    echo ""
fi

# Ask user which configuration to run
echo "Available configurations:"
echo "1. Standard ${MODEL_TYPE}, non-longcot"
echo "2. Standard ${MODEL_TYPE}, longcot"
if [[ "${TOKENIZER_PATH}" != "/path/to"* ]]; then
    echo "3. Prompted ${MODEL_TYPE}, non-longcot"
    echo "4. Prompted ${MODEL_TYPE}, longcot"
    echo "5. All configurations"
    echo ""
    read -p "Enter configuration number (1-5): " choice
else
    echo "3. All configurations"
    echo ""
    read -p "Enter configuration number (1-3): " choice
fi

# Execute selected configuration(s)
case $choice in
    1)
        echo "Running Configuration 1..."
        ${CMD_1}
        ;;
    2)
        echo "Running Configuration 2..."
        ${CMD_2}
        ;;
    3)
        if [[ "${TOKENIZER_PATH}" != "/path/to"* ]]; then
            echo "Running Configuration 3..."
            ${CMD_3}
        else
            echo "Running all configurations..."
            echo "Running Configuration 1..."
            ${CMD_1}
            echo "Running Configuration 2..."
            ${CMD_2}
        fi
        ;;
    4)
        if [[ "${TOKENIZER_PATH}" != "/path/to"* ]]; then
            echo "Running Configuration 4..."
            ${CMD_4}
        else
            echo "Invalid choice. Please run the script again."
            exit 1
        fi
        ;;
    5)
        if [[ "${TOKENIZER_PATH}" != "/path/to"* ]]; then
            echo "Running all configurations..."
            echo "Running Configuration 1..."
            ${CMD_1}
            echo "Running Configuration 2..."
            ${CMD_2}
            echo "Running Configuration 3..."
            ${CMD_3}
            echo "Running Configuration 4..."
            ${CMD_4}
        else
            echo "Invalid choice. Please run the script again."
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "DPO dataset creation complete!"
echo "Check the output directory: ${OUTPUT_BASE_DIR}"
