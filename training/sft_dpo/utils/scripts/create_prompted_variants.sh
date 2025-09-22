#!/bin/bash -l

# >>> Set input base model paths and output paths here!!
# Llama-3.1-8B
INPUT_LLAMA="/path/to/Llama-3.1-8B"
OUTPUT_LLAMA_PROMPT="/path/to/prompt-Llama-3.1-8B" # thinking
OUTPUT_LLAMA_NT_PROMPT="/path/to/nt_prompt-Llama-3.1-8B" # no thinking

# Qwen2.5-7B
INPUT_QWEN="/path/to/Qwen2.5-7B"
OUTPUT_QWEN_PROMPT="/path/to/prompt-Qwen2.5-7B" # thinking
OUTPUT_QWEN_NT_PROMPT="/path/to/nt_prompt-Qwen2.5-7B" # no thinking

# Guard rails: ensure variables are customized
for var in INPUT_LLAMA OUTPUT_LLAMA_PROMPT OUTPUT_LLAMA_NT_PROMPT INPUT_QWEN OUTPUT_QWEN_PROMPT OUTPUT_QWEN_NT_PROMPT; do
  if [[ "${!var}" == "/path/to"* ]]; then
    echo "Error: Please set $var to a real path before running."
    exit 1
  fi
done

# >>> Optionally set your environment here (uncomment and set if needed)
# source /path/to/conda/profile.d/conda.sh
# conda activate <your-env>

# Run conversions
python utils/src/convert_base_to_prompted_model.py \
  --model_path "${INPUT_LLAMA}" \
  --out_path   "${OUTPUT_LLAMA_PROMPT}" \
  --max_new_tokens 64

python utils/src/convert_base_to_nothink_prompted_model.py \
  --model_path "${INPUT_LLAMA}" \
  --out_path   "${OUTPUT_LLAMA_NT_PROMPT}" \
  --max_new_tokens 64

python utils/src/convert_base_to_prompted_model.py \
  --model_path "${INPUT_QWEN}" \
  --out_path   "${OUTPUT_QWEN_PROMPT}" \
  --max_new_tokens 64

python utils/src/convert_base_to_nothink_prompted_model.py \
  --model_path "${INPUT_QWEN}" \
  --out_path   "${OUTPUT_QWEN_NT_PROMPT}" \
  --max_new_tokens 64

echo "All prompted variants created successfully."


