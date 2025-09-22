#!/bin/bash -l

# Build an SFT dataset from sampled responses (Gemini or OpenAI output).
# Fill in the placeholders below before running.

# >>> Set the input sampled responses path
# - If it ends with .json, it will be loaded as JSON
# - Otherwise, it will be treated as a HuggingFace dataset directory (load_from_disk)
INPUT_PATH="/path/to/sampled_responses.json"

# >>> Set the output directory for the resulting dataset
# - If MODEL_FAMILY=both, subfolders llama/ and qwen/ will be created under this directory
OUTPUT_DIR="/path/to/output_sft_dataset"

# >>> Choose model family: llama | qwen | both
MODEL_FAMILY="llama"

# >>> Thinking mode: set to true for <think> ... </think> + <response> ... </response>; false for baseline
THINKING="true"

# Guard rails: ensure variables are customized
for var in INPUT_PATH OUTPUT_DIR; do
  if [[ "${!var}" == "/path/to"* ]]; then
    echo "Error: Please set $var to a real path before running."
    exit 1
  fi
done

case "$MODEL_FAMILY" in
  llama|qwen|both) ;;
  *) echo "Error: MODEL_FAMILY must be one of: llama | qwen | both"; exit 1;;
esac

case "${THINKING,,}" in
  true|false) ;;
  *) echo "Error: THINKING must be 'true' or 'false'"; exit 1;;
esac

# >>> Optionally set your environment here (uncomment and set if needed)
# source /path/to/conda/profile.d/conda.sh
# conda activate <your-env>

mkdir -p "${OUTPUT_DIR}"

CMD=(python src/build_sft_dataset.py \
  --input "${INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_family "${MODEL_FAMILY}")

if [[ "${THINKING,,}" == "true" ]]; then
  CMD+=(--thinking)
fi

echo "Running: ${CMD[@]}"
"${CMD[@]}"

echo "Saved SFT dataset to: ${OUTPUT_DIR}"


