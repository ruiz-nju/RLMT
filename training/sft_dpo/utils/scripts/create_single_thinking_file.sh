#!/bin/bash -l
# Create a single thinking-augmented SFT file using Gemini Flash.
# Fill in the placeholders below before running.

# >>> Set paths to your input prompts JSON and desired output JSON
INPUT_JSON="/path/to/prompts.json"
OUTPUT_JSON="/path/to/output_thinking.json"

# >>> Choose the Gemini model 
GEMINI_MODEL="gemini-2.5-flash-preview-05-20"

# Guard rails: ensure variables are customized
for var in INPUT_JSON OUTPUT_JSON; do
  if [[ "${!var}" == "/path/to"* ]]; then
    echo "Error: Please set $var to a real path before running."
    exit 1
  fi
done

# Ensure the Gemini API key is present
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "Error: GEMINI_API_KEY is not set. Export it before running."
  exit 1
fi

# >>> Optionally set your environment here (uncomment and set if needed)
# source /path/to/conda/profile.d/conda.sh
# conda activate <your-env>

mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "Running Gemini sampling with thinking..."
python ../src/sample_gemini_sft_examples.py \
  --input  "$INPUT_JSON" \
  --output "$OUTPUT_JSON" \
  --model  "$GEMINI_MODEL" \
  --use-thinking

echo "Saved thinking SFT examples to: $OUTPUT_JSON"

# --- Alternative (GPT-4.1-mini) command ---
# Requires OPENAI_API_KEY and optionally a different output path
# OPENAI_MODEL="gpt-4.1-mini"
# python ../src/sample_openai_sft_examples.py \
#   --input  "$INPUT_JSON" \
#   --output "$OUTPUT_JSON" \
#   --model  "$OPENAI_MODEL" \
#   --use-thinking


