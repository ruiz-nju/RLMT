#!/bin/bash -l

# >>> Set your conda environment here (optional)
# ENV="<your-env>" # >>> Replace with your environment name
# source /path/to/conda/profile.d/conda.sh
# conda activate ${ENV}

# Configuration (placeholders — replace with your paths)
INPUT_JSON="/path/to/your/responses.json"           # >>> Replace with your responses JSON (list of {prompt, responses})
OUTPUT_DIR="/path/to/your/outputs"                  # >>> Replace with your output directory for scores
REWARD_MODEL="/path/to/your/reward/model"           # >>> Replace with your reward model path or HF repo id

# Optional formatting keys (adjust if your schema differs)
PROMPT_KEY="prompt"
RESPONSE_KEY="response"
ANSWER_KEY="answer"
THOUGHT_KEY="thinking"

# Inference parameters
BATCH_SIZE=16
MAX_LENGTH=4096

mkdir -p "${OUTPUT_DIR}"

echo "Scoring (response-only mode)..."
CMD_RESPONSE_ONLY="python ../src/score_responses.py \
  --in_path ${INPUT_JSON} \
  --out_path ${OUTPUT_DIR}/scores_response_only_$DEFAULT \
  --model ${REWARD_MODEL} \
  --mode response_only \
  --prompt_key ${PROMPT_KEY} \
  --response_key ${RESPONSE_KEY} \
  --batch_size ${BATCH_SIZE} \
  --max_length ${MAX_LENGTH}"

echo "${CMD_RESPONSE_ONLY}"
# Uncomment to run directly:
# eval ${CMD_RESPONSE_ONLY}

echo "\nScoring (longcot mode, include thinking)..."
CMD_LONGCOT="python ../src/score_responses.py \
  --in_path ${INPUT_JSON} \
  --out_path ${OUTPUT_DIR}/scores_longcot_$DEFAULT \
  --model ${REWARD_MODEL} \
  --mode longcot \
  --include_thinking \
  --thinking_separator \"\\n\" \
  --prompt_key ${PROMPT_KEY} \
  --answer_key ${ANSWER_KEY} \
  --thought_key ${THOUGHT_KEY} \
  --batch_size ${BATCH_SIZE} \
  --max_length ${MAX_LENGTH}"

echo "${CMD_LONGCOT}"
# Uncomment to run directly:
# eval ${CMD_LONGCOT}

echo "\nEdit INPUT_JSON, OUTPUT_DIR, and REWARD_MODEL above, then uncomment eval lines to run."


