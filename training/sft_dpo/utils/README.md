# Utilities for data preparation

This directory contains utilities that read JSONs and structure the data in a format directly usable for SFT or DPO training. 
It also has files that create prompted (`zero`) versions of Llama and Qwen models for further training.

## Creating zero versions of models

To create `zero` models, we rely on the base model being to complete a prompt in-context, for example:
```
A conversation between User and Assistant. Following the User's query, the Assistant first plans a response, and then provides the response. The internal reasoning process is enclosed within <think> </think> tags and the response is enclosed within <response> </response> tags, i.e., in the format <think> reasoning process here </think> <response> response here </response>.
User: <query> How do I bake a cake? </query>
Assistant:
```
To standardize the prompt, we simply assign the desired format to the model's tokenizer. For ease of implementation, we use the python files `src/convert_base_to_prompted_model.py` and `src/convert_base_to_nothink_prompted_model.py` to create the prompted models with the modified chat templates.
You create these variants for Llama and Qwen by using the script `scripts/create_prompted_variants.sh`.

## Creating SFT datasets
Creation of the SFT dataset requires (1) prompting a strong model to generate responses to a set of prompts with simulated thinking traces (i.e. ask the source to generate a fake thinking trace before the response), and (2) structuring the responses into the standard format for training.

For the former, use the bash script `scripts/create_single_thinking_file.sh`, which calls `src/sample_gemini_sft_examples.py` or `src/sample_openai_sft_examples.py` with thinking.
The same dataset can be used to create thinking or non-thinking SFT datasets (the latter by simply discarding the thinking traces).
For this, use the bash script `scripts/create_sft_dataset.sh` (which calls `src/build_sft_dataset.py`).

## Creating DPO datasets

Creation of the DPO dataset involves (1) prompting the model to-be-trained to generate a number of responses to a set of prompts, (2) scoring these responses with a reward model (after stripping out thinking, if enabled), and (3) picking chosen and rejected responses (best and worst) to form the DPO dataset.

For step 1, see `scripts/launch_sample_responses.sh` for an example of how to use `src/sample_responses.py` to sample responses.
Similarlty, see `scripts/launch_score_responses_example.sh` for an example of how to use `src/score_responses.py` to score the responses.
Finally, fill in the placeholders in `scripts/create_dpo_dataset.sh` and run it to create the DPO dataset (it calls `src/build_dpo_dataset.py`).