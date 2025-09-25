# Training

This half of the codebase contains the code required for training the lanugage models (thinking or non-thinking) with SFT, DPO, PPO, and GRPO.

## Utilities

Utilities for (1) prompting of a strong model to generate SFT data, (2) structuring of SFT data (formatting of thought traces, etc.) for training, (3) prompting and scoring of responses from an LM to be trained with DPO, and (4) structuring of DPO data for training are provided in `sft_dpo/utils`. Please refer to the README in `sft_dpo/utils` for more details.

## SFT

The SFT code (based on trl and on Ai2's [open-instruct](https://github.com/allenai/open-instruct)) is provided in `sft_dpo/src/sft.py`.
It relies on FSDP for distributed training, and can train even a 32B model on a single node with a 16k context length.
An example run script is provided in `sft_dpo/scripts/launch_sft_example.sh`, which should be filled in with appropriate paths and launched from within `sft_dpo`.
The provided hyperparameters were the ones used in our experiments.
An SFT run usually takes around 2.5 hours on a single node with 8 H100s to train an 8B model for two epochs on ~6k datapoints.

## DPO

After creating and formatting your DPO dataset with the utilities provided in `sft_dpo/utils`, you can train the model with DPO using the code in `sft_dpo/src/dpo.py`.
An example run script is provided in `sft_dpo/scripts/launch_dpo_example.sh`, which should be filled in with appropriate paths and launched from within `sft_dpo`.
The provided hyperparameters were the ones used in our experiments.
A DPO run usually takes around 1 hour on a single node with 8 H100s to train an 8B model for two epochs on ~6k datapoints.

## PPO and GRPO
For PPO and GRPO, we use verl (https://github.com/volcengine/verl).
The corresponding code is provided in `ppo_grpo/`.
For the most part, the code is the same as the verl code in the above link, with the following differences:
- We implement a new Reward Model Worker, the `SequenceRewardModelWorker` in  `ppo_grpo/verl/workers/fsdp_workers.py`. This worker strips thinking traces out based on the longCoT config before scoring with a reward model.
- We allow a `strict` option that treats the response as `null` if the thinking portion is not correctly formatted as per the passed longCoT config. We only use this option for our instruct models as the results for other models seem unaffected by this option.
- We provide data preparation utilities inside `ppo_grpo/scripts/data` to prepare the datasets for training (which will be placed inside `ppo_grpo/data` by default).
- For an explanation of the various hyperparamaters and options, refer to `ppo_grpo/configs/grpo__llamabase__warm-start__think.yaml`'s comments.

Please refer to further documentation provided by verl for details about what the various hyperparameters mean and how to set them.
We provide several configuration files for GRPO in `ppo_grpo/configs/` and corresponding run scripts in `ppo_grpo/scripts/train` to launch them.
These include scripts for Llama-3.1-8B (base and instruct), Qwen2.5-7B (base and instruct), and the prompted/zero versions of these models.
We also provide two example run scripts for PPO in the same directory, along with their corresponding configuration files.
Our runs use FSDPv2 with verl, and are able to fit an 8B model on a single node with 8 H100s.
The provided scripts reflect the hyperparameters used in our experiments, except that they train for 2 epochs (234 steps), whereas we take the checkpoint around step 117 (specifically, step 135) since the model tends to overfit the reward after that.
These runs usually take 14-16 hours to complete with 8 H100s and the 
[Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) reward model.