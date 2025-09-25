# Language Models That Think, Chat Better

This github repository contains the code for the arXiv preprint [Language Models That Think, Chat Better](http://arxiv.org/abs/2509.20357) by Adithya Bhaskar*, Xi Ye*, and Danqi Chen.
This code includes benchmarking code for evaluating local and API-based language models on several benchmarks, and SFT, DPO, PPO, and GRPO code for training language models with thinking (i.e., RLMT as introduced in the paper) and without (normal RLHF).

## Setup
Please find the necessary dependencies inside `requirements.txt`. We recommend installing [PyTorch](https://pytorch.org/) first when creating your environment, followed by the other dependencies.
For flash attention, you may have to use the `--no_build_isolation` flag when installing it (refer to `https://github.com/Dao-AILab/flash-attention` for more installation help).

> **NOTE**: During the project, we used two different environments for the verl component of the code, and the rest of it. We have since managed to merge them, but please feel free to email us or open an issue if you have trouble.

## Benchmarking
Our benchmarking code is self-contained and designed to be easy to use and extend to other benchmarks.
It is contained in the `eval` directory, where we inlcude a README file with more instructions.

## Training
For SFT and DPO, we rely on `trl`. 
For PPO and GRPO, however, we used `verl` for its high efficiency and scalability.
We provide the SFT/DPO code inside `training/sft_dpo` and the PPO/GRPO code inside `training/ppo_grpo`.
There are other utilities inside `training/sft_dpo/utils` for the preparation of SFT and DPO datasets.
For more details, please refer to
- The README inside `training` for details on how to launch SFT, DPO, PPO, and GRPO.
- The README inside `training/utils` for more details on (1) how we prepare the "zero" versions of models for seamless training, (2) how we prompt models and API endpoints to create SFT and DPO datasets, and (3) formatting of the datasets for SFT/DPO.

## Data release

### SFT datasets

We release all datasets (SFT prompts with Gemini 2.5 Flash 0417's thoughts and responses, and the RL prompt mixture) on huggingface at this [HuggingFace collection](https://huggingface.co/collections/princeton-nlp/rlmt-experiments-68d0e7704d0c8fa49a9c1e3d).

### Trained model checkpoints

We release all model checkpoints evaluated in the paper (main experiments only) on huggingface at this [HuggingFace collection](https://huggingface.co/collections/princeton-nlp/rlmt-experiments-68d0e7704d0c8fa49a9c1e3d).

## Contact
If you run into any issues, please email us at `adithyab@princeton.edu` or `xi.ye@princeton.edu`. You can also open a GitHub issue in this repository.

## Citation
If this work or repository was helpful in your work, please cite as:
```
@misc{bhaskar2025language,
    title={Language Models that Think, Chat Better}, 
    author={Adithya Bhaskar and Xi Ye and Danqi Chen},
    year={2025},
    journal={arXiv preprint arXiv:2509.20357},
}
```