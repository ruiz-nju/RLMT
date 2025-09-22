import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
import transformers
import trl
import torch
import shutil
from accelerate import Accelerator
import datetime


@dataclass
class TrainingConfig:
    train_file_path: str = field()
    
    model_name: str = field(default="models/Qwen2.5-7B")
    block_size: int = field(default=20000)
    wandb_project: Optional[str] = field(default="rlmt")
    use_load_from_disk: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.DPOConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name or "32B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    load_fn = load_from_disk if config.use_load_from_disk else load_dataset
    dataset = load_fn(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        tokenizer.pad_token = "<|fim_pad|>"
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

    args.max_length = config.block_size
    eval_key = "validation" if "validation" in dataset else (
        "test" if "test" in dataset else "train"
    )

    trainer = trl.DPOTrainer(
        model,
        processing_class=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset[eval_key],
        args=args,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()

    # Finally: if the input path has `longcot_config.json` then we need to copy it to the output directory
    if os.path.exists(os.path.join(config.model_name, "longcot_config.json")):
        shutil.copy(
            os.path.join(config.model_name, "longcot_config.json"),
            os.path.join(args.output_dir, "longcot_config.json")
        )
        print(f"Copied longcot_config.json to {args.output_dir}")


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    train()
    torch.distributed.destroy_process_group()