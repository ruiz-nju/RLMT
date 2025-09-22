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

@dataclass
class TrainingConfig:
    train_file_path: str = field()
    
    model_name: str = field(default="models/Qwen2.5-32B-Instruct")
    tokenizer_name: str = field(default=None)
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="rlmt")
    use_load_from_disk: bool = field(default=False)
    dagger: bool = field(default=False)
    add_longcot_config: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

def get_default_tokenizer_name(model_name):
    if "Llama" in model_name:
        return "Llama-3.1-8B-Instruct"
    elif "Qwen" in model_name:
        return "models/Qwen2.5-7B"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_default_tokens(model_name):
    if "Llama" in model_name:
        return "<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|reserved_special_token_5|>"
    elif "Qwen" in model_name:
        return "<|im_start|>user", "<|im_start|>assistant\n", "<|fim_pad|>"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    modeling_class = transformers.AutoModelForCausalLM

    # loading model
    kwargs = {}
    if "70B" in config.model_name or "32B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = modeling_class.from_pretrained(config.model_name, **kwargs)
    else:
        model = modeling_class.from_pretrained(config.model_name)

    load_fn = load_from_disk if config.use_load_from_disk else load_dataset
    dataset = load_fn(config.train_file_path)
    
    # Remove messages field to prevent TRL from auto-formatting since we want to use the text field
    if 'messages' in dataset['train'].column_names:
        dataset = dataset.remove_columns(['messages'])

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    
    # Check if chat template is set, if not, use the one from Llama-3.1-8B-Instruct
    if tokenizer.chat_template is None:
        logging.info("No chat template found, loading from models/Llama-3.1-8B-Instruct")
        instruct_tokenizer = transformers.AutoTokenizer.from_pretrained(
            get_default_tokenizer_name(config.model_name)
        )
        tokenizer.chat_template = instruct_tokenizer.chat_template
    
    instruction_template, response_template, pad_token = get_default_tokens(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = pad_token

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Add longcot config if requested
    if config.add_longcot_config:
        import json
        longcot_config = {
            "longcot": True,
            "longcot_delimiter": "<response>",
            "end_delimiter": None,
            "start_think_marker": None
        }
        with open(os.path.join(args.output_dir, "longcot_config.json"), "w") as f:
            json.dump(longcot_config, f, indent=4)
        logging.info(f"Saved longcot_config.json to {args.output_dir}")
    
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    train()
    torch.distributed.destroy_process_group()