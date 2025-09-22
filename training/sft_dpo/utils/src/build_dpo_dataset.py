import os
import json
import argparse
import random
import datasets
import numpy as np
import shutil

from transformers import AutoTokenizer
from tqdm import tqdm

from transformers.utils import logging
logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", "-d", required=True, help="Path to scored responses dataset")
    parser.add_argument("--out_path", "-o", required=True, help="Output path for DPO dataset")
    parser.add_argument("--model_type", "-mt", choices=["llama", "qwen"], required=True, 
                       help="Model family (llama or qwen)")
    parser.add_argument("--tokenizer_path", "-tp", default=None, 
                       help="Custom tokenizer path (optional, uses default if not provided)")
    parser.add_argument("--use_longcot", action="store_true", 
                       help="Use longcot format (thinking + response)")
    parser.add_argument("--is_prompted", action="store_true", 
                       help="Is prompted model")
    parser.add_argument("--overwrite", "-ow", action="store_true")
    parser.add_argument("--min_responses", "-mr", type=int, default=4)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--max_length", "-ml", type=int, default=8192)
    parser.add_argument("--train_split", type=float, default=0.97, 
                       help="Fraction of data to use for training")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    return args

def get_default_tokenizer_path(model_type):
    """Get default tokenizer path for model type."""
    if model_type == "llama":
        return "models/Llama-3.1-8B-Instruct"
    elif model_type == "qwen":
        return "models/Qwen2.5-7B-Instruct"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_chat_standard(prompt, response):
    """Get standard chat format (no thinking)."""
    return [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": response,
        }
    ]

def get_chat_longcot(prompt, thought, answer):
    """Get longcot chat format (thinking + response)."""
    response = f"<think>\n{thought}\n</think>\n{answer}"
    return [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": response,
        }
    ]

def get_chat_prompted_standard(prompt, response):
    """Get prompted standard format (with response tags)."""
    return [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": f"<response> {response} </response>",
        }
    ]

def get_chat_prompted_longcot(prompt, thought, answer):
    """Get prompted longcot format (with think and response tags)."""
    response = f"<think> {thought} </think>\n<response> {answer} </response>"
    return [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": response,
        }
    ]

def main():
    args = parse_args()
    datasets.disable_caching()
    
    # Load dataset
    if args.dataset.endswith(".json"):
        dataset = json.load(open(args.dataset))
    else:
        dataset = datasets.load_dataset(args.dataset)
        dataset = dataset["train" if "train" in dataset else "train_sft"]
    
    # Load tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path is not None else get_default_tokenizer_path(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Determine if this is a prompted model (has custom tokenizer path)
    is_prompted = args.is_prompted
    
    dataset_list = []
    for i in tqdm(range(len(dataset))):
        instance = dataset[i]
        if len(instance["responses"]) < args.min_responses:
            continue
            
        prompt = instance["prompt"]
        responses = sorted(instance["responses"], key=lambda r: -r["score"])
        best_response, worst_response = responses[0], responses[-1]
        
        # Choose appropriate chat format based on model type and longcot setting
        if args.use_longcot:
            if is_prompted:
                chosen = get_chat_prompted_longcot(
                    prompt,
                    best_response["thought"],
                    best_response["answer"]
                )
                rejected = get_chat_prompted_longcot(
                    prompt,
                    worst_response["thought"],
                    worst_response["answer"]
                )
            else:
                chosen = get_chat_longcot(
                    prompt,
                    best_response["thought"],
                    best_response["answer"]
                )
                rejected = get_chat_longcot(
                    prompt,
                    worst_response["thought"],
                    worst_response["answer"]
                )
        else:
            if is_prompted:
                chosen = get_chat_prompted_standard(
                    prompt,
                    best_response["response"]
                )
                rejected = get_chat_prompted_standard(
                    prompt,
                    worst_response["response"]
                )
            else:
                chosen = get_chat_standard(
                    prompt,
                    best_response["response"]
                )
                rejected = get_chat_standard(
                    prompt,
                    worst_response["response"]
                )
        
        # Check length constraints
        chosen_len = len(tokenizer.encode(
            tokenizer.apply_chat_template(chosen, tokenize=False),
        ))
        if chosen_len > args.max_length:
            continue
            
        rejected_len = len(tokenizer.encode(
            tokenizer.apply_chat_template(rejected, tokenize=False),
        ))
        if rejected_len > args.max_length:
            continue
        
        dataset_list.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    
    # Shuffle and split dataset
    np.random.seed(args.seed)
    perm = np.random.permutation(len(dataset_list)).tolist()
    dataset_list = [dataset_list[i] for i in perm]
    
    len_train = int(args.train_split * len(dataset_list))
    
    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_list(dataset_list[:len_train]),
        "validation": datasets.Dataset.from_list(dataset_list[len_train:]),
    })
    
    print(f"Created DPO dataset with {len(dataset_list)} examples:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    
    if args.use_longcot:
        print("  Format: Longcot (thinking + response)")
    else:
        print("  Format: Standard (response only)")
    
    if is_prompted:
        print("  Model type: Prompted")
    else:
        print(f"  Model type: {args.model_type}")
    
    # Handle output directory
    if os.path.exists(args.out_path):
        if args.overwrite:
            print(f"Overwriting existing dataset at {args.out_path}")
            shutil.rmtree(args.out_path)
        else:
            raise ValueError(f"Dataset already exists at {args.out_path}. Use --overwrite to replace.")
    
    dataset.save_to_disk(args.out_path)
    print(f"Dataset saved to: {args.out_path}")

if __name__ == "__main__":
    main()
