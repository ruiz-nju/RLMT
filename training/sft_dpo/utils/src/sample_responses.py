import os
import json
import argparse
import random
from vllm import LLM, SamplingParams
from tqdm import tqdm
import datasets

from transformers.utils import logging
logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", "-d", default="/path/to/your/dataset")
    parser.add_argument("--use_load_from_disk", "-uld", action="store_true")
    parser.add_argument("--out_path", "-o", default="/path/to/your/output.json")
    parser.add_argument("--model", "-m", default="/path/to/your/model")
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--tensor_parallel_size", "-tps", type=int, default=1)
    parser.add_argument("--max_model_length", "-ml", type=int, default=16384)
    parser.add_argument("--temperature", "-t", type=float, default=0.6)
    parser.add_argument("--overwrite", "-ow", action="store_true")
    parser.add_argument("--batch_size", "-bs", type=int, default=8)
    parser.add_argument("--max_tokens", "-mt", type=int, default=12000)
    parser.add_argument("--start", "-s", type=int, default=0)
    parser.add_argument("--end", "-e", type=int, default=-1)
    parser.add_argument("--n", "-n", type=int, default=8)
    
    # Key options
    parser.add_argument("--prompt_key", "-pk", default="prompt")
    parser.add_argument("--question_key", "-qk", default="question")
    parser.add_argument("--solution_key", "-sk", default=None)
    
    # Longcot options
    parser.add_argument("--longcot_config", "-lc", type=str, default=None, 
                       help="Path to longcot_config.json (optional, auto-detects from model directory)")
    
    args = parser.parse_args()
    return args

def load_longcot_config(config_path):
    """Load longcot configuration from JSON file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def format_chat_instance_longcot(instance, longcot_config, tokenizer, prompt_key):
    """Format chat instance for longcot mode."""
    prompt = instance[prompt_key]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    while prompt[-1]["role"] == "assistant":
        prompt.pop()
    prompt.append({
        "role": "assistant",
        "content": "$MARKER"
    })
    formatted = tokenizer.apply_chat_template(prompt, tokenize=False)
    
    # Use start_think_marker from config, or default to "<think>\n"
    start_think_marker = longcot_config.get("start_think_marker", "<think>\n") if longcot_config else "<think>\n"
    formatted = formatted[:formatted.index("$MARKER")] + start_think_marker
    return formatted

def format_chat_instance_standard(instance, question_key):
    """Format chat instance for standard mode."""
    if isinstance(instance[question_key], list):
        # it's already in the right format
        chat = instance[question_key]
        if chat[-1]["role"] == "assistant":
            chat = chat[:-1]
    else:
        chat = [
            {
                "role": "user",
                "content": instance[question_key]
            }
        ]
    return chat

def parse_longcot_response(response, longcot_config):
    """Parse longcot response into thought and answer."""
    if not longcot_config:
        return False, None, None
    
    longcot_delimiter = longcot_config.get("longcot_delimiter", "<response>")
    end_delimiter = longcot_config.get("end_delimiter")
    
    if longcot_delimiter not in response or (end_delimiter is not None and end_delimiter not in response):
        return False, None, None
    
    thought = response[:response.index(longcot_delimiter)].strip()
    answer = response[response.index(longcot_delimiter) + len(longcot_delimiter):]
    if end_delimiter is not None:
        answer = answer[:answer.index(end_delimiter)]
    answer = answer.strip()
    return True, thought, answer

def main():
    args = parse_args()
    datasets.disable_caching()
    
    # Load dataset
    if args.dataset.endswith(".json"):
        dataset = json.load(open(args.dataset))
    elif args.use_load_from_disk:
        dataset = datasets.load_from_disk(args.dataset)
        dataset = dataset["train" if "train" in dataset else "train_sft"]
    else:
        dataset = datasets.load_dataset(args.dataset)
        dataset = dataset["train" if "train" in dataset else "train_sft"]
    
    start = args.start
    end = args.end if args.end != -1 else len(dataset)
    start, end = max(0, start), min(len(dataset), end)
    
    # Load model
    model = LLM(
        args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_length,
        dtype="bfloat16" if not args.no_bf16 else "float32",
    )
    tokenizer = model.get_tokenizer()
    
    # Determine if this is a longcot model
    longcot_config = None
    if args.longcot_config:
        # Explicit config path provided
        longcot_config = load_longcot_config(args.longcot_config)
    else:
        # Try to auto-detect from model directory
        model_config_path = os.path.join(args.model, "longcot_config.json")
        longcot_config = load_longcot_config(model_config_path)
    
    use_longcot = longcot_config is not None
    if use_longcot:
        print(f"Detected longcot model with config: {longcot_config}")
    else:
        print("Detected standard (non-longcot) model")
    
    # Resume from existing output if it exists
    if os.path.exists(args.out_path) and not args.overwrite:
        print("Resuming from existing output file.")
        results = json.load(open(args.out_path))
        start += len(results)
    else:
        results = []
    
    # Process dataset
    bar = tqdm(range(start, end, args.batch_size))
    for start_ in bar:
        end_ = min(start_ + args.batch_size, end)
        
        if use_longcot:
            # Longcot mode: format with thinking markers
            chats = [
                format_chat_instance_longcot(dataset[i], longcot_config, tokenizer, args.prompt_key)
                for i in range(start_, end_)
            ]
            outputs = model.generate(
                chats,
                SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    n=args.n,
                    skip_special_tokens=False,
                )
            )
            
            # Parse longcot responses
            for i in range(start_, end_):
                parsed = []
                for output in outputs[i - start_].outputs:
                    success, thought, answer = parse_longcot_response(
                        output.text.strip(),
                        longcot_config
                    )
                    if success:
                        parsed.append({
                            "thought": thought,
                            "answer": answer,
                        })
                instance = dataset[i]
                instance["responses"] = parsed
                results.append(instance)
        else:
            # Standard mode: regular chat generation
            chats = [
                format_chat_instance_standard(dataset[i], args.question_key)
                for i in range(start_, end_)
            ]
            outputs = model.chat(
                chats,
                SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    n=args.n
                )
            )
            
            # Parse standard responses
            for i in range(start_, end_):
                parsed = [
                    output.text.strip()
                    for output in outputs[i - start_].outputs
                ]
                if args.solution_key is not None:
                    instance = {
                        "question": dataset[i][args.question_key],
                        "solution": dataset[i][args.solution_key],
                        "responses": parsed,
                    }
                else:
                    instance = {
                        "question": dataset[i][args.question_key],
                        "responses": parsed,
                    }
                results.append(instance)
        
        # Save intermediate results
        if end_ // 5 != start_ // 5:
            with open(args.out_path, "w") as f:
                json.dump(results, f, indent=4)
    
    # Save final results
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
