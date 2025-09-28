import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score responses with a reward model. Supports response-only and longcot formats. For longcot, only the final answer is scored, never the thinking content."
    )

    # IO
    parser.add_argument("--in_path", "-i", type=str, required=True,
                        help="Path to JSON file containing a list of items with 'prompt' and 'responses'.")
    parser.add_argument("--out_path", "-o", type=str, required=True,
                        help="Path to write scored JSON.")

    # Model
    parser.add_argument("--model", "-m", type=str, default="models/Skywork-Reward-Llama-3.1-8B-v0.2",
                        help="Reward model path or HF hub id.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"],
                        help="Torch dtype to use for the reward model.")

    # Data formatting
    parser.add_argument("--prompt_key", "-pk", type=str, default="prompt",
                        help="Key in dataset entries for the user prompt.")
    parser.add_argument("--response_key", type=str, default="response",
                        help="Key inside response dicts for response-only datasets.")
    parser.add_argument("--answer_key", type=str, default="answer",
                        help="Key inside response dicts for the final answer in longcot datasets.")
    parser.add_argument("--thought_key", type=str, default="thinking",
                        help="Key inside response dicts for the thinking content in longcot datasets.")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "response_only", "longcot"],
                        help="How to format assistant content. 'auto' detects presence of thinking.")

    # Inference
    parser.add_argument("--batch_size", "-bs", type=int, default=16,
                        help="Total number of responses per forward pass. Effective batch per prompt is batch_size/num_expected_resps_per_input.")
    parser.add_argument("--num_expected_resps_per_input", "-n", type=int, default=8,
                        help="Expected number of responses per dataset entry; used to derive actual step size.")
    parser.add_argument("--max_length", "-ml", type=int, default=4096,
                        help="Max tokens for the reward model input.")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Start index of dataset slice to process.")
    parser.add_argument("--end", "-e", type=int, default=-1,
                        help="End index (exclusive). -1 means till end.")
    parser.add_argument("--overwrite", "-ow", action="store_true",
                        help="Overwrite existing output if present. If not set and file exists, will resume by appending new results.")

    args = parser.parse_args()

    # Expand $DEFAULT in out_path using basename of input
    default_str = os.path.basename(args.in_path)
    if args.start != 0 or args.end != -1:
        default_str = default_str.replace(".json", f"__s_{args.start}__e_{args.end}.json")
    args.out_path = args.out_path.replace("$DEFAULT", default_str)

    return args


def _get_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    return None


def _extract_assistant_text(
    response_item: Any,
    mode: str,
    response_key: str,
    answer_key: str,
    thought_key: str,
) -> str:
    # String responses
    if isinstance(response_item, str):
        return response_item

    # Dict responses
    if not isinstance(response_item, dict):
        raise ValueError(f"Unsupported response item type: {type(response_item)}")

    # Auto-detect longcot if thinking exists
    is_longcot = (
        mode == "longcot" or (mode == "auto" and (thought_key in response_item and response_item.get(thought_key)))
    )

    if is_longcot:
        # Always return only the final answer/response, never the thinking
        final_text = response_item.get(answer_key) or response_item.get(response_key)
        return str(final_text) if final_text is not None else ""

    # response-only
    if response_key in response_item:
        return str(response_item[response_key])
    if answer_key in response_item:
        return str(response_item[answer_key])
    # Fallback: join known textual fields
    return str(response_item)


def _ensure_response_dict_list(entry: Dict[str, Any], response_key: str) -> None:
    # Ensure responses are list of dicts with a stable field to attach scores
    if len(entry.get("responses", [])) == 0:
        return
    first = entry["responses"][0]
    if isinstance(first, str):
        entry["responses"] = [{response_key: r} for r in entry["responses"]]


@torch.no_grad()
def main():
    args = parse_args()

    torch_dtype = _get_torch_dtype(args.dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch_dtype if torch_dtype is not None else None,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset: List[Dict[str, Any]] = json.load(open(args.in_path))
    outputs: List[Dict[str, Any]] = []

    start, end = args.start, args.end
    if end == -1 or end > len(dataset):
        end = len(dataset)

    # Resume if possible
    if not args.overwrite and os.path.exists(args.out_path):
        try:
            outputs = json.load(open(args.out_path))
            start = max(start, len(outputs))
        except Exception:
            # If file is corrupted or empty, start from provided index
            pass

    # We iterate by prompts but batch by flattened responses
    step = max(1, args.batch_size // max(1, args.num_expected_resps_per_input))

    for i in tqdm(range(start, end, step)):
        formatted: List[str] = []
        i_ = min(i + step, end)

        # Prepare model inputs
        for j in range(i, i_):
            entry = dataset[j]
            prompt = entry.get(args.prompt_key, entry.get("prompt"))
            if prompt is None:
                raise KeyError(f"Entry {j} missing prompt under key '{args.prompt_key}'.")

            for response_item in entry.get("responses", []):
                assistant_text = _extract_assistant_text(
                    response_item=response_item,
                    mode=args.mode,
                    response_key=args.response_key,
                    answer_key=args.answer_key,
                    thought_key=args.thought_key,
                )
                chat = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
                formatted.append(tokenizer.apply_chat_template(chat, tokenize=False))

        if len(formatted) == 0:
            continue

        input_ids = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).input_ids.to("cuda")

        rewards = model(input_ids).logits.detach().cpu().reshape(-1).tolist()

        # Attach scores to output structure
        cur = 0
        for j in range(i, i_):
            entry = dataset[j]

            # Normalize prompt key to 'prompt' for downstream tools
            if args.prompt_key != "prompt" and args.prompt_key in entry:
                entry["prompt"] = entry.pop(args.prompt_key)

            _ensure_response_dict_list(entry, args.response_key)

            for r in range(len(entry.get("responses", []))):
                if cur >= len(rewards):
                    break
                entry["responses"][r]["score"] = rewards[cur]
                cur += 1

            outputs.append(entry)

        assert cur == len(rewards), f"Expected {cur} rewards, got {len(rewards)} for slice {i}:{i_}"

        # Periodically flush
        if i_ // 5 != i // 5:
            json.dump(outputs, open(args.out_path, "w+"), indent=4)

    json.dump(outputs, open(args.out_path, "w+"), indent=4)


if __name__ == "__main__":
    main()


