#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from shutil import rmtree
from transformers import AutoTokenizer


def load_inputs(input_path: str) -> List[Dict[str, Any]]:
    """Load sampled responses.

    - If input_path ends with .json, load as a JSON list.
    - Otherwise, assume it is a HuggingFace dataset directory saved via save_to_disk.
    """
    if input_path.endswith(".json"):
        with open(input_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON {input_path} must contain a list of entries")
        return data

    # Load from HuggingFace dataset on disk
    from datasets import load_from_disk

    ds = load_from_disk(input_path)
    # Handle both Dataset and DatasetDict
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            ds = ds["train"]
        else:
            # Fallback to first split if any
            first_split = next(iter(ds.keys()))
            ds = ds[first_split]
    # Convert to list of dicts
    return ds.to_list()


def has_forbidden_tags(text: str) -> bool:
    for tag in ["<think>", "</think>", "<response>", "</response>"]:
        if tag in text:
            return True
    return False


def construct_messages(prompt: str, response: str, thought: str, thinking: bool) -> List[Dict[str, str]]:
    if thinking:
        assistant_content = f"<think>\n{thought}\n</think>\n\n{response}"
    else:
        assistant_content = response
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_content},
    ]


def build_examples(entries: List[Dict[str, Any]], thinking: bool) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    ids_seen = set()
    for e in tqdm(entries):
        msgs = e.get("messages", [])
        if len(msgs) != 2:
            continue

        prompt = msgs[0].get("content", "")
        if not isinstance(prompt, str) or not prompt:
            continue

        # Required fields depending on thinking mode
        if thinking:
            thought = e.get("thought")
            response = e.get("response")
            if not isinstance(thought, str) or not isinstance(response, str):
                continue
            # Skip examples that already contain tags
            if has_forbidden_tags(thought) or has_forbidden_tags(response):
                continue
        else:
            response = e.get("response")
            if not isinstance(response, str):
                continue
            if has_forbidden_tags(response):
                continue
            thought = ""  # unused

        ex_id = e.get("id")
        if ex_id in ids_seen:
            continue
        ids_seen.add(ex_id)

        messages = construct_messages(prompt, response, thought, thinking)
        examples.append(
            {
                "id": ex_id,
                "source": e.get("source"),
                "messages": messages,
            }
        )
    return examples


def tokenize_and_build(
    examples: List[Dict[str, Any]],
    tokenizer_path: str,
) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    texts: List[str] = []
    for ex in examples:
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        texts.append(text)

    rows = []
    for ex, text in zip(examples, texts):
        row = dict(ex)
        row["text"] = text
        rows.append(row)

    ds = Dataset.from_list(rows)
    return DatasetDict({"train": ds})


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from sampled responses (Gemini or OpenAI)")
    parser.add_argument(
        "--input",
        required=True,
        help="Sampled responses path: JSON file (.json) or a HF dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the resulting dataset(s)",
    )
    parser.add_argument(
        "--model_family",
        choices=["llama", "qwen", "both"],
        default="llama",
        help="Which tokenizer to format for",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="If set, expects entries to include 'thought' and will embed <think>...</think>",
    )
    parser.add_argument(
        "--llama_tokenizer_path",
        default="models/Llama-3.1-8B-Instruct",
        help="Path to Llama tokenizer/model for chat template",
    )
    parser.add_argument(
        "--qwen_tokenizer_path",
        default="models/Qwen2.5-7B-Instruct",
        help="Path to Qwen tokenizer/model for chat template",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    entries = load_inputs(args.input)
    examples = build_examples(entries, thinking=args.thinking)

    # Decide save layout
    if args.model_family in ("llama", "both"):
        llama_ds = tokenize_and_build(examples, args.llama_tokenizer_path)
        llama_dir = (
            args.output_dir
            if args.model_family == "llama"
            else os.path.join(args.output_dir, "llama")
        )
        if os.path.exists(llama_dir):
            rmtree(llama_dir)
        llama_ds.save_to_disk(llama_dir)

    if args.model_family in ("qwen", "both"):
        qwen_ds = tokenize_and_build(examples, args.qwen_tokenizer_path)
        qwen_dir = (
            args.output_dir
            if args.model_family == "qwen"
            else os.path.join(args.output_dir, "qwen")
        )
        if os.path.exists(qwen_dir):
            rmtree(qwen_dir)
        qwen_ds.save_to_disk(qwen_dir)

    print("Done.")


if __name__ == "__main__":
    main()


