import os
import json
import argparse
from tqdm import tqdm
from langdetect import detect
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def _get_openai_response(model_name: str, prompt: str, client: OpenAI) -> str:
    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    text_segments = []
    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text_segments.append(getattr(content, "text", ""))
            elif getattr(item, "type", None) == "output_text":
                text_segments.append(getattr(item, "text", ""))
    except Exception:
        pass
    return "".join(text_segments).strip()


def get_openai_response_with_retries(
    model_name: str,
    prompt: str,
    client: OpenAI,
    max_retries: int = 2,
    use_thinking: bool = False,
):
    if use_thinking:
        _prompt = (
            f"{prompt}\n\n"
            "FORMAT: First showcase a detailed planning phase where you plan your response within <think>...</think> tags. "
            "Then produce the actual response within <response>...</response> tags. "
            "The content within the <think>...</think> tags should *not* refer to the fact that a planning phase was prompted - they should refer to the user prompt only."
        )
    else:
        _prompt = prompt

    for _ in range(max_retries):
        try:
            response_text = _get_openai_response(model_name, _prompt, client).strip()

            if use_thinking:
                if not (
                    response_text.startswith("<think>")
                    and "<response>" in response_text
                    and response_text.count("<think>") == 1
                ):
                    continue

                thinking_end = (
                    response_text.find("</think>")
                    if "</think>" in response_text
                    else response_text.find("<response>")
                )
                response_start = response_text.find("<response>") + len("<response>")
                response_end = (
                    response_text.find("</response>")
                    if "</response>" in response_text
                    else len(response_text)
                )
                thought = response_text[len("<think>") : thinking_end].strip()
                reply = response_text[response_start:response_end].strip()

                if any(
                    tag in thought or tag in reply
                    for tag in ["<think>", "</think>", "<response>", "</response>"]
                ):
                    continue

                return {"thought": thought, "response": reply}
            else:
                return response_text
        except Exception as e:
            print(e)
            continue
    return None


def process_entry(entry, client: OpenAI, model_name: str, use_thinking: bool = False):
    if len(entry.get("messages", [])) != 2:
        return None

    text = entry["messages"][0]["content"]
    if not is_english(text):
        return None

    result = get_openai_response_with_retries(
        model_name,
        text,
        client,
        use_thinking=use_thinking,
    )

    if result is None:
        return None

    if use_thinking:
        return {
            "id": entry.get("id"),
            "messages": entry.get("messages"),
            "thought": result["thought"],
            "response": result["response"],
            "source": entry.get("source"),
        }
    else:
        return {
            "id": entry.get("id"),
            "messages": entry.get("messages"),
            "response": result,
            "source": entry.get("source"),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate OpenAI responses with optional thinking phase for SFT examples",
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file with prompts")
    parser.add_argument("--output", required=True, help="Path to output JSON file for responses")
    parser.add_argument(
        "--use-thinking", action="store_true", help="Enable <think> and <response> tags"
    )
    parser.add_argument(
        "--model", default="gpt-4.1-mini", help="OpenAI model to use (default: gpt-4.1-mini)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=None, help="Thread pool size (defaults to 2x CPU cores)"
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        prompts = json.load(f)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    data = []
    max_workers = args.max_workers or min(20, (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_entry, entry, client, args.model, args.use_thinking
            ): i
            for i, entry in enumerate(prompts)
        }
        bar = tqdm(as_completed(futures), total=len(futures), desc="Processing prompts")
        for future in bar:
            result = future.result()
            if result:
                data.append(result)
                if len(data) % 100 == 0:
                    with open(args.output, "w+") as f:
                        json.dump(data, f, indent=4)
                    bar.set_description(f"At {len(data)}/{len(prompts)}")

    with open(args.output, "w+") as f:
        json.dump(data, f, indent=4)

    print(f"Saved {len(data)} responses to {args.output}")


