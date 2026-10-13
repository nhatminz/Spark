"""Local datasets, chat tokenization, and deterministic batch ordering."""

import hashlib
import json
import random
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".parquet":
        from pyarrow import parquet

        rows = parquet.read_table(path).to_pylist()
    else:
        raise ValueError("Data must be JSON, JSONL, or Parquet")
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("Data must contain a nonempty list of objects")
    return rows


def data_signature(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def messages(row):
    turns = row.get("messages", row.get("conversations"))
    if not isinstance(turns, list) or not turns:
        raise ValueError("Pretraining requires messages or conversations")
    result = []
    for turn in turns:
        role = turn.get("role", turn.get("from"))
        role = {"human": "user", "gpt": "assistant", "ASSISTANT": "assistant"}.get(
            role, role
        )
        content = turn.get("content", turn.get("value"))
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(
            content, str
        ):
            raise ValueError("Invalid conversation role or content")
        result.append({"role": role, "content": content})
    if result[0]["role"] != "system":
        result.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    return result


def conversation_tokens(tokenizer, row, limit):
    turns = messages(row)
    ids, mask = [], []
    for index, turn in enumerate(turns):
        current = tokenizer.apply_chat_template(
            turns[: index + 1], tokenize=True, add_generation_prompt=False
        )
        if current[: len(ids)] != ids:
            raise ValueError(
                "Chat template must preserve token prefixes when adding turns"
            )
        mask.extend([int(turn["role"] == "assistant")] * (len(current) - len(ids)))
        ids = current
    return ids[:limit], mask[:limit]


def prompt_tokens(tokenizer, row, limit):
    prompt = row.get("messages", row.get("prompt"))
    if prompt is None:
        question = row.get("question")
        if not isinstance(question, str):
            raise ValueError("Training rows require question, prompt, or messages")
        prompt = [
            {"role": "system", "content": "You are a math problem assistant."},
            {
                "role": "user",
                "content": (
                    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                    "            Write a response that appropriately completes the request.\n"
                    "            Your response should include your thought process enclosed within <think></think> tags\n"
                    "            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n\n"
                    "            ### Instruction:\n"
                    + question
                    + "\nPlease reason step by step, and put your final answer within \\boxed{}"
                ),
            },
        ]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    if (
        not isinstance(prompt, list)
        or not prompt
        or prompt[-1].get("role") == "assistant"
    ):
        raise ValueError("Prompt must end before the assistant response")
    ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True
    )
    if not ids or len(ids) > limit:
        raise ValueError("Prompt is empty or exceeds max-length; adjust max-length")
    answer = row.get("answer")
    if answer is None and isinstance(row.get("reward_model"), dict):
        answer = row["reward_model"].get("ground_truth")
    if answer is None:
        raise ValueError("Training rows require answer or reward_model.ground_truth")
    answer = str(answer).split("####")[-1].strip()
    if "\\boxed{" not in answer:
        answer = "\\boxed{" + answer + "}"
    return ids, answer


def windows(length, batch_size, accumulation, seed, epoch):
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    batches = [
        order[start : start + batch_size] for start in range(0, length, batch_size)
    ]
    return [
        batches[start : start + accumulation]
        for start in range(0, len(batches), accumulation)
    ]
