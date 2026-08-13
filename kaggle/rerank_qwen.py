"""Kaggle GPU notebook: score question/table pairs with Qwen3-Reranker.

The alternative to `rerank_bge.py`, and the family the organizer slides measured
(Qwen3-Embedding-8B plus a reranker took recall@10 from 63.9% to 80.8%). Reads
the same pairs.jsonl and writes the same scores.jsonl, so the local fusion step
does not care which of the two produced it.

Qwen3-Reranker is not a cross-encoder with a classification head. It is a causal
model asked a yes/no question, and the score is read off the logits for those two
tokens at the final position. That means a specific prompt format, left padding
so the last position is the real one in every row of a batch, and no
`AutoModelForSequenceClassification`.

Model sizes, all open and inside the 14B competition limit:

    Qwen/Qwen3-Reranker-0.6B   ~20 min for 50k pairs on a T4, weakest
    Qwen/Qwen3-Reranker-4B     ~2-3 h on a T4, the sensible default
    Qwen/Qwen3-Reranker-8B     needs both T4s or 4-bit; strongest

Scores are written per question as they are produced, so a session that times out
still leaves usable output for the questions it reached.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("QWEN_RERANKER", "Qwen/Qwen3-Reranker-4B")
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs.jsonl"
SCORES_PATH = "/kaggle/working/scores.jsonl"
MAX_LENGTH = 320
BATCH_SIZE = 16

INSTRUCTION = (
    "Given a Vietnamese financial question, judge whether the table contains the "
    "figure the question asks for."
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_pairs(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def already_scored(path):
    """Question IDs from a previous run, so a restarted session resumes."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).eval()
    print(f"device={device} model={MODEL_NAME}", flush=True)

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = MAX_LENGTH - len(prefix_ids) - len(suffix_ids)

    records = load_pairs(PAIRS_PATH)
    done_ids = already_scored(SCORES_PATH)
    if done_ids:
        print(f"resuming, {len(done_ids)} questions already scored", flush=True)
    pending = [record for record in records if record["id"] not in done_ids]
    total = sum(len(record["candidates"]) for record in pending)
    print(f"{len(pending)} questions, {total} pairs to score", flush=True)

    started, done = time.time(), 0
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            prompts = [
                f"<Instruct>: {INSTRUCTION}\n<Query>: {record['question']}\n<Document>: {candidate['text']}"
                for candidate in record["candidates"]
            ]
            scores = []
            for start in range(0, len(prompts), BATCH_SIZE):
                batch = tokenizer(
                    prompts[start:start + BATCH_SIZE],
                    truncation=True, max_length=budget, add_special_tokens=False,
                )["input_ids"]
                batch = [prefix_ids + ids + suffix_ids for ids in batch]
                encoded = tokenizer.pad(
                    {"input_ids": batch}, padding=True, return_tensors="pt",
                ).to(model.device)
                with torch.inference_mode():
                    logits = model(**encoded).logits[:, -1, :]
                # Probability the model answers "yes" rather than "no".
                pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
                scores.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist())
            out.write(json.dumps({
                "id": record["id"],
                "scores": {
                    candidate["table_id"]: round(score, 6)
                    for candidate, score in zip(record["candidates"], scores)
                },
            }, ensure_ascii=False) + "\n")
            out.flush()
            done += len(prompts)
            if done % 2000 < BATCH_SIZE:
                rate = done / (time.time() - started)
                print(f"{done}/{total} pairs, {rate:.1f}/s, eta {(total - done) / rate / 60:.0f} min", flush=True)

    print(f"wrote {SCORES_PATH} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
