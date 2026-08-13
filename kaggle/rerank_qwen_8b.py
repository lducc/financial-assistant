"""Kaggle GPU notebook: score candidate pairs with Qwen3-Reranker-8B in 8-bit.

Paste into a notebook with the accelerator set to GPU and the exported pairs
attached as a dataset, then run. It reads pairs_v3.jsonl and writes scores.jsonl
for `scripts/apply_rerank_scores.py` to fuse locally.

8.19B parameters is 16.4 GB in fp16, which does not fit a 16 GB T4, so the model
loads in 8-bit at about 8.2 GB — one card, one process, near-lossless. That costs
speed: bitsandbytes computes outlier features in higher precision, roughly 1.5-2x
fp16 on Turing. Expect 9-15 h for 50,335 pairs. Scores append per question and
finished IDs are skipped, so if the 12-hour session ends, rerun the cell and it
carries on from where it stopped.

Retrieval is untouched. Only the order of already-retrieved candidates changes,
so discarding the run means deleting one file.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs_v3.jsonl"
SCORES_PATH = "/kaggle/working/scores.jsonl"
# 16% of candidates run past 320 tokens, and the line-item inventory that tells
# the model what a table holds sits early in the text, so 512 keeps what matters.
MAX_LENGTH = 512
BATCH_SIZE = 8

INSTRUCTION = (
    "The company, reporting period and statement scope of every candidate have "
    "already been verified. Judge only one thing: does this table contain the "
    "line item the question asks for? Vietnamese accounting labels differ by a "
    "single word and mean different figures — chi phí lãi vay is not lãi tiền "
    "gửi, and a note restating a figure counts as containing it."
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


def build_query(record):
    """Question plus the line item it names, when the export carries one.

    The question states the company, the year and the output unit, none of which
    separate the candidates: the gate has already fixed all three, so 99% of them
    come from a gold report. Naming the line item puts the one open question in
    front of the model instead of leaving it to be inferred.
    """
    items = record.get("line_items") or []
    if not items:
        return record["question"]
    return f"{record['question']}\nChỉ tiêu cần tìm: {'; '.join(items)}"


def score_question(record, tokenizer, model, prefix_ids, suffix_ids, yes_id, no_id, budget):
    query = build_query(record)
    prompts = [
        f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {candidate['text']}"
        for candidate in record["candidates"]
    ]
    # A batch costs its longest member and candidates run from 300 to 2,800
    # characters, so group similar lengths and restore the original order after.
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
    scores = [0.0] * len(prompts)
    for start in range(0, len(order), BATCH_SIZE):
        chunk = order[start:start + BATCH_SIZE]
        encoded = tokenizer(
            [prompts[index] for index in chunk],
            truncation=True, max_length=budget, add_special_tokens=False,
        )["input_ids"]
        padded = tokenizer.pad(
            {"input_ids": [prefix_ids + ids + suffix_ids for ids in encoded]},
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**padded).logits[:, -1, :]
        # Qwen3-Reranker answers a yes/no question; the score is the probability
        # of "yes" at the final position, which is why padding must be on the left.
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        for index, value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            scores[index] = value
    return scores


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    print(f"{MODEL_NAME} loaded in 8-bit, max_length={MAX_LENGTH}", flush=True)

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = MAX_LENGTH - len(prefix_ids) - len(suffix_ids)

    records = load_pairs(PAIRS_PATH)
    done = already_scored(SCORES_PATH)
    if done:
        print(f"resuming, {len(done)} questions already scored", flush=True)
    pending = [record for record in records if record["id"] not in done]
    total = sum(len(record["candidates"]) for record in pending)
    print(f"{len(pending)} questions, {total} pairs", flush=True)

    started, scored = time.time(), 0
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            values = score_question(
                record, tokenizer, model, prefix_ids, suffix_ids, yes_id, no_id, budget,
            )
            out.write(json.dumps({
                "id": record["id"],
                "scores": {
                    candidate["table_id"]: round(value, 6)
                    for candidate, value in zip(record["candidates"], values)
                },
            }, ensure_ascii=False) + "\n")
            out.flush()
            scored += len(values)
            if scored % 1000 < BATCH_SIZE:
                rate = scored / (time.time() - started)
                print(f"{scored}/{total} pairs, {rate:.1f}/s, eta {(total - scored) / rate / 60:.0f} min", flush=True)

    print(f"wrote {SCORES_PATH} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
