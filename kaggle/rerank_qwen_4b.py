"""Kaggle GPU notebook: score candidate pairs with Qwen3-Reranker-4B.

The model that worked. On the first export it gave +0.0355 F2 on the benchmark
and +0.023 live, against BGE-reranker-v2-m3 losing 0.15 — a general web-passage
reranker is wrong for Vietnamese financial tables, this family is not.

Reads pairs_v4.jsonl and writes scores.jsonl, which `apply_rerank_scores.py`
fuses locally. Retrieval is untouched: only the order of already-retrieved
candidates changes, so a bad run is discarded by deleting one file.

Candidates are sorted by length before batching, since a batch costs its longest
member and they run from 300 to 2,800 characters — worth 25-40% of the runtime.
Scores append per question and finished IDs are skipped, so a session that hits
the 12-hour limit is rerun rather than restarted.

Runtime for 50,335 pairs at max_length 1024: roughly 6-8 h on a T4.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs_v4.jsonl"
SCORES_PATH = "/kaggle/working/scores.jsonl"
# 16% of candidates run past 320 tokens and the line-item inventory sits early in
# the text. 1024 covers the whole inventory for all but the widest tables, so
# nothing that decides the ranking is cut; it is what the scored runs used.
MAX_LENGTH = 1024
BATCH_SIZE = 16

INSTRUCTION = (
    "Every candidate is a table from the correct company, period and statement, "
    "so none of those separate them — the whole judgement is which table inside "
    "that report reports the figure. Answer yes if the table holds at least one "
    "of the line items under 'Chỉ tiêu cần tìm' as a row of its own, with a "
    "value for the period asked; a question often needs several figures and a "
    "table only has to supply one of them. A matching label is not enough on "
    "its own: the same wording appears on rows that only reference the item — "
    "related-party and subsidiary listings, movement and allocation schedules, "
    "and rows that are column headers rather than line items. Prefer the "
    "statement or the note that reports the figure, and count a note or segment "
    "breakdown restating it as yes."
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
    # A batch costs its longest member, so group similar lengths together and put
    # the scores back in the original order afterwards.
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
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        probabilities = torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()
        for index, value in zip(chunk, probabilities):
            scores[index] = value
    return scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    ).eval()
    print(f"device={device} model={MODEL_NAME} max_length={MAX_LENGTH}", flush=True)

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
            if scored % 2000 < BATCH_SIZE:
                rate = scored / (time.time() - started)
                print(f"{scored}/{total} pairs, {rate:.1f}/s, eta {(total - scored) / rate / 60:.0f} min", flush=True)

    print(f"wrote {SCORES_PATH} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
