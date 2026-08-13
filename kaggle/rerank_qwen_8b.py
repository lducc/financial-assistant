"""Kaggle GPU notebook: score candidate pairs with Qwen3-Reranker-8B.

Same contract as `rerank_qwen_4b.py` — reads pairs_v3.jsonl, writes scores.jsonl,
changes nothing about retrieval — but 8B does not fit a single T4 in fp16, so the
memory strategy is the difference between the two files:

* **T4 x2 (default).** `device_map="auto"` splits the layers across both 16 GB
  cards. The cards run in sequence, not in parallel, so this buys capacity rather
  than speed, and sharding the data is not available — both GPUs are already
  holding one model.
* **4-bit on a single card** (`LOAD_4BIT=1`). NF4 quantization puts 8B in about
  6 GB, which frees the second T4 to run a second shard. Two shards at 4-bit
  usually finish sooner than one unquantized model across both cards, at some
  cost in fidelity.

Runtime for 50,335 pairs at max_length 512: roughly 6-9 h split across two T4s,
or 3-5 h as two 4-bit shards. Both exceed a single session at the slow end, and
that is fine — scores append per question and finished IDs are skipped, so rerun
the cell and it continues.

Whether 8B is worth it over 4B is an empirical question, not a given: measure
both on the benchmark before spending a submission. 4B already returned +0.0355
F2 there, and the gap between sizes on a reranking task is usually smaller than
the gap between a right and wrong model family.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.environ.get("QWEN_RERANKER", "Qwen/Qwen3-Reranker-8B")
PAIRS_PATH = os.environ.get("PAIRS_PATH", "/kaggle/input/vifinqa-rerank-pairs/pairs_v3.jsonl")
SHARD = int(os.environ.get("SHARD", "0"))
SHARDS = int(os.environ.get("SHARDS", "1"))
SCORES_PATH = os.environ.get("SCORES_PATH", f"/kaggle/working/scores_8b_{SHARD}.jsonl")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
# 8B activations are twice 4B's, so keep batches smaller to stay inside 16 GB.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
LOAD_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"

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
        records = [json.loads(line) for line in handle if line.strip()]
    return [record for index, record in enumerate(records) if index % SHARDS == SHARD]


def already_scored(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def load_model():
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if LOAD_4BIT:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        # Pin the whole model to one card so the other is free for a second shard.
        kwargs["device_map"] = {"": 0}
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs).eval()


def score_question(record, tokenizer, model, prefix_ids, suffix_ids, yes_id, no_id, budget):
    prompts = [
        f"<Instruct>: {INSTRUCTION}\n<Query>: {record['question']}\n<Document>: {candidate['text']}"
        for candidate in record["candidates"]
    ]
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
        for index, value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            scores[index] = value
    return scores


def main():
    if not torch.cuda.is_available():
        raise SystemExit("8B needs a GPU; select T4 x2 in the notebook accelerator settings")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = load_model()
    print(
        f"model={MODEL_NAME} 4bit={LOAD_4BIT} gpus={torch.cuda.device_count()} "
        f"shard={SHARD}/{SHARDS} max_length={MAX_LENGTH}", flush=True,
    )

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
    print(f"{len(pending)} questions, {total} pairs in this shard", flush=True)

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
