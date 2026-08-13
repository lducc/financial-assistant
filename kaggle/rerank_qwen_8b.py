"""Kaggle GPU notebook: score candidate pairs with Qwen3-Reranker-8B.

Same contract as `rerank_qwen_4b.py` — reads pairs_v3.jsonl, writes scores.jsonl,
changes nothing about retrieval — but 8B does not fit a single T4 in fp16, so the
memory strategy is the difference between the two files:

* **8-bit on one card** (`LOAD_8BIT=1`, the simplest). 8.19B parameters at int8
  is about 8.2 GB, which leaves a T4 plenty of headroom, and int8 is close to
  lossless. It is also the slowest: bitsandbytes decomposes outliers in higher
  precision, which costs roughly 1.5-2x fp16 on Turing.
* **4-bit on one card** (`LOAD_4BIT=1`). NF4 puts the model near 5.5 GB and runs
  closer to fp16 speed, at some fidelity cost.
* **fp16 across both cards** (the default). `device_map="auto"` splits the layers,
  because 16.4 GB of weights does not fit one 16 GB card. The cards run in
  sequence, so this buys capacity rather than speed and leaves no GPU free for a
  second shard.

Quantizing to one card is what makes both GPUs usable: pin one process per card
and give each half the questions. See the launcher at the bottom of this file —
Kaggle hands both T4s to a single session, so the parallelism has to happen
inside one notebook rather than across two.

Runtime for 50,335 pairs at max_length 512, as two shards: roughly 3-5 h at 4-bit,
5-8 h at 8-bit. One model across both cards is 6-9 h. The slower paths exceed a
12-hour session; scores append per question and finished IDs are skipped, so
rerunning the cell continues rather than restarts.

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
LOAD_8BIT = os.environ.get("LOAD_8BIT", "0") == "1"
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
    if LOAD_8BIT or LOAD_4BIT:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True)
            if LOAD_8BIT
            else BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        )
        # Quantized weights fit one card, so pin the model there and leave the
        # other GPU free for a second shard of the questions.
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
    precision = "8bit" if LOAD_8BIT else "4bit" if LOAD_4BIT else "fp16"
    print(
        f"model={MODEL_NAME} precision={precision} gpus={torch.cuda.device_count()} "
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


# --- Kaggle launcher -------------------------------------------------------
# Kaggle gives both T4s to one session, so run two processes here rather than two
# notebooks. Save this file as rerank_qwen_8b.py in a cell, then run:
#
#   import subprocess, os
#   env = {**os.environ, "LOAD_8BIT": "1", "SHARDS": "2",
#          "PAIRS_PATH": "/kaggle/input/vifinqa-rerank-pairs/pairs_v3.jsonl"}
#   jobs = [
#       subprocess.Popen(["python", "rerank_qwen_8b.py"],
#                        env={**env, "CUDA_VISIBLE_DEVICES": str(gpu), "SHARD": str(gpu)},
#                        stdout=open(f"/kaggle/working/log_{gpu}.txt", "w"),
#                        stderr=subprocess.STDOUT)
#       for gpu in (0, 1)
#   ]
#   for job in jobs:
#       job.wait()
#
# Each process sees one GPU as device 0, writes scores_8b_{SHARD}.jsonl, and the
# local step merges them with repeated --scores flags. Tail the logs from another
# cell to watch progress.
