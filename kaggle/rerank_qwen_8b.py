"""Score exported candidate pairs with a Qwen3 reranker; writes scores.jsonl."""

import json
import sys
import os
import time
import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
MODEL_REVISION = "77d193c791ed757ca307ee72715aa132723da912"
PAIRS_PATH = "/kaggle/input/vifinqa-rerank-pairs/pairs_bench_v4.jsonl"
SCORES_PATH = "/kaggle/working/scores.jsonl"
QUANTIZATION = "fp16"
RESUME_PATHS = []  # prior score shards: skips whole questions
SKIP_PATH = ""  # a finished scores.jsonl: skips individual candidates

if len(sys.argv) > 2 and not sys.argv[1].startswith("-"):
    PAIRS_PATH, SCORES_PATH = sys.argv[1], sys.argv[2]
    SKIP_PATH = sys.argv[3] if len(sys.argv) > 3 else ""

MAX_LENGTH = 1024
RERANK_DEPTH = int(os.environ.get("RERANK_DEPTH", "130"))
MAX_BATCH = 16
PROGRESS_INTERVAL = 1_000
PROGRESS_WIDTH = 30

INSTRUCTION = (
    "The Query asks about a Vietnamese listed company's financial statements. "
    "The Document is one table from a filing whose first line identifies its "
    "ticker, reporting period, and scope (hợp nhất or công ty mẹ). The candidate "
    "pool is already narrowed to compatible filings, but reject a Document when "
    "that filing line conflicts with the Query. Answer yes only when the table "
    "reports an exact requested accounting item with a value for the requested "
    "period. For a multi-step question, a table is yes when it supplies any one "
    "required input. Do not accept a merely related label, a reference to an "
    "item, a related-party or subsidiary listing, a movement or allocation "
    "schedule, or a column header. Prefer the statement or note that reports the "
    "figure; a note or segment breakdown that validly restates it is yes."
)

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256():
    text = json.dumps(
        {"instruction": INSTRUCTION, "prefix": PREFIX, "suffix": SUFFIX},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pairs(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def already_scored(path):
    if not path or not os.path.exists(path):
        return set()
    return {record["id"] for record in load_pairs(path)}


def already_judged(path):
    if not path or not os.path.exists(path):
        return {}
    judged = {}
    for record in load_pairs(path):
        judged.setdefault(record["id"], set()).update(record["scores"])
    return judged


def to_judge(record, skip=frozenset()):
    judged = []
    for index, candidate in enumerate(record["candidates"]):
        rank = candidate.get("sparse_rank") or candidate.get("dense_rank") or index + 1
        if rank <= RERANK_DEPTH and candidate["table_id"] not in skip:
            judged.append(index)
    return judged


def build_queries(record):
    return [record["question"]]


def pair_count(record, skip=frozenset()):
    return len(to_judge(record, skip)) * len(build_queries(record))


def progress_bar(completed, total, started):
    fraction = completed / total if total else 1.0
    filled = round(PROGRESS_WIDTH * fraction)
    elapsed = max(time.time() - started, 1e-9)
    rate = completed / elapsed
    remaining_minutes = (total - completed) / rate / 60 if rate else 0.0
    return (
        f"[{'#' * filled}{'-' * (PROGRESS_WIDTH - filled)}] "
        f"{completed}/{total} ({fraction:.1%}) | {rate:.1f} pairs/s | "
        f"ETA {remaining_minutes:.0f} min"
    )


def packed_batches(lengths, order):
    budget = 8 * MAX_LENGTH
    batch = []
    for index in order:
        width = max([lengths[index]] + [lengths[i] for i in batch])
        if batch and (width * (len(batch) + 1) > budget or len(batch) >= MAX_BATCH):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def last_position_logits(model, padded):
    """Logits for the final position only, to avoid a full-sequence lm_head tensor."""
    with torch.inference_mode():
        hidden = model.model(**padded, use_cache=False).last_hidden_state[:, -1, :]
        return model.lm_head(hidden).float()


def score_question(record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skip=frozenset()):
    queries = build_queries(record)
    candidates = record["candidates"]
    scores = [0.0] * len(candidates)
    judged = to_judge(record, skip)
    if not judged:
        return scores, judged
    rows = [(index, position) for index in judged for position in range(len(queries))]
    prefix_ids, suffix_ids = prompt_ids
    encoded = {
        row: prefix_ids + ids + suffix_ids
        for row, ids in zip(rows, tokenizer(
            [
                f"<Instruct>: {INSTRUCTION}\n<Query>: {queries[position]}"
                f"\n<Document>: {candidates[index]['text']}"
                for index, position in rows
            ],
            truncation=True, max_length=budget, add_special_tokens=False,
        )["input_ids"])
    }
    lengths = {row: len(ids) for row, ids in encoded.items()}
    pending = list(packed_batches(lengths, sorted(rows, key=lambda row: lengths[row])))
    while pending:
        chunk = pending.pop()
        padded = tokenizer.pad(
            {"input_ids": [encoded[row] for row in chunk]}, padding=True, return_tensors="pt",
        ).to(model.device)
        try:
            logits = last_position_logits(model, padded)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            del padded
            torch.cuda.empty_cache()
            middle = len(chunk) // 2
            pending.extend([chunk[:middle], chunk[middle:]])
            continue
        # P("yes") at the final position (hence left padding)
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        for (index, position), value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            scores[index] = max(scores[index], value)
    return scores, judged


def score_payload(record, values, judged):
    """Only judged candidates are written, since 0 is a real score."""
    kept = [(index, record["candidates"][index]) for index in judged]
    return {
        "id": record["id"],
        "scores": {candidate["table_id"]: round(values[index], 6) for index, candidate in kept},
    }


def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa", revision=MODEL_REVISION,
    ).eval()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no GPU available")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", revision=MODEL_REVISION,
    )
    model = load_model()
    print(f"{MODEL_NAME} in {QUANTIZATION}, depth={RERANK_DEPTH}", flush=True)

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prompt_ids = (
        tokenizer.encode(PREFIX, add_special_tokens=False),
        tokenizer.encode(SUFFIX, add_special_tokens=False),
    )
    budget = MAX_LENGTH - sum(len(ids) for ids in prompt_ids)

    records = load_pairs(PAIRS_PATH)
    done = already_scored(SCORES_PATH)
    for path in RESUME_PATHS:
        done |= already_scored(path)
    skip = already_judged(SKIP_PATH)
    skipped = lambda record: skip.get(record["id"], frozenset())
    pending = [record for record in records if record["id"] not in done]
    total = sum(pair_count(record, skipped(record)) for record in pending)
    print(f"{len(pending)} questions, {total} pairs"
          + (f", resuming past {len(done)}" if done else "")
          + (f", skipping {sum(len(s) for s in skip.values())} judged pairs" if skip else ""), flush=True)

    started, scored = time.time(), 0
    next_progress = PROGRESS_INTERVAL
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            values, judged = score_question(
                record, tokenizer, model, prompt_ids, yes_id, no_id, budget, skipped(record),
            )
            out.write(json.dumps(
                score_payload(record, values, judged), ensure_ascii=False,
            ) + "\n")
            out.flush()
            scored += pair_count(record, skipped(record))
            if scored >= next_progress or scored == total:
                print(progress_bar(scored, total, started), flush=True)
                next_progress += PROGRESS_INTERVAL

    elapsed = time.time() - started
    manifest = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "quantization": QUANTIZATION,
        "max_length": MAX_LENGTH,
        "rerank_depth": RERANK_DEPTH,
        "max_batch": MAX_BATCH,
        "prompt_sha256": prompt_sha256(),
        "pairs_sha256": sha256(PAIRS_PATH),
        "scores_sha256": sha256(SCORES_PATH),
        "resume_sha256": [sha256(path) for path in RESUME_PATHS],
        "skip_sha256": sha256(SKIP_PATH)
        if SKIP_PATH and os.path.exists(SKIP_PATH) else None,
        "questions_in_pairs": len(records),
        "questions_scored_this_run": len(pending),
        "pairs_scored_this_run": scored,
        "elapsed_seconds": round(elapsed, 3),
        "gpu": [torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())],
        "torch": torch.__version__,
    }
    manifest_path = SCORES_PATH + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"wrote {SCORES_PATH} and {manifest_path} in {elapsed / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
