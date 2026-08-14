"""Kaggle GPU notebook: score candidate pairs with Qwen3-Reranker-8B in 8-bit.

Paste into a notebook with the accelerator set to GPU and the exported pairs
attached as a dataset, then run. It writes scores.jsonl for
`scripts/apply_rerank_scores.py` to fuse locally. Every path is also readable
from the environment, so the same file runs unchanged on a rented GPU.

8.19B parameters is 16.4 GB in fp16, which does not fit a 16 GB T4, so the model
loads in 8-bit at about 8.2 GB — one card, one process, near-lossless. That costs
speed: bitsandbytes computes outlier features in higher precision, roughly 1.5-2x
fp16 on Turing. Measured on a T4, 739 tokens/s: 2.3 h for the 233-question
benchmark export, 10.2 h for all 1,012 questions. Scores append per question and
finished IDs are skipped, so a session that hits the 12 h limit is rerun rather
than restarted; across sessions, point RESUME_PATH at the downloaded scores.

Retrieval is untouched. Only the order of already-retrieved candidates changes,
so discarding the run means deleting one file.
"""

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
# Run 1 of 3. Then pairs_bench_d100.jsonl, then pairs_v4.jsonl — renaming
# scores.jsonl after each, since the resume skips IDs it has already written and
# the two benchmark files cover the same 233 questions.
PAIRS_PATH = os.environ.get(
    "PAIRS_PATH", "/kaggle/input/vifinqa-rerank-pairs/pairs_bench_v4.jsonl",
)
SCORES_PATH = os.environ.get("SCORES_PATH", "/kaggle/working/scores.jsonl")
# Kaggle caps a session at 12 h and wipes /kaggle/working between them, so a run
# longer than that resumes by hand: download scores.jsonl, upload it as a
# dataset, and name it here. Questions already in it are skipped, and the new
# session writes only the rest — concatenate the two files afterwards.
RESUME_PATH = os.environ.get("RESUME_PATH")  # e.g. a downloaded scores.jsonl
# 16% of candidates run past 320 tokens, and the line-item inventory that tells
# the model what a table holds sits early in the text. 1024 covers the whole
# inventory for all but the widest tables, so nothing that decides the ranking is
# cut; it is what the scored runs used.
MAX_LENGTH = 1024
# Batches are packed to a token budget rather than a fixed width. Candidates are
# already length-sorted, so this saves almost no padding — measured at 6.3M
# padded tokens either way — but it widens a batch from 8 to roughly 24 rows at
# the same arithmetic, which is worth having on a card this size. Keeping only
# the final position's logits is what makes a budget this large fit.
TOKEN_BUDGET = 8 * MAX_LENGTH
MAX_BATCH = 16
# How deep into the sparse order to judge, set above the deepest export so
# nothing is skipped by accident: pairs_v4 and pairs_bench_v4 stop at rank 50,
# pairs_bench_d100 goes to 100. Lowering it is the one knob that trades quality
# for time — gold sits at rank 1-10 for 69% of tables and past 30 for only 8%, so
# depth 30 cuts the run by 40% and costs 0.006 MRR@5 and 0.008 recall@5 (hard
# 0.694 -> 0.682), measured through the real fusion path. Small, but every number
# moves the wrong way, so nothing is dropped unless a session has to finish.
RERANK_DEPTH = 100
# int8 is the setting that has been measured here, and it stays. The task turns
# on separating labels that differ by one word, where the yes-probabilities sit
# close together, and that is the first thing coarser quantization blurs. "nf4"
# is faster on Turing and "fp16" is both faster and exact but needs a T4 x2 to
# hold 16.4 GB; neither has been scored, so neither is the default.
QUANTIZATION = os.environ.get("QUANTIZATION", "int8")

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


def packed_batches(lengths, order):
    """Groups of length-sorted rows, each costing at most TOKEN_BUDGET tokens.

    Padding makes a batch cost its widest row times its width, so packing to a
    token budget keeps every batch about the same size in work while letting the
    short candidates run many at a time.
    """
    batch = []
    for index in order:
        width = max([lengths[index]] + [lengths[i] for i in batch])
        if batch and (width * (len(batch) + 1) > TOKEN_BUDGET or len(batch) >= MAX_BATCH):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def last_position_logits(model, padded):
    """Vocabulary logits for the final position only.

    Running the model normally projects 4096 -> 151,936 at every position, which
    at batch 24 and a 1024 window is a 7.5 GB tensor built to read 48 numbers,
    and it is what put a 16 GB T4 out of memory. Calling the body and applying
    the head to the last hidden state alone gives the identical numbers. The
    `logits_to_keep` argument does the same thing on new transformers but is
    silently ignored by older ones, so the head is applied here instead.
    """
    with torch.inference_mode():
        hidden = model.model(**padded, use_cache=False).last_hidden_state[:, -1, :]
        return model.lm_head(hidden).float()


def score_question(record, tokenizer, model, prefix_ids, suffix_ids, yes_id, no_id, budget):
    query = build_query(record)
    candidates = record["candidates"]
    scores = [0.0] * len(candidates)
    # Candidates arrive in sparse rank order, and the tail is scored so rarely
    # that it is cheaper to leave it to the sparse ranking than to judge it.
    scored = [i for i, candidate in enumerate(candidates)
              if candidate.get("sparse_rank", i + 1) <= RERANK_DEPTH]
    prompts = {
        index: f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {candidates[index]['text']}"
        for index in scored
    }
    encoded = {
        index: prefix_ids + ids + suffix_ids
        for index, ids in zip(scored, tokenizer(
            [prompts[index] for index in scored],
            truncation=True, max_length=budget, add_special_tokens=False,
        )["input_ids"])
    }
    lengths = {index: len(ids) for index, ids in encoded.items()}
    order = sorted(scored, key=lambda index: lengths[index])
    pending = list(packed_batches(lengths, order))
    while pending:
        chunk = pending.pop()
        padded = tokenizer.pad(
            {"input_ids": [encoded[index] for index in chunk]},
            padding=True, return_tensors="pt",
        ).to(model.device)
        try:
            logits = last_position_logits(model, padded)
        except torch.cuda.OutOfMemoryError:
            # Halve and retry rather than lose the run: memory depends on the
            # widest row, so one unlucky batch should not end a 10 hour session.
            if len(chunk) == 1:
                raise
            del padded
            torch.cuda.empty_cache()
            middle = len(chunk) // 2
            pending.extend([chunk[:middle], chunk[middle:]])
            continue
        # Qwen3-Reranker answers a yes/no question; the score is the probability
        # of "yes" at the final position, which is why padding must be on the left.
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        for index, value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            scores[index] = value
    return scores, len(scored)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if QUANTIZATION == "fp16":
        # 16.4 GB does not fit one 16 GB T4, but it fits across two, and layer
        # splitting costs only a transfer at each boundary. That buys back more
        # than it costs: bitsandbytes int8 computes outlier features in higher
        # precision and runs 1.5-2x slower than fp16 on Turing. Faster and
        # lossless, so it needs GPU T4 x2 rather than P100.
        config, placement = None, "auto"
    elif QUANTIZATION == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        placement = {"": 0}
    else:
        config, placement = BitsAndBytesConfig(load_in_8bit=True), {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=config,
        torch_dtype=torch.float16,
        device_map=placement,
        # Turing has no flash-attention 2, but SDPA still beats the eager path.
        attn_implementation="sdpa",
    ).eval()
    print(f"{MODEL_NAME} loaded in {QUANTIZATION}, max_length={MAX_LENGTH}, depth={RERANK_DEPTH}", flush=True)

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = MAX_LENGTH - len(prefix_ids) - len(suffix_ids)

    records = load_pairs(PAIRS_PATH)
    done = already_scored(SCORES_PATH) | (already_scored(RESUME_PATH) if RESUME_PATH else set())
    if done:
        print(f"resuming, {len(done)} questions already scored", flush=True)
    pending = [record for record in records if record["id"] not in done]
    total = sum(
        sum(1 for index, candidate in enumerate(record["candidates"])
            if candidate.get("sparse_rank", index + 1) <= RERANK_DEPTH)
        for record in pending
    )
    print(f"{len(pending)} questions, {total} pairs to score", flush=True)

    started, scored = time.time(), 0
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            values, judged = score_question(
                record, tokenizer, model, prefix_ids, suffix_ids, yes_id, no_id, budget,
            )
            # Candidates past RERANK_DEPTH are left out rather than written as
            # zero: a zero is a score, and would rank them below everything the
            # model disliked instead of leaving them to the sparse order.
            out.write(json.dumps({
                "id": record["id"],
                "scores": {
                    candidate["table_id"]: round(value, 6)
                    for index, (candidate, value) in enumerate(zip(record["candidates"], values))
                    if candidate.get("sparse_rank", index + 1) <= RERANK_DEPTH
                },
            }, ensure_ascii=False) + "\n")
            out.flush()
            scored += judged
            if scored % 1000 < MAX_BATCH:
                rate = scored / (time.time() - started)
                print(f"{scored}/{total} pairs, {rate:.1f}/s, eta {(total - scored) / rate / 60:.0f} min", flush=True)

    print(f"wrote {SCORES_PATH} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
