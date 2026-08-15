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

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Reranker-8B")
# A LoRA adapter from kaggle/train_reranker.py, attached to the same base model.
# Left unset the run is zero-shot, which is every score file committed so far.
# The prompt is identical either way, so a tuned run stays comparable to them.
ADAPTER_PATH = os.environ.get("ADAPTER_PATH")
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
# A finished score file whose (question, table) pairs are already judged and must
# not be judged again. RESUME_PATH skips whole questions, which is what a stopped
# session needs; this skips individual candidates, which is what a deeper export
# needs — pairs_v4_d100.jsonl repeats all 50,335 pairs scores_v4.jsonl already
# holds, so without this a depth-100 run costs 90,726 pairs instead of 40,391.
# `apply_rerank_scores.py` unions the two files back together.
SKIP_PATH = os.environ.get("SKIP_PATH")
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
# is faster on Turing; "fp16" runs the released weights without the int8 outlier
# path, which is the largest source of the run-to-run drift measured in
# docs/ASSESSMENT.md §8, but it needs a T4 x2 to hold 16.4 GB and then runs as a
# pipeline across the two cards, so it is not simply the single-card 1.5-2x. It
# would also not make scoring reproducible: fp16 matmuls are not batch-invariant
# either. Neither has been scored, so neither is the default.
QUANTIZATION = os.environ.get("QUANTIZATION", "int8")
# Score each named line item as its own query and keep the best, instead of one
# query listing all of them. 550 of 1,012 questions name two or three items, and
# 66.5% of the gold tables for those carry only one — so the combined query asks
# the right table for figures it never held. Costs 1.77x the pairs on the full
# corpus, 88,864 against 50,335.
#
# arXiv 2606.08577 is why this is worth the GPU time after 566fd47 rejected
# decomposition at -0.0302: it finds decomposition harms at the retrieval stage
# through semantic dilution and helps at the reranking stage, and 566fd47 fused
# its sub-queries into retrieval. Same idea, different stage.
#
# Also writes the unreduced per-(item, candidate) matrix, which costs nothing
# here and is the only way to test a set-level coverage objective later.
PER_ITEM = os.environ.get("PER_ITEM", "0") == "1"

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


def already_judged(path):
    """Table IDs per question from a previous run, so a deeper export resumes."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        judged = {}
        for line in handle:
            if line.strip():
                record = json.loads(line)
                judged.setdefault(record["id"], set()).update(record["scores"])
        return judged


def to_judge(record, skip=frozenset()):
    """Candidate positions this run should score.

    Candidates arrive in sparse rank order, and the tail is scored so rarely that
    it is cheaper to leave it to the sparse ranking than to judge it.
    """
    return [
        index for index, candidate in enumerate(record["candidates"])
        if candidate.get("sparse_rank", index + 1) <= RERANK_DEPTH
        and candidate["table_id"] not in skip
    ]


def build_queries(record):
    """The queries to score each candidate against, best score winning.

    The question states the company, the year and the output unit, none of which
    separate the candidates: the gate has already fixed all three, so 99% of them
    come from a gold report. Naming the line item puts the one open question in
    front of the model instead of leaving it to be inferred.

    With PER_ITEM, each named item becomes its own query. 550 of 1,012 questions
    name two or three, and 66.5% of their gold tables carry only one of them, so
    a single query listing all of them asks a table for figures it was never
    going to hold. Scoring the items separately and keeping the best asks each
    table only what it can answer. It costs about 1.45x the pairs, that being the
    mean item count.
    """
    items = record.get("line_items") or []
    if not items:
        return [record["question"]]
    if PER_ITEM and len(items) > 1:
        return [f"{record['question']}\nChỉ tiêu cần tìm: {item}" for item in items]
    return [f"{record['question']}\nChỉ tiêu cần tìm: {'; '.join(items)}"]


def pair_count(record, skip=frozenset()):
    """Forward passes this record costs: one per (candidate, query)."""
    return len(to_judge(record, skip)) * len(build_queries(record))


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


def score_question(record, tokenizer, model, prefix_ids, suffix_ids, yes_id, no_id, budget,
                   skip=frozenset()):
    queries = build_queries(record)
    candidates = record["candidates"]
    scores = [0.0] * len(candidates)
    scored = to_judge(record, skip)
    # One row per (candidate, query), keyed by query position rather than by the
    # query text so two identical line items stay two rows. Rows are reduced back
    # to a per-candidate best after scoring, so the output format never changes
    # with PER_ITEM; the unreduced matrix is kept beside it because a set-level
    # coverage objective needs to know which item each score belongs to, and
    # recovering that later would cost another full pass on the GPU.
    matrix = {index: [0.0] * len(queries) for index in scored}
    rows = [(index, position) for index in scored for position in range(len(queries))]
    prompts = [
        f"<Instruct>: {INSTRUCTION}\n<Query>: {queries[position]}\n<Document>: {candidates[index]['text']}"
        for index, position in rows
    ]
    encoded = {
        row: prefix_ids + ids + suffix_ids
        for row, ids in zip(rows, tokenizer(
            prompts, truncation=True, max_length=budget, add_special_tokens=False,
        )["input_ids"])
    }
    lengths = {row: len(ids) for row, ids in encoded.items()}
    order = sorted(rows, key=lambda row: lengths[row])
    pending = list(packed_batches(lengths, order))
    while pending:
        chunk = pending.pop()
        padded = tokenizer.pad(
            {"input_ids": [encoded[row] for row in chunk]},
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
        # Best query wins: a table that holds one of several named items should
        # score on that item, not be averaged down by the ones it does not hold.
        # The reduction is max rather than the min or product the decomposition
        # literature uses, because our constraint is disjunctive — a table counts
        # if it supplies any one of the named items, not all of them.
        for (index, position), value in zip(chunk, torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist()):
            matrix[index][position] = value
            scores[index] = max(scores[index], value)
    return scores, matrix, scored


def score_payload(record, values, matrix, judged, per_item):
    """The scores.jsonl line for one question.

    Only the candidates this run judged are written. Everything else is left out
    rather than written as zero — a zero is a score, and would rank a candidate
    below everything the model disliked instead of leaving it to the sparse order
    or to the earlier file it was already scored in.
    """
    kept = [(index, record["candidates"][index]) for index in judged]
    payload = {
        "id": record["id"],
        "scores": {candidate["table_id"]: round(values[index], 6) for index, candidate in kept},
    }
    items = record.get("line_items") or []
    if per_item and items:
        # One list per candidate, positionally aligned with `line_items` — echoed
        # here so a consumer never has to re-derive it from the pairs file and
        # risk a different lexicon. Questions naming no item are scored on the
        # bare question and so have no matrix to write; there is nothing to
        # decompose, and there are 20 of them in 1,012.
        payload["line_items"] = items
        payload["per_item"] = {
            candidate["table_id"]: [round(value, 6) for value in matrix[index]]
            for index, candidate in kept
        }
    return payload


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if QUANTIZATION == "fp16":
        # 16.4 GB does not fit one 16 GB T4, but it fits across two. Accelerate
        # splits it by layer, so the cards run as a pipeline: one computes while
        # the other waits, and a hidden state crosses at each boundary. Against
        # that, bitsandbytes int8 computes outlier features in higher precision
        # and runs 1.5-2x slower than fp16 per card. Which way the two effects
        # net out on this workload is unmeasured. Needs GPU T4 x2, not P100.
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
    if ADAPTER_PATH:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH).eval()
    print(
        f"{MODEL_NAME} loaded in {QUANTIZATION}, max_length={MAX_LENGTH}, depth={RERANK_DEPTH}"
        + (f", adapter={ADAPTER_PATH}" if ADAPTER_PATH else ", zero-shot"),
        flush=True,
    )

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = MAX_LENGTH - len(prefix_ids) - len(suffix_ids)

    records = load_pairs(PAIRS_PATH)
    done = already_scored(SCORES_PATH) | (already_scored(RESUME_PATH) if RESUME_PATH else set())
    if done:
        print(f"resuming, {len(done)} questions already scored", flush=True)
    skip = already_judged(SKIP_PATH)
    if skip:
        print(f"skipping {sum(len(s) for s in skip.values())} pairs from {SKIP_PATH}", flush=True)
    pending = [record for record in records if record["id"] not in done]
    skipped = lambda record: skip.get(record["id"], frozenset())
    total = sum(pair_count(record, skipped(record)) for record in pending)
    print(f"{len(pending)} questions, {total} pairs to score, per_item={PER_ITEM}", flush=True)

    started, scored = time.time(), 0
    with open(SCORES_PATH, "a", encoding="utf-8") as out:
        for record in pending:
            values, matrix, judged = score_question(
                record, tokenizer, model, prefix_ids, suffix_ids, yes_id, no_id, budget,
                skipped(record),
            )
            payload = score_payload(record, values, matrix, judged, PER_ITEM)
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            out.flush()
            scored += pair_count(record, skipped(record))
            if scored % 1000 < MAX_BATCH:
                rate = scored / (time.time() - started)
                print(f"{scored}/{total} pairs, {rate:.1f}/s, eta {(total - scored) / rate / 60:.0f} min", flush=True)

    print(f"wrote {SCORES_PATH} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
