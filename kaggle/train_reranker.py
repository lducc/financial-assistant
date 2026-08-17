"""LoRA fine-tune Qwen3-Reranker on our own labels, on a Kaggle GPU.

Why this and not another zero-shot idea: an oracle reordering of the candidates
retrieval already returns scores 0.8020 benchmark F2 against 0.6549 shipped, so
0.147 sits inside the ranking stage. Every zero-shot lever around it is spent —
the budget is at its measured optimum, reallocating slots across reports is worth
+0.0044 with the interval covering zero, depth 100 is worth +0.0100, and listwise
generation measured -0.0458. Fine-tuning is the one lever with the headroom still
in front of it, and the organizers permit it.

Training data comes from `scripts/export_rerank_training.py`, which reads
`annotations/train/accepted.jsonl` — 311 questions discovered by folded row-label
search over raw OCR with no retriever in the loop, and disjoint from the 233
benchmark questions used to score the result. That disjointness is the whole
experiment: train on one set, measure on the other, or the number means nothing.

The scoring function is unchanged. Inference reads the probability of "yes" at
the final position, so training optimises exactly that quantity rather than a
separate classification head that would have to be reconciled with it later.

## Steps

1. Create a Kaggle Dataset holding `training.jsonl` (about 6 MB) and set
   `TRAINING_PATH`.
2. New notebook, Accelerator **GPU T4 x2** or **P100**, attach the dataset, paste
   this file into one cell, run.
3. Download `/kaggle/working/adapter/` and keep it next to the repo.
4. Re-score the full pool with the adapter loaded (see `rerank_qwen_8b.py`, which
   takes `ADAPTER_PATH`), then fuse and measure on the benchmark:

       python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_v4.jsonl \
           --scores scores_tuned.jsonl --output output/rerank/ranking_tuned.json
       python3 scripts/compare_rankings.py \
           --baseline output/rerank/ranking_v4_fuse.json \
           --candidate output/rerank/ranking_tuned.json

   Accept only on the standing rule: mean up, CI excluding zero, no tier down.

## Model choice

4B fp16 leaves room for activations on a 16 GB T4 and trains in roughly two
hours; 8B needs 4-bit weights and gradient checkpointing and takes most of a
session. 8B zero-shot beats 4B zero-shot (MRR@5 0.8190 against 0.7937), but a
tuned 4B against an untuned 8B is an open question and 4B is the cheaper way to
find out whether tuning helps at all. Start at 4B. If it moves the benchmark,
repeat at 8B.
"""

import json
import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
TRAINING_PATH = "/kaggle/input/vifinqa-rerank-training/training_corpus.jsonl"
OUTPUT_DIR = "/kaggle/working/adapter"
# Held-out questions for a loss curve that means something. Groups are split by
# question, never by row: two rows from one question in different splits would
# leak the document text across the boundary.
VALIDATION_SHARE = 0.15
MAX_LENGTH = 1024
EPOCHS = 2
LEARNING_RATE = 1e-4
# One optimiser step covers one question: its positives against its hard
# negatives. The loss is a softmax over the candidates of a single question, so
# the group has to arrive intact.
GROUPS_PER_STEP = 1
MAX_GROUP = 12
MAX_GROUPS = 4000  # 0 keeps them all; a group is one optimiser step, so this is the session budget
LORA_RANK = 16
QUANTIZATION = "fp16"  # "fp16" for 4B, "nf4" for 8B
SEED = 20260814

# Identical to rerank_qwen_8b.py. If either copy changes, both must.
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


def load_groups(path):
    """Rows regrouped by (question, query), which is the unit the loss ranks over."""
    grouped = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped.setdefault((row["id"], row["query"]), []).append(row)
    groups = []
    for (identifier, query), rows in grouped.items():
        positives = [row for row in rows if row["label"] == 1]
        negatives = [row for row in rows if row["label"] == 0]
        if not positives or not negatives:
            # A group with only one class has no ordering to teach.
            continue
        groups.append({"id": identifier, "query": query, "positives": positives, "negatives": negatives})
    return groups


def split_by_question(groups, share, rng):
    """Hold out whole questions, so no document crosses the split."""
    identifiers = sorted({group["id"] for group in groups})
    rng.shuffle(identifiers)
    held = set(identifiers[: max(1, round(share * len(identifiers)))])
    return (
        [group for group in groups if group["id"] not in held],
        [group for group in groups if group["id"] in held],
    )


def encode(tokenizer, query, document):
    body = tokenizer(
        f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}",
        truncation=True, max_length=MAX_LENGTH, add_special_tokens=False,
    )["input_ids"]
    return tokenizer.encode(PREFIX, add_special_tokens=False) + body + tokenizer.encode(
        SUFFIX, add_special_tokens=False
    )


def group_scores(model, tokenizer, group, yes_id, no_id, rng, train=True):
    """Score one question's candidates, returning the log-odds of "yes" for each.

    Left padding, because the score is read at the final position and right
    padding would read the pad instead.
    """
    positives = group["positives"] if not train else [rng.choice(group["positives"])]
    negatives = group["negatives"][: MAX_GROUP - len(positives)]
    rows = positives + negatives
    encoded = [torch.tensor(encode(tokenizer, group["query"], row["document"])) for row in rows]
    flipped = pad_sequence(
        [ids.flip(0) for ids in encoded], batch_first=True, padding_value=tokenizer.pad_token_id
    ).flip(1)
    attention = pad_sequence(
        [torch.ones_like(ids).flip(0) for ids in encoded], batch_first=True, padding_value=0
    ).flip(1)
    hidden = model.model(
        input_ids=flipped.to(model.device), attention_mask=attention.to(model.device), use_cache=False,
    ).last_hidden_state[:, -1, :]
    logits = model.lm_head(hidden).float()
    # The margin between "yes" and "no" is the quantity inference ranks on, so it
    # is the quantity the loss operates on. Training a separate head would leave
    # scoring and training optimising two different functions.
    return logits[:, yes_id] - logits[:, no_id], len(positives)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("select a GPU accelerator in the notebook settings")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    groups = load_groups(TRAINING_PATH)
    if MAX_GROUPS:
        rng.shuffle(groups)
        groups = groups[:MAX_GROUPS]
    train_groups, validation_groups = split_by_question(groups, VALIDATION_SHARE, rng)
    print(
        f"{len(groups)} groups over {len({g['id'] for g in groups})} questions; "
        f"{len(train_groups)} train / {len(validation_groups)} validation",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    config = None
    if QUANTIZATION == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=config, torch_dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    )

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.0
    )
    steps = EPOCHS * math.ceil(len(train_groups) / GROUPS_PER_STEP)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=steps, pct_start=0.1
    )

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for group in validation_groups:
                scores, positive_count = group_scores(
                    model, tokenizer, group, yes_id, no_id, rng, train=False
                )
                # The only thing that matters downstream is whether a gold table
                # outranks the hard negatives, so that is what is reported.
                correct += int(scores.argmax().item() < positive_count)
                total += 1
        model.train()
        return correct / max(1, total)

    print(f"validation top-1 before training: {evaluate():.4f}", flush=True)
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        rng.shuffle(train_groups)
        running = 0.0
        for index, group in enumerate(train_groups, 1):
            scores, positive_count = group_scores(model, tokenizer, group, yes_id, no_id, rng)
            # Softmax over one question's candidates: the positive has to beat the
            # negatives it is actually shown beside, which is the comparison the
            # submission budget makes.
            loss = torch.nn.functional.cross_entropy(
                scores.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=scores.device)
            )
            loss.backward()
            running += loss.item()
            if index % GROUPS_PER_STEP == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            if index % 50 == 0:
                print(f"epoch {epoch + 1} {index}/{len(train_groups)} loss {running / 50:.4f}", flush=True)
                running = 0.0
        print(f"epoch {epoch + 1} validation top-1: {evaluate():.4f}", flush=True)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"adapter written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
