# Embed the table corpus and the questions with Qwen3-Embedding.
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
TABLES_PATH = "/kaggle/input/vifinqa-dense/tables.jsonl"
QUESTIONS_PATH = "/kaggle/input/vifinqa-dense/questions.jsonl"
OUTPUT_DIR = "/kaggle/working/dense"

MAX_LENGTH = 256      # p90 of a passage; the tail is period and unit boilerplate
BATCH_SIZE = 32
DIMS = 1024           # Matryoshka truncation; the full 2560 would be 750 MB
LIMIT = 0             # nonzero for a smoke test

# Qwen3-Embedding is instruction-tuned on the query side only; documents are embedded bare
INSTRUCT = ("Given a Vietnamese financial question, retrieve the financial "
            "statement table that contains the answer")


def tokenize(texts, tokenizer):
    end = tokenizer.eos_token_id
    encoded = [tokenizer(text, truncation=True, max_length=MAX_LENGTH - 1,
                         add_special_tokens=True)["input_ids"] for text in texts]
    encoded = [ids if ids and ids[-1] == end else ids + [end] for ids in encoded]
    return tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")


# the last position must hold the EOS token that pooling reads
def encode(texts, model, tokenizer, label):
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenize(texts[start:start + BATCH_SIZE], tokenizer).to(model.device)
        with torch.inference_mode():
            # left padding: the final column is every row's last real token
            pooled = model(**batch).last_hidden_state[:, -1].float()
        # truncate before normalising; a Matryoshka prefix is not unit length
        pooled = torch.nn.functional.normalize(pooled[:, :DIMS], p=2, dim=1)
        vectors.append(pooled.half().cpu().numpy())
        done = min(start + BATCH_SIZE, len(texts))
        if (start // BATCH_SIZE) % 50 == 0 or done == len(texts):
            print(f"{label} {done}/{len(texts)}", flush=True)
    return np.concatenate(vectors)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def self_check(tables, questions):
    sample = tables[:20000].astype(np.float32)
    scores = sample @ questions[:50].astype(np.float32).T
    best, average = scores.max(axis=0).mean(), scores.mean()
    print(f"self-check: best cosine {best:.3f}, random pair {average:.3f}, "
          f"spread {best - average:.3f}", flush=True)
    if best - average < 0.05:
        print("WARNING: the best match is no better than an average one. "
              "Do not build on this index.", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", revision=MODEL_REVISION,
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map={"": 0},
        revision=MODEL_REVISION,
    ).eval()

    table_vectors = None
    if TABLES_PATH:
        tables = read_jsonl(TABLES_PATH)[: LIMIT or None]
        print(f"{len(tables)} passages", flush=True)
        table_vectors = encode([row["text"] for row in tables], model, tokenizer, "tables")
        np.save(f"{OUTPUT_DIR}/table_vectors.npy", table_vectors)
        with open(f"{OUTPUT_DIR}/table_ids.json", "w", encoding="utf-8") as handle:
            json.dump([[row["report_id"], row["table_id"]] for row in tables], handle)

    questions = read_jsonl(QUESTIONS_PATH)
    queries = [f"Instruct: {INSTRUCT}\nQuery: {row['question']}" for row in questions]
    question_vectors = encode(queries, model, tokenizer, "questions")
    np.save(f"{OUTPUT_DIR}/question_vectors.npy", question_vectors)
    with open(f"{OUTPUT_DIR}/question_ids.json", "w", encoding="utf-8") as handle:
        json.dump([row["id"] for row in questions], handle)

    if table_vectors is not None:
        self_check(table_vectors, question_vectors)
    print("done", OUTPUT_DIR, flush=True)


main()
