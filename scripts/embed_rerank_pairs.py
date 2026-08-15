#!/usr/bin/env python3
"""Cache E5 embeddings for the reranker's own candidate texts.

DualView (arXiv 2605.18767) reports an 11M-parameter head over frozen E5
embeddings beating 560M cross-encoders on multi-hop document reranking. Its own
ablation is the reason to try the idea here: a plain 0.6M MLP over those cached
embeddings takes MuSiQue Recall@4 from 71.9 to 98.4, and the whole dual-view
architecture adds 1.0 on top. The win is supervised training on cached
embeddings, not the architecture — so that is the rung to test first, and it is
cheap enough to run on the laptop GPU rather than a Kaggle session.

Embedding the `pairs_v4.jsonl` candidate text rather than re-deriving text from
the tables keeps the comparison honest: the learned head and the 8B cross-encoder
then see exactly the same string, so a difference between them is the model and
not the representation.

Texts repeat across questions — 43,891 unique of 50,335 — so they are embedded
once and shared.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_NAME = "intfloat/multilingual-e5-base"


def encode(texts: list[str], model, tokenizer, device, batch_size: int, max_length: int) -> np.ndarray:
    import torch

    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    # Length-sorted batches so padding is not paid for twice; the inverse
    # permutation puts rows back in the caller's order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            batch = tokenizer(
                [texts[i] for i in chunk], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            # Mean pooling over real tokens, then L2 norm — the E5 recipe.
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out[chunk] = pooled.float().cpu().numpy()
            if (start // batch_size) % 50 == 0:
                print(f"  embedded {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "output" / "rerank" / "pairs_v4.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "dense")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    records = [
        json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    # E5 wants the asymmetric prefixes; without them the two sides land in
    # different regions of the space and cosine similarity is meaningless.
    queries = {r["id"]: "query: " + r["question"] for r in records}
    texts, index = [], {}
    for record in records:
        for candidate in record["candidates"]:
            body = "passage: " + candidate["text"]
            if body not in index:
                index[body] = len(texts)
                texts.append(body)
    print(f"{len(records)} questions, {len(texts)} unique candidate texts", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)
    if device == "cuda":
        model = model.half()
    print(f"{MODEL_NAME} on {device}", flush=True)

    query_ids = sorted(queries)
    query_matrix = encode([queries[i] for i in query_ids], model, tokenizer, device,
                          args.batch_size, args.max_length)
    print("queries done", flush=True)
    passage_matrix = encode(texts, model, tokenizer, device, args.batch_size, args.max_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "queries.npy", query_matrix)
    np.save(args.output_dir / "passages.npy", passage_matrix)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "max_length": args.max_length,
        "query_ids": query_ids,
        "passage_index": {table_id: index["passage: " + text] for table_id, text in (
            (c["table_id"], c["text"]) for r in records for c in r["candidates"]
        )},
    }, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "queries": query_matrix.shape, "passages": passage_matrix.shape,
        "output": str(args.output_dir),
    }, default=list, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
