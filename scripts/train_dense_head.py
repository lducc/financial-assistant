#!/usr/bin/env python3
"""Train a small supervised head over frozen E5 embeddings, and score it honestly.

This is the rung DualView's own ablation says does the work: E5-Cosine 71.9 ->
E5-MLP 98.4 on MuSiQue Recall@4, with the full 11M dual-view architecture adding
1.0 on top. If a head this size gets anywhere near our 8B cross-encoder here, the
remaining plan changes; if it does not, the line closes for the cost of an
afternoon on the laptop GPU.

The split is the one that already exists and does not leak: train on
`annotations/train/accepted.jsonl` (312 questions, discovered by raw-OCR search
with no retriever in the loop) and evaluate on `annotations/benchmark.jsonl` (233
questions, disjoint, and the instrument that predicted our only live A/B within
0.005). The head never sees a benchmark question.

Scored against the same candidates, the same budget, and the same gold definition
as every other ranking this project has measured, so the comparison against the
8B cross-encoder is like for like.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import table_budget
from vifinqa.scoring import (
    cluster_bootstrap, clusters_for, gold_of, gold_tables_for, prefix_score,
)

ROOT = Path(__file__).resolve().parents[1]

SEED = 20260815


def build(records: list[dict], pairs: dict, manifest: dict, queries: np.ndarray, passages: np.ndarray):
    """One group per question: candidate embeddings, labels, and table ids."""
    query_row = {identifier: row for row, identifier in enumerate(manifest["query_ids"])}
    index = manifest["passage_index"]
    groups = []
    for record in records:
        identifier = record["id"]
        pool = pairs.get(identifier)
        if pool is None or identifier not in query_row:
            continue
        gold = set(gold_tables_for(record["annotation"], "binding"))
        rows, labels, ids = [], [], []
        for candidate in pool["candidates"]:
            table_id = candidate["table_id"]
            if table_id not in index:
                continue
            rows.append(index[table_id])
            labels.append(1.0 if table_id in gold else 0.0)
            ids.append(table_id)
        if not rows or not any(labels):
            continue
        groups.append({
            "id": identifier,
            "query": queries[query_row[identifier]],
            "passages": passages[np.array(rows)],
            "labels": np.array(labels, dtype=np.float32),
            "tables": ids,
        })
    return groups


class Head:
    """Interaction MLP over [q, d, q*d, |q-d|] — the standard bi-encoder head."""

    def __init__(self, dim: int, hidden: int, device):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        ).to(device)
        self.device = device

    def features(self, query, passages):
        import torch
        q = query.unsqueeze(0).expand_as(passages)
        return torch.cat([q, passages, q * passages, (q - passages).abs()], dim=-1)

    def scores(self, query, passages):
        return self.net(self.features(query, passages)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-dir", type=Path, default=ROOT / "output" / "dense")
    parser.add_argument("--pairs", type=Path, default=ROOT / "output" / "rerank" / "pairs_v4.jsonl")
    parser.add_argument("--train", type=Path, default=ROOT / "annotations" / "train" / "accepted.jsonl")
    parser.add_argument("--eval", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--traces", type=Path, default=ROOT / "output" / "diagnostics" / "traces.jsonl")
    parser.add_argument("--baseline", type=Path, default=ROOT / "output" / "rerank" / "ranking_v4_tiered.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "rerank" / "ranking_dense_head.json")
    args = parser.parse_args()

    import torch
    torch.manual_seed(SEED)

    manifest = json.loads((args.dense_dir / "manifest.json").read_text(encoding="utf-8"))
    queries = np.load(args.dense_dir / "queries.npy")
    passages = np.load(args.dense_dir / "passages.npy")
    pairs = {r["id"]: r for r in load_jsonl(args.pairs)}
    train_groups = build(load_jsonl(args.train), pairs, manifest, queries, passages)
    eval_records = load_jsonl(args.eval)
    eval_groups = build(eval_records, pairs, manifest, queries, passages)
    assert not ({g["id"] for g in train_groups} & {g["id"] for g in eval_groups}), "train/eval leak"
    print(f"train {len(train_groups)} questions | eval {len(eval_groups)} questions", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = Head(queries.shape[1], args.hidden, device)
    total = sum(p.numel() for p in head.net.parameters())
    print(f"head parameters: {total/1e6:.2f}M on {device}", flush=True)
    optimiser = torch.optim.AdamW(head.net.parameters(), lr=args.lr, weight_decay=0.01)

    tensors = [(torch.tensor(g["query"], device=device), torch.tensor(g["passages"], device=device),
                torch.tensor(g["labels"], device=device)) for g in train_groups]
    rng = np.random.default_rng(SEED)
    for epoch in range(args.epochs):
        head.net.train()
        running = 0.0
        for i in rng.permutation(len(tensors)):
            q, p, y = tensors[i]
            s = head.scores(q, p)
            # Binary cross-entropy keeps the score calibrated; a listwise softmax
            # over the same group is what actually orders the candidates. Both,
            # because F2 needs every gold table found, not just the first.
            bce = torch.nn.functional.binary_cross_entropy_with_logits(s, y)
            listwise = -(torch.log_softmax(s, dim=0) * (y / y.sum())).sum()
            loss = bce + listwise
            loss.backward(); optimiser.step(); optimiser.zero_grad(set_to_none=True)
            running += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{args.epochs} loss {running/len(tensors):.4f}", flush=True)

    head.net.eval()
    ranking, cosine_ranking = {}, {}
    with torch.inference_mode():
        for g in eval_groups:
            q = torch.tensor(g["query"], device=device); p = torch.tensor(g["passages"], device=device)
            s = head.scores(q, p).cpu().numpy()
            ranking[str(g["id"])] = [g["tables"][j] for j in np.argsort(-s)]
            cos = g["passages"] @ g["query"]
            cosine_ranking[str(g["id"])] = [g["tables"][j] for j in np.argsort(-cos)]

    traces = {t["id"]: t for t in load_jsonl(args.traces)}
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    def score(rank_map):
        per = {}
        for identifier, trace in traces.items():
            candidates = list(dict.fromkeys(trace["ranked_tables"]))
            order = [t for t in rank_map.get(str(identifier), []) if t in set(candidates)]
            order += [t for t in candidates if t not in set(order)]
            budget = table_budget(len(trace["selected_docs"]), "auto")
            per[identifier] = prefix_score(gold_of(trace, "binding"), order, budget)["f2"]
        return per

    sparse = score({})
    cos = score(cosine_ranking)
    learned = score(ranking)
    cross = score(baseline)
    clusters = clusters_for(list(traces.values()), eval_records)
    n = len(sparse)
    mean = lambda d: sum(d.values()) / n
    print()
    print(f"{'ranking':<40} {'F2':>8} {'vs 8B':>10}")
    for name, d in (("sparse BM25 (no rerank)", sparse), ("E5 cosine, untrained", cos),
                    (f"E5 + learned head ({total/1e6:.2f}M)", learned),
                    ("Qwen3-Reranker-8B (shipped)", cross)):
        print(f"{name:<40} {mean(d):>8.4f} {mean(d)-mean(cross):>+10.4f}")
    b = cluster_bootstrap(cross, learned, clusters, 10000)
    print(f"\nlearned head vs 8B: delta {b['delta']:+.4f}  CI [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]")
    args.output.write_text(json.dumps(ranking, ensure_ascii=False), encoding="utf-8")
    print(f"ranking written to {args.output}")


if __name__ == "__main__":
    main()
