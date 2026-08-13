# Cross-encoder reranking on Kaggle

Every team ahead of us on table retrieval runs a reranker; we do not. Our MRR@5 is
0.482 against 0.51–0.63 for the systems above us, and reranking is the stage that
moves exactly that number. A local attempt with an English web-passage
cross-encoder (`mmarco-mMiniLMv2`) lost 0.15 F2 — the model was wrong for
Vietnamese financial tables, not the idea. This runs the multilingual model the
leaders use, on a GPU we do not have locally.

Only the ordering of already-retrieved candidates changes. The document gate, the
candidate set, and the table IDs are produced locally and untouched here, so
discarding the result means deleting one file.

## Steps

**1. Export the pairs (local, ~25 min)**

```
python3 scripts/export_rerank_pairs.py
```

Writes `output/rerank/pairs.jsonl` — one record per question with its top-50
candidates, each carrying the table representation the model will score. About
36 MB for all 1,012 questions.

**2. Upload to Kaggle**

Create a Dataset named `vifinqa-rerank-pairs` containing `pairs.jsonl`. If you
name it differently, change `PAIRS_PATH` at the top of `rerank_bge.py`.

**3. Run one of the notebooks (Kaggle)**

New notebook → Accelerator: **GPU T4 x2** or **P100** → attach the dataset →
paste one script into a cell → Run. Both print throughput and an ETA, and both
write `/kaggle/working/scores.jsonl` (~1 MB) in the same format, so step 4 does
not care which one produced it.

| Script | Model | Params | 50k pairs on a T4 |
|---|---|---:|---|
| `rerank_bge.py` | `BAAI/bge-reranker-v2-m3` | 568M | 20–35 min |
| `rerank_qwen.py` | `Qwen/Qwen3-Reranker-0.6B` | 0.6B | ~20 min |
| `rerank_qwen.py` | `Qwen/Qwen3-Reranker-4B` (default) | 4B | 2–3 h |
| `rerank_qwen.py` | `Qwen/Qwen3-Reranker-8B` | 8B | needs both T4s or 4-bit |

Pick the Qwen size with an environment variable before running:
`os.environ["QWEN_RERANKER"] = "Qwen/Qwen3-Reranker-0.6B"`.

Start with BGE — it is fast, and if reranking helps at all it will show there.
Qwen-4B is the stronger model and the family the organizer slides measured
(recall@10 63.9% → 80.8% with a reranker), so it is worth the longer run once
BGE has shown the approach pays.

All four are open models inside the 14B limit and released before the
2026-06-01 cutoff. Kaggle needs internet enabled to download them.

The two scripts score differently under the hood. BGE is a cross-encoder with a
classification head, so its logit is the score directly. Qwen3-Reranker is a
causal model asked whether the document answers the query, and the score is the
probability it answers "yes" — which needs a specific prompt format and left
padding so the final position is the real one in every row of a batch.
`rerank_qwen.py` also appends to `scores.jsonl` and skips questions it already
scored, so a session that hits the 12-hour limit can be rerun to finish the rest.

**4. Download and fuse (local, seconds)**

Put `scores.jsonl` in `output/rerank/`, then:

```
python3 scripts/apply_rerank_scores.py            # reciprocal-rank fusion, the default
python3 scripts/apply_rerank_scores.py --mode replace   # trust the model outright
```

Writes `output/rerank/ranking.json`. Fusion is the default because on this task
every replacement has lost and every fusion that earned its place has won;
`--mode replace` exists so the difference can be measured rather than assumed.

## What to check before shipping it

Measure on the benchmark first — the reranked ordering has to beat the sparse one
on the 192 labelled questions, and the tier breakdown has to not regress. If the
model helps, the submission is rebuilt with the fused ordering. If it does not,
delete `scores.jsonl` and nothing else changes.
