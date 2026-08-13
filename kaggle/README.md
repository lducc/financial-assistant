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

**3. Run the notebook (Kaggle, ~20–35 min on T4)**

New notebook → Accelerator: **GPU T4 x2** or **P100** → attach the dataset →
paste `rerank_bge.py` into a cell → Run. It prints throughput and an ETA every
5,000 pairs, and writes `/kaggle/working/scores.jsonl` (~1 MB).

Model is `BAAI/bge-reranker-v2-m3`: multilingual, 568M parameters, open, released
well before the 2026-06-01 cutoff, so it is inside the competition's model rules.
Kaggle needs internet enabled on the notebook to download it.

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
