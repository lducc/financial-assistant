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

**1. Export the pairs (local, ~35 min)**

```
python3 scripts/export_rerank_pairs.py --output output/rerank/pairs_v2.jsonl
```

Writes one record per question with its top-50 candidates and the table
representation the model scores — 50,335 pairs, 52 MB.

The representation changed after the first run. Qwen-4B moved live F2 by only
+0.023, and the likely reason was that each candidate was described by the single
row BM25 matched: the model was judging tables through the sparse ranker's choice
and could not recover when that choice was wrong. Candidates now also carry the
table's line-item inventory (`Các chỉ tiêu: …`), which says what the table
actually contains. Pass `--inventory 0` to reproduce the old representation.

**2. Upload to Kaggle**

Create a Dataset named `vifinqa-rerank-pairs` containing `pairs_v3.jsonl`, and
set `PAIRS_PATH` at the top of the script to match. Qwen scored the first export
at F2 +0.0355 on the benchmark and +0.023 live, so it is the model to use;
`rerank_bge.py` is kept only to document that a general web-passage reranker
loses 0.15 F2 here.

**3. Run one of the notebooks (Kaggle)**

New notebook → Accelerator: **GPU T4 x2** or **P100** → attach the dataset →
paste one script into a cell → Run. Both print throughput and an ETA, and both
write `/kaggle/working/scores.jsonl` (~1 MB) in the same format, so step 4 does
not care which one produced it.

| Script | Model | Memory | 50,335 pairs |
|---|---|---|---|
| `rerank_qwen_4b.py` | Qwen3-Reranker-4B, fp16 | ~8 GB | 3–4 h |
| `rerank_qwen_8b.py` | Qwen3-Reranker-8B, 8-bit | ~8.2 GB | 9–15 h, resumable |
| `rerank_bge.py` | bge-reranker-v2-m3 | ~1 GB | kept only as the record of a failure |

**Run 4B first.** It is the configuration that has actually worked here: +0.0355
F2 on the benchmark, +0.023 live. Whether 8B beats it is an open question — on
reranking tasks the gap between sizes is usually smaller than the gap between the
right and wrong model family, and BGE losing 0.15 F2 is what that gap looks like.
Measure both on the benchmark before spending a submission on either.

**8B loads in 8-bit** because 8.19B parameters is 16.4 GB in fp16 and a T4 has
16 GB. At int8 it is about 8.2 GB, fits one card, and stays near-lossless. It is
the slower of the two models by more than the parameter count suggests:
bitsandbytes computes outlier features in higher precision, roughly 1.5–2× fp16
on Turing.

Both Qwen scripts sort candidates by length before batching, since a batch costs
its longest member — worth 25–40% of the runtime — and both append per question,
skipping IDs already done, so a session that hits the 12-hour limit is rerun
rather than restarted.

All models are open, inside the 14B limit, and released before the 2026-06-01
cutoff. Kaggle needs internet enabled to download them.

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
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_v3.jsonl \
    --scores output/rerank/scores.jsonl
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_v3.jsonl \
    --scores output/rerank/scores.jsonl --mode replace   # trust the model outright
```

Writes `output/rerank/ranking.json`. Fusion is the default because on this task
every replacement has lost and every fusion that earned its place has won;
`--mode replace` exists so the difference can be measured rather than assumed.

## What to check before shipping it

Measure on the benchmark first — the reranked ordering has to beat the sparse one
on the 192 labelled questions, and the tier breakdown has to not regress. If the
model helps, the submission is rebuilt with the fused ordering. If it does not,
delete `scores.jsonl` and nothing else changes.
