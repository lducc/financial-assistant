# ViFinQA

Our second-place system for R2AI Stage 2, question answering over Vietnamese financial
statements. For each question it picks the source tables from the organizer reports,
copies them out as evidence CSVs, and answers with a Pandas program that runs on those
CSVs.

- Submitted package: [`artifacts/submission_best_20260904_declared.zip`](artifacts/submission_best_20260904_declared.zip)
- Organizer dossier: [SUBMISSION_DOCUMENTATION.md](SUBMISSION_DOCUMENTATION.md)
- Demo: [demo/README.md](demo/README.md)

## Overview

```text
question
  -> reports with the same ticker, year and scope
  -> candidate tables from BM25, a dense index and note links
  -> Qwen3-Reranker-8B scores each candidate
  -> fused ranking; multi-part questions also take tables within 3 logits of the best
  -> evidence CSVs
  -> Qwen3.5-9B writes a Pandas program, replayed in a sandbox
  -> submission.zip
```

Retrieval is fixed before answering, so the answer stage cannot change which reports
and tables are submitted. Every evidence CSV is a copy of an organizer table, and every
answer is computed from those CSVs.

## Example

From the fixture in `tests/test_answers.py`. The ranker selects this table:

```text
Chỉ tiêu          Số đầu năm   31/12/2022   31/12/2023
Doanh thu thuần   1.000        1.200        1.400
```

> Tăng trưởng doanh thu năm 2023 so với năm 2022 là bao nhiêu phần trăm?
> (By what percentage did revenue grow from 2022 to 2023?)

The program reads the 2022 and 2023 cells of the revenue row and returns
`(1400 - 1200) / 1200 * 100 = 16.67`.

## Method

| Part | Choice |
|---|---|
| Document gate | Ticker, year and scope matched from the question |
| Sparse retrieval | BM25 over table text, top 50 |
| Dense retrieval | `Qwen/Qwen3-Embedding-4B`, top 40 from the gated reports |
| Note links | Up to 3 tables reached through note references in the top 30 |
| Reranker | `Qwen/Qwen3-Reranker-8B`, zero-shot |
| Fusion | Reciprocal rank of first-stage and reranker order, weight 0.5 |
| Answer | `Qwen/Qwen3.5-9B` writes a Pandas program, checked against an allowlist and replayed |

Every model is a public checkpoint used as is. Revisions are pinned in `kaggle/` and
`notebooks/final_answer.ipynb`.

## Reproduce

Request the data from the organizers and place it as follows:

```text
data/raw/vifinqa/code_stock.csv
data/raw/vifinqa/questions/questions.jsonl
data/raw/vifinqa/financial_statements/<ticker>/<year>/...
```

```sh
git clone https://github.com/lducc/financial-assistant && cd financial-assistant
pip install -e .
pytest
```

Check a submitted package, with no data or GPU:

```sh
unzip artifacts/submission_best_20260904_declared.zip -d output/package
python scripts/validate_submission.py output/package
python scripts/verify_execution.py --package output/package
```

Rebuild from the organizer data. Steps 1, 3 and 5 need a GPU.

1. Dense index: `scripts/export_table_texts.py`, then `kaggle/embed_tables.py`.
2. Candidates: BM25, dense and note-link tables for every question.
3. Rerank: `kaggle/rerank_qwen_8b.py` scores every candidate.
4. Package: fuse the scores, select tables, and write `submission.zip` with `run.py`.
5. Answer: `scripts/build_answer_inputs.py`, then every cell of `notebooks/final_answer.ipynb` on an A100.

<details>
<summary>Commands</summary>

```sh
RUN=output/final_run
Q=data/raw/vifinqa/questions/questions.jsonl

# 1. copy the four outputs of kaggle/embed_tables.py into output/dense/
python scripts/export_table_texts.py

# 2
python scripts/build_line_item_lexicon.py
python scripts/verify_dense_artifacts.py --dense output/dense \
  --tables output/dense/tables.jsonl --questions $Q --manifest $RUN/dense_manifest.json
python scripts/export_rerank_pairs.py --depth 50 --dense output/dense \
  --dense-depth 40 --hierarchy \
  --output $RUN/pairs_hybrid.jsonl --dense-tables $RUN/dense_tables.json
python scripts/apply_rerank_scores.py --pairs $RUN/pairs_hybrid.jsonl \
  --first-stage-only --output $RUN/ranking_hybrid_first_stage.json
python scripts/add_graph_tables.py --pairs $RUN/pairs_hybrid.jsonl \
  --base-ranking $RUN/ranking_hybrid_first_stage.json \
  --depth 30 --max-added 3 --hierarchy \
  --output $RUN/ranking_graph_candidates.json --trace $RUN/graph.trace.jsonl \
  --expanded-pairs $RUN/pairs_hybrid_graph.jsonl
python scripts/merge_table_sidecars.py --input $RUN/dense_tables.json \
  --input $RUN/ranking_graph_candidates.graph_tables.json --output $RUN/extra_tables.json
python scripts/validate_rerank_pairs.py --pairs $RUN/pairs_hybrid_graph.jsonl \
  --extra-tables $RUN/extra_tables.json --graph-trace $RUN/graph.trace.jsonl \
  --questions $Q --dense-manifest $RUN/dense_manifest.json \
  --manifest $RUN/candidate_manifest.json

# 3. on the GPU, then copy scores.jsonl and scores.jsonl.manifest.json into $RUN
python kaggle/rerank_qwen_8b.py pairs_hybrid_graph.jsonl scores.jsonl

# 4
python scripts/validate_rerank_scores.py --pairs $RUN/pairs_hybrid_graph.jsonl \
  --scores $RUN/scores.jsonl --manifest $RUN/scores.jsonl.manifest.json
python scripts/apply_rerank_scores.py --pairs $RUN/pairs_hybrid_graph.jsonl \
  --scores $RUN/scores.jsonl --weight 0.5 --output $RUN/ranking_hybrid_qwen.json
python scripts/select_slot_tables.py --pairs $RUN/pairs_hybrid_graph.jsonl \
  --scores $RUN/scores.jsonl --base-ranking $RUN/ranking_hybrid_qwen.json \
  --starting-ranking $RUN/ranking_hybrid_qwen.json --logit-gap 3 \
  --output $RUN/ranking_adaptive_gap3.json --trace $RUN/ranking_adaptive_gap3.trace.jsonl
python run.py --ranking $RUN/ranking_adaptive_gap3.json \
  --extra-tables $RUN/extra_tables.json --table-top-k ranking --rerank-depth 130 \
  --output-dir $RUN/package_run

# 5. on the A100, then run notebooks/final_answer.ipynb
python scripts/build_answer_inputs.py $RUN/package_run/submission.zip \
  --output-dir /content/answer_inputs
```

</details>

## Code

```text
run.py                      gate, retrieve, write evidence CSVs, build the package
src/docs.py                 organizer files, question parsing, document gate
src/vifinqa/retrieval.py    BM25 over tables in the gated reports
src/vifinqa/rerank.py       table text shown to the reranker
src/vifinqa/fusion.py       reciprocal-rank fusion
src/vifinqa/graph.py        note-link candidates
src/vifinqa/slots.py        gap-3 table selection
src/vifinqa/answers.py      cell binding and Pandas programs
src/vifinqa/tables.py       table parsing and CSV evidence
scripts/                    stage scripts and validators
kaggle/                     dense index and reranker jobs (GPU)
notebooks/final_answer.ipynb   answer stage (GPU)
demo/                       chat UI over the same stages
```
