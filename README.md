# ViFinQA

Our second-place system for R2AI Stage 2, Vietnamese financial table QA. For each
question it picks source tables from the organizer reports, copies them out as
evidence CSVs, and answers with a Pandas program that runs on those CSVs.

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

Every model is a public checkpoint used as is. There is no training step.

| Checkpoint | Revision | Stage |
| --- | --- | --- |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` | dense index |
| `Qwen/Qwen3-Reranker-8B` | `77d193c791ed757ca307ee72715aa132723da912` | reranking |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | answers |

## Setup

```bash
git clone https://github.com/lducc/vifinqa-final-submission.git
cd vifinqa-final-submission
pip install -e .
pytest
```

Request the data from the organizers and place it as follows:

```text
data/raw/vifinqa/code_stock.csv
data/raw/vifinqa/questions/questions.jsonl
data/raw/vifinqa/financial_statements/<ticker>/<year>/...
```

## Reproduce

Steps 1, 3 and 5 run on a GPU. The rest run on a CPU.

1. Dense index. Run `python scripts/export_table_texts.py`, then run
   `kaggle/embed_tables.py` on a GPU with `output/dense/tables.jsonl` and
   `questions.jsonl` (paths at the top of the file). Copy its four output files
   into `output/dense/`.

2. Candidates.

   ```bash
   RUN=output/final_run
   Q=data/raw/vifinqa/questions/questions.jsonl

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
   ```

3. Rerank. On the GPU, run
   `python kaggle/rerank_qwen_8b.py pairs_hybrid_graph.jsonl scores.jsonl` and copy
   `scores.jsonl` and `scores.jsonl.manifest.json` into `$RUN`.

4. Select tables and package.

   ```bash
   python scripts/validate_rerank_scores.py --pairs $RUN/pairs_hybrid_graph.jsonl \
     --scores $RUN/scores.jsonl --manifest $RUN/scores.jsonl.manifest.json
   python scripts/apply_rerank_scores.py --pairs $RUN/pairs_hybrid_graph.jsonl \
     --scores $RUN/scores.jsonl --weight 0.5 \
     --output $RUN/ranking_hybrid_qwen.json
   python scripts/select_slot_tables.py --pairs $RUN/pairs_hybrid_graph.jsonl \
     --scores $RUN/scores.jsonl --base-ranking $RUN/ranking_hybrid_qwen.json \
     --starting-ranking $RUN/ranking_hybrid_qwen.json --logit-gap 3 \
     --output $RUN/ranking_adaptive_gap3.json --trace $RUN/ranking_adaptive_gap3.trace.jsonl
   python run.py --ranking $RUN/ranking_adaptive_gap3.json \
     --extra-tables $RUN/extra_tables.json --table-top-k ranking --rerank-depth 130 \
     --output-dir $RUN/package_run
   python scripts/validate_submission.py $RUN/package_run/package
   python scripts/verify_execution.py --package $RUN/package_run/package
   ```

5. Answer. On an A100, run
   `python scripts/build_answer_inputs.py $RUN/package_run/submission.zip --output-dir /content/answer_inputs`,
   then every cell of `notebooks/final_answer.ipynb`. It writes the final ZIP to `/content/`.

## Results

| Package in `artifacts/` | SHA-256 |
| --- | --- |
| `submission_best_20260826.zip` | `97baa1201450801e50480ce68db055d7bdbc99029cac93c21654cdf471789067` |
| `submission_best_20260904_declared.zip` | `79050960648188ecacbe50eca813ec51d9f8e0bcebfc2b175e58682b8e81dee3` |

Each package answers all 1,012 questions. The 20260826 package scored table F2
0.5495, document F2 0.9711 and answer execution 0.1561 on the public leaderboard.
The organizer dossier is [SUBMISSION_DOCUMENTATION.md](SUBMISSION_DOCUMENTATION.md).

## Demo

`demo/` is a chat UI that runs the same stages on a fresh question and traces each
number to its source cell. See [demo/README.md](demo/README.md).
