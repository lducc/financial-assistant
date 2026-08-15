# Rebuilding the working tree on another machine

`output/` and `data/raw/` are gitignored, so a clone carries the code and the
labels but none of the derived data. Almost all of it is deterministic and can be
regenerated in about an hour, unattended. One category cannot: the reranker score
files are the output of GPU runs and are committed deliberately, overriding the
ignore rule, because re-earning them costs roughly fifteen GPU-hours.

## What is committed despite the ignore rule

| file | what produced it |
|---|---|
| `output/rerank/scores.jsonl` | first Qwen run, short representation, 512 window |
| `output/rerank/scores_qwen.jsonl` | second Qwen run, same representation |
| `output/rerank/scores_qwen4b.jsonl` | Qwen3-Reranker-4B, 1024 window, `pairs.jsonl` |
| `output/rerank/scores_qwen8b.jsonl` | Qwen3-Reranker-8B, 1024 window, `pairs.jsonl` — the live 0.486 submission |
| `output/rerank/scores_bench_v4.jsonl` | 8B on the benchmark, inventory representation and line items |
| `output/rerank/scores_bench_d100.jsonl` | 8B on the benchmark at depth 100 |
| `output/rerank/scores_v4.jsonl` | 8B over all 1,012 questions, inventory representation — **the shipped `submission_v4`** |
| `output/rerank/orders_bench{,_rev}.jsonl` | listwise 8B, 233 benchmark questions, both presentations |

All seven score files carry identical candidate sets, so any of them fuses against
any pairs export; only the candidate *text* the model saw differs between them.

`scores_v4.jsonl` and `scores_bench_v4.jsonl` are the same representation but not
the same run: no question has an identical ordering, top-1 agrees on 189 of 233,
and the two fuse to 0.6549 and 0.6508 respectively.

That floor is higher than it looks and it is not a property of these two files.
Paired per question, re-scoring the identical pairs in another session is worth
**+0.0076 F2, CI [-0.0063, +0.0235]** — 138 of 11,592 scores agree, mean |delta|
0.0098. int8 is not batch-invariant. So no score file here may be compared with
another as if the difference were method; see `docs/ASSESSMENT.md` §8 and use
`scripts/compare_rerank_runs.py`, or score both cells in one fp16 session.

## Rebuilding everything else

```bash
# 1. the corpus (383 MB) — see docs/HANDOFF.md for the source
#    data/raw/vifinqa/{financial_statements,questions,code_stock.csv}

# 2. candidate pairs, ~50 min each
python3 scripts/export_rerank_pairs.py --output output/rerank/pairs_v4.jsonl
python3 scripts/export_rerank_pairs.py --depth 100 --output output/rerank/pairs_v4_d100.jsonl

# 3. benchmark subsets, seconds
python3 - <<'PY'
import json
ids={json.loads(l)['id'] for l in open('annotations/benchmark.jsonl',encoding='utf-8') if l.strip()}
for src,dst in (('pairs_v4','pairs_bench_v4'),('pairs_v4_d100','pairs_bench_d100')):
    with open(f'output/rerank/{dst}.jsonl','w',encoding='utf-8') as out:
        for line in open(f'output/rerank/{src}.jsonl',encoding='utf-8'):
            if line.strip() and json.loads(line)['id'] in ids:
                out.write(line)
PY

# 4. rankings from the committed scores, seconds
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_v4.jsonl \
    --scores output/rerank/scores_qwen8b.jsonl --output output/rerank/ranking_8b_fuse.json
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_bench_v4.jsonl \
    --scores output/rerank/scores_bench_v4.jsonl --output output/rerank/ranking_bench_v4.json
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_bench_d100.jsonl \
    --scores output/rerank/scores_bench_d100.jsonl --output output/rerank/ranking_bench_d100.json

# 5. listwise windows, seconds — both presentations
python3 scripts/export_listwise_windows.py --pairs output/rerank/pairs_bench_v4.jsonl \
    --ranking output/rerank/ranking_bench_v4.json --output output/rerank/windows_bench.jsonl
python3 scripts/export_listwise_windows.py --pairs output/rerank/pairs_bench_v4.jsonl \
    --ranking output/rerank/ranking_bench_v4.json --reverse --output output/rerank/windows_bench_rev.jsonl
python3 scripts/export_listwise_windows.py --pairs output/rerank/pairs_v4.jsonl \
    --ranking output/rerank/ranking_8b_fuse.json --output output/rerank/windows_full.jsonl
python3 scripts/export_listwise_windows.py --pairs output/rerank/pairs_v4.jsonl \
    --ranking output/rerank/ranking_8b_fuse.json --reverse --output output/rerank/windows_full_rev.jsonl

# 6. the shipped ranking and submission, ~35 min
python3 scripts/apply_rerank_scores.py --pairs output/rerank/pairs_v4.jsonl \
    --scores output/rerank/scores_v4.jsonl --output output/rerank/ranking_v4_fuse.json
python3 run.py --ranking output/rerank/ranking_v4_fuse.json --output-dir output/submission_v4
python3 scripts/validate_submission.py output/submission_v4/package
```

Use `.venv/bin/python`; the system interpreter has neither pytest nor torch.

Run `.venv/bin/pip install -e .` once in a fresh checkout. Scripts used to
prepend `src/` to `sys.path` themselves; they import `vifinqa` now, so without
the install every one of them fails at its first import.

## Checks that should pass afterwards

```bash
.venv/bin/python -m pytest -q                  # 63 passing
python3 scripts/verify_benchmark.py            # VALID
python3 scripts/audit_entity_resolution.py     # 0 unsupported resolutions
```

Benchmark F2 under `--gold binding` at the shipped budget, via
`scripts/diagnose_retrieval.py --ranking`:

| ranking | F2 |
|---|---:|
| `ranking_8b_fuse.json` | 0.6248 |
| `ranking_bench_v4.json` | 0.6508 |
| `ranking_v4_fuse.json` | 0.6549 |
| `ranking_listwise_bench.json` | 0.6050 |
| `ranking_bench_d100.json`, on a depth-100 cache | 0.6650 |

`ranking_bench_d100.json` scores 0.6646 against the committed depth-50 cache,
because `reorder` cannot submit a table retrieval did not return; 0.6650 needs
the depth-100 candidates. The difference between the two is the whole realized
value of the extra candidates — 3 of the 9,158 reach the submitted budget.

Earlier revisions of this file claimed 0.6562 and 0.6692 for the second and last
of those. Those figures came from inline commands that no longer exist and do not
reproduce; candidate sets match 233/233, so retrieval is not the cause. The table
above is what the committed scripts produce.

## Where things stand

Live, best to date: **Tables F2 0.5221**, Docs F2 0.9711, answer 0.1561,
execution 0.1285 — `output/submission_v4_tiered`, built from
`ranking_v4_tiered.json` (`--replace-tiers hard,intermediate`). The two
submissions before it read 0.486 and 0.5118.

Answer and execution accuracy have not moved with any of it; see
`docs/ASSESSMENT.md` §4e.
Closed: listwise reranking, measured at −0.0458 and rejected — do not run
`windows_full*.jsonl`. See `docs/ASSESSMENT.md`.
Next GPU run: the `PER_ITEM` A/B, both cells in one session over
`pairs_bench_v4.jsonl`. Depth 100 is **not** next: +0.0100 was measured across
two sessions and is +0.0066 of candidates on +0.0076 of scoring drift, and only
3 of the 9,158 candidates it adds reach the submitted budget. See
`docs/ASSESSMENT.md` §8.
