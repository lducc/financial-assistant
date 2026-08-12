# Research history

The local corpus is OCR-extracted Vietnamese financial statements plus a company
registry and organizer questions. OCR is treated as immutable input: table IDs
are `report_id|start_line`, and every submitted table is materialized as a CSV.

Public leaderboard submissions score retrieval evidence only. They do not
establish answer correctness, so this repository intentionally does not execute
Pandas programs or claim answer quality.

## Current baseline

V1 is deterministic contextual BM25. It first selects documents by company,
year, and statement scope; then ranks parsed tables from those documents using
rows together with title, nearby context, headers, periods, and units. The
submission always uses fixed top-5 tables. `--table-mode role-coverage` is an
optional multi-year reranking experiment; baseline remains the default.

## Trust boundary

`annotations/gold_150.jsonl` contains 150 reconstructed retrieval labels. Its
document/table/cell bindings come from the preserved E027 evidence record and
validate against raw OCR. The original answer programs, operation graphs, and
per-record taxonomy were not recoverable, so it is retrieval ground truth only.

Candidate answer/Pandas artifacts live separately in
`annotations/gold_150_candidate_submissions.jsonl` and are never treated as
gold.

`annotations/pilot_v1/agent_labels.jsonl` contains twelve source-cell audits.
They are useful for retrieval diagnostics only: six are user-reviewed and six
still need independent review. Neither set is organizer ground truth or supports
answer-quality claims. Run `python3 scripts/validate_pilot.py --labels
annotations/gold_150.jsonl` against local corpus to verify every reconstructed
binding reaches its stated raw OCR cell.

## Measured on gold-150 dev (105 records, `report-coverage`)

| Change | Submitted F2 | P | R | candidate recall@50 |
|---|---|---|---|---|
| Fixed top-5 | 0.4356 | 0.2476 | 0.6173 | 0.9052 |
| One table per gated report | 0.4716 | 0.4946 | 0.4689 | 0.9052 |
| **Three tables per gated report** | **0.5343** | 0.2645 | 0.7338 | 0.9052 |
| Corpus-wide BM25 IDF (rejected) | 0.4923 | 0.2433 | 0.6762 | 0.9122 |

The budget sweep peaks sharply at three tables per gated report and holds on both
halves of the dev split. Corpus-wide document frequency, built by
`scripts/build_row_idf.py` over all 1,535,824 parsed rows, was rejected: it lifts
deep candidate recall slightly but costs 4 points of submitted F2, because
statistics local to the gated slice downweight terms that are boilerplate inside
the company's own reports, which is what top-k ranking needs.

Candidate recall@50 is 0.9052 while F2@5 recall is 0.6029, so the gold table is
usually retrieved and mis-ranked rather than missed. Ranking, not candidate
generation, is where the remaining table headroom is.

## Rejected directions and next hypothesis

Historical pseudo-ground-truth rules, answer planners, text repair, and numeric
execution were removed because they blurred the retrieval boundary or produced
unverifiable confidence. The next hypothesis is that field-aware weighting of
title, headers, row labels, periods, and units can improve table recall while
preserving the same document gate and fixed-K submission contract.
