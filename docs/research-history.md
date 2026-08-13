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

## Benchmark trust

All 150 gold-150 records re-verified against raw OCR: `python3
scripts/validate_pilot.py --labels annotations/gold_150.jsonl` prints `VALID`,
so every table ID, start line, and raw cell still resolves in the corpus.

Seeding bias, measured rather than assumed: 181 of 420 gold tables (43.1%) also
appear in the submission-2333 candidate file, and 78 of 150 questions (52.0%)
have their entire gold set inside those candidates. The remaining 57% of gold
tables were located independently during the source audit. The set leans toward
what one public retriever found, but it is not a copy of it.

## Question tiers

`scripts/classify_questions.py` assigns the organizer's four difficulty tiers to
all 1,012 questions using `docs.parse_question` for entities, years, and scope.
Against the published counts it lands at easy 331 (361), medium 184 (235),
intermediate 256 (200), hard 241 (216) — absolute deviation 162 of 1,012. The
tiers are not calibrated to those counts beyond this check.

The tiers hold up against evidence they never saw, on gold-150:

| Tier | Questions | Mean gold tables | Mean gold reports |
|---|---:|---:|---:|
| easy | 46 | 1.00 | 1.00 |
| medium | 39 | 2.08 | 1.77 |
| intermediate | 42 | 3.50 | 3.07 |
| hard | 23 | 6.35 | 4.39 |

Every easy question has exactly one gold table in one report.

## Table budget, cross-validated

`scripts/cross_validate_retrieval.py` retrieves once to depth 50, then scores
selection policies offline over five folds blocked by connected report groups,
with a paired cluster bootstrap against a fixed top-5 baseline. All 150 records:

| Policy | F2 | P | R | mean k | Δ vs fixed-5 (95% CI) |
|---|---:|---:|---:|---:|---|
| fixed-5 | 0.4553 | 0.2480 | 0.6550 | 5.00 | baseline |
| fixed-10 | 0.4127 | 0.1653 | 0.7924 | 10.00 | −0.0425 [−0.0772, −0.0059] |
| one per report | 0.4687 | 0.4908 | 0.4661 | 2.34 | +0.0135 [−0.0405, +0.0695] |
| two per report | 0.5238 | 0.3294 | 0.6224 | 4.68 | +0.0686 [+0.0343, +0.1025] |
| **three per report** | **0.5532** | 0.2701 | 0.7637 | 7.02 | **+0.0979 [+0.0727, +0.1242]** |
| four per report | 0.5064 | 0.2111 | 0.7980 | 9.35 | +0.0511 [+0.0225, +0.0801] |
| tier-aware multiplier | 0.5305 | 0.3918 | 0.6270 | 5.95 | +0.0752 [+0.0252, +0.1259] |

Three per report improves every tier (easy 0.54→0.70, medium 0.52→0.55,
intermediate 0.40→0.48, hard 0.26→0.39) and the frozen 45-record holdout agrees:
F2 0.5972 against 0.5011, Δ +0.0961 [+0.0496, +0.1511]. One per report — the
structurally obvious budget — does not separate from the baseline. Conditioning
the multiplier on the difficulty tier is not better than applying it uniformly.

## Extending the benchmark (v3)

Resampling the cached traces shows what the current 150 records can decide: CI
half-width 0.041 F2 overall, so only effects above ~0.08 are resolvable, and per
tier it is 0.018 easy, 0.085 medium, 0.090 intermediate, 0.113 hard. The accepted
budget effect was 0.098 — barely clear of the floor, and nothing about hard
questions is measurable at n=23. `scripts/build_annotation_queue.py` therefore
draws 250 further questions weighted 40/30/20/10 toward hard and intermediate,
excluding any question whose gated reports already appear in gold-150 so new
records form new bootstrap clusters instead of thickening existing ones.

Discovery is deliberately not our retriever. `scripts/search_evidence.py` does
diacritic-folded substring matching over the raw OCR of the gated reports, in
document order, with no scoring; `scripts/propose_evidence.py` derives the metric
phrase from the question and relaxes it from the front, since Vietnamese noun
phrases lead with a generic head ("số dư", "tổng giá trị") and carry the
discriminative words at the tail. Rows are proposals; a reviewer picks.

Calibration batch, 20 questions: 18 labelled, 0 binding errors against raw OCR, 2
deferred rather than guessed — one names a fund absent from the report, one asks
for "chứng khoán nợ" where three portfolios carry that label. Median 3.5
candidate rows per question after relaxation, 7 of 20 unambiguous.

These labels were discovered independently of our BM25, and they agree with
gold-150 on the budget: three tables per gated report is again the best policy
(F2 0.6457 against 0.5674 for a fixed five, Δ +0.0783). At n=18 the interval
crosses zero, so this corroborates the direction rather than proving it.

Batch 0, 50 questions: 21 labelled, 0 binding errors, 29 deferred. The deferrals
are not evenly spread — 12 of 13 hard questions could not be labelled. Hard
questions are multi-hop by definition: an intermediate result picks the year,
company, or row whose table is the answer. Where the selection criterion and the
target share a statement the label is exact and cheap (id 526: earnings per share
picks the year, and both live in the income statement). Where they live in
different statements, labelling requires computing the intermediate value, which
this pass deliberately does not do. Extending the hard tier therefore needs a
scope decision, not more of the same effort.

Merged benchmark, 189 records and 157 clusters: three per report holds at
+0.0896 F2, CI [+0.0662, +0.1129]. The interval half-width tightened from 0.0258
to 0.0234, roughly the 1/sqrt(n) the resampling predicted.

| Split | Records | Clusters | Three-per-report Δ vs fixed-5 |
|---|---:|---:|---|
| gold-150 | 150 | 119 | +0.0979 [+0.0727, +0.1242] |
| v3 fresh labels | 39 | — | corroborating, intervals wide |
| merged | 189 | 157 | +0.0896 [+0.0662, +0.1129] |

## Ranking, measured on the 192-record benchmark

Diagnostics first: the document gate loses nothing on any record, so every gold
report is gated and the losses are downstream. Raising the ranked depth from 50
to 200 moves `candidate_miss` from 0.209 to 0.065 and `rank_miss` from 0.201 to
0.345, which settles the question — gold tables are reachable and mis-ranked, not
missing. Ranking is the bottleneck.

| Change | F2 delta | 95% CI | Verdict |
|---|---|---|---|
| Round-robin interleave by report | +0.0044 | [−0.0065, +0.0152] | rejected |
| Metric query replacing the question query | +0.0073 | [−0.0163, +0.0307] | rejected, easy −0.0258 |
| **Metric query fused as a third ranking** | **+0.0242** | **[+0.0061, +0.0430]** | **accepted** |

Round-robin was meant to cure report starvation, and starvation is real, but it
lives beyond the submitted budget: reordering inside the budget touched only 18
of 192 questions.

Swapping the query for the metric view helps derived questions (intermediate
+0.0431, hard +0.0222) and hurts easy ones (−0.0258), where the question already
reads like a row label. Fusing it instead keeps both: easy is unchanged to four
decimals while medium gains +0.0446, intermediate +0.0384, hard +0.0160. Pooled
recall 0.7536 to 0.7883, MRR 0.6122 to 0.6622.

## Rejected

Corpus-wide BM25 IDF, built by `scripts/build_row_idf.py` over all 1,535,824
parsed rows: it lifts candidate recall@50 from 0.9052 to 0.9122 but drops
submitted F2 from 0.5343 to 0.4923 on the dev split. Statistics local to the
gated slice downweight terms that are boilerplate inside the company's own
reports, which is what top-k ranking needs.

Candidate recall@50 is 0.9052 while recall@5 is 0.6029, so the gold table is
usually retrieved and mis-ranked rather than missed. Ranking, not candidate
generation, is where the remaining table headroom is.

## Rejected directions and next hypothesis

Historical pseudo-ground-truth rules, answer planners, text repair, and numeric
execution were removed because they blurred the retrieval boundary or produced
unverifiable confidence. The next hypothesis is that field-aware weighting of
title, headers, row labels, periods, and units can improve table recall while
preserving the same document gate and fixed-K submission contract.
