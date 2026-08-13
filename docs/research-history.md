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

## Second ranking round, against de-biased labels

Live feedback arrived: Tables F2 0.4221 (P 0.2149, R 0.5735, MRR@5 0.4951), Docs
F2 0.9711. Working back from it, organizer gold is ~3.17 tables per question
against the 2.57 our labels held, so `scripts/complete_benchmark_labels.py` adds
interchangeable tables — same report, same exact raw cell, overlapping row label.
That took the benchmark to 720 gold tables (3.75 per question) and lifted local
F2 to 0.5969, still above live because our benchmark averages 6.56 gated reports
against 8.46 corpus-wide.

| Change | F2 delta | 95% CI | Verdict |
|---|---|---|---|
| Drop degenerate candidate tables | +0.0029 | [−0.0004, +0.0087] | kept, see note |
| Multi-row table score replacing the raw view | −0.0031 | [−0.0199, +0.0127] | rejected |
| **Multi-row table score fused as a ranking** | **+0.0180** | **[+0.0050, +0.0320]** | **accepted** |
| Statement-type prior from question vocabulary | −0.0404 | [−0.0738, −0.0080] | rejected |

The degenerate filter did not clear the interval bar, but it drops 2.8% of
candidates that cannot answer a numeric question, never removed a gold table in a
3,447-table sample, and unclassified fragments were 18.5% of what we submitted
against 7.5% of gold. Kept as hygiene, not as a measured gain.

Scoring a table by its three strongest rows rather than its single best row
repeats the earlier pattern: as a replacement it helps intermediate (+0.0166) and
hard (+0.0262) and hurts easy (−0.0227) and medium (−0.0160); fused as its own
ranking every tier improves and recall goes 0.7588 to 0.7826.

Query decomposition against the corpus's own vocabulary is the strongest form of
the idea and it still fails: **−0.0302 F2, CI [−0.0553, −0.0059]**, with
intermediate down 0.1053 and only hard gaining (+0.0484).
`scripts/build_line_item_lexicon.py` collects row labels appearing in at least
ten reports and drops those matching more than 15% of questions, which removes
"của công ty mẹ", "số dư", "cuối năm" by frequency rather than by hand. The
result is 7,870 phrases matching 1,011 of 1,012 questions, and the decomposition
is clean — "tỷ lệ cho vay khách hàng trên tổng tiền gửi khách hàng" yields
exactly `cho vay khach hang` and `tien gui khach hang`.

The lexicon is kept as a derived artifact because it is correct and reusable; it
is the fusion that cannot absorb it. Which points at the real ceiling: the
reciprocal-rank pool weights every signal identically, so adding a signal that is
merely good dilutes the ones that are better. Two signals earned their place
(metric view, supporting rows) and everything since has diluted the mix. Getting
further almost certainly requires weighting signals by reliability — which means
fitting weights, on a benchmark that has already been shown not to transfer.

Cross-encoder reranking, the organizers' own biggest reported jump (recall@10
63.9% to 80.2%), was tried with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over
the top 20 candidates. It is far worse than no reranking: **−0.1520 F2, CI
[−0.1921, −0.1113]**, easy −0.2434, MRR 0.754 to 0.580. The model is trained on
web passage ranking and a table representation of headers, periods, and a numeric
row is out of its distribution. This is a model-quality failure, not a compute
one — the organizers' gain came from a reranker in the billions of parameters,
which four CPU cores cannot run. Throughput was made workable first (sequence
length 512 to 192, all cores, batches to 64: 3.3 to 6.5 pairs per second), so the
result is about quality, not truncation.

Section paths from the account-code hierarchy — reading "dự phòng giảm giá hàng
tồn kho" as sitting under "hàng tồn kho" under "tài sản ngắn hạn", which is the
group context the organizers' preprocessing adds — came to −0.0065, CI [−0.0209,
+0.0054], with easy down 0.0269. The paths themselves are extracted correctly;
adding them as a ranking signal did not pay.

Slot decomposition — split a question on its connectives ("trên", "và", "giữa"),
build one query per line item, and give each its own slot — was the largest
structural idea and it failed in all three forms tried:

| Form | F2 delta | 95% CI |
|---|---|---|
| Reserve a table per report × operand | −0.0020 | [−0.0113, +0.0072] |
| Fuse operand rankings into the global one | −0.0135 | [−0.0297, −0.0001] |
| Same, with a lighter token filter for operands | −0.0290 | [−0.0476, −0.0129] |

Splitting is the weak link, not the idea. On 1,012 questions the rule finds two
or three operands in 200 of them, but the fragments it produces carry period and
unit words ("tổng tài sản cuối 2016 phần trăm") or company names, and those
queries retrieve worse than the whole question does. MRR fell in every variant,
0.754 to 0.717. A parser that isolates the line-item noun phrase — rather than
cutting on connectives — would be needed before this is worth retrying.

A score-margin cutoff — submit every table scoring within 90% of the top table,
floored at one per gated report and capped at three — was swept offline from
cached scores. It trades well on paper (precision 0.3855 to 0.4273, recall 0.7826
to 0.7664, mean 5.39 tables instead of 6.56) but comes to +0.0096 F2, CI
[−0.0013, +0.0198], and the hard tier loses 0.0152. Rejected. Live precision is
0.2149 against 0.3855 locally, so an adaptive cutoff may behave better there than
here — but that is a guess, and guesses do not ship.

The statement prior failed hard. The diagnosis behind it stands — we submit 54.6%
notes against 27% in gold, and 9.6% balance sheets against 38.5% — but mapping
question vocabulary to a statement is too blunt: "số dư", "vay", and "đầu tư"
appear in most questions, so nearly everything was pushed toward balance sheets
and easy lost 0.09. Only the hard tier gained (+0.0805). A sharper signal is
needed before this idea returns.

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
