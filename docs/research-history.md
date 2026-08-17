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

## The benchmark over-labels gold by 39%, and live told us so

Live feedback on the Qwen3-Reranker-8B submission returned the per-metric
breakdown for the first time: Tables F2 0.486 (P 0.3213, R 0.5706, MRR@5 0.5685),
Docs F2 0.9711 (P 0.9611, R 0.9767, MRR@5 0.9792), against 4B on identical pairs
at Tables F2 0.4553. Reranking is now worth +0.043 live over the sparse-era
0.4431, and 8B beats 4B by 0.031.

The precision-to-recall ratio identifies the organizers' gold set. Submitting k
tables against G gold gives P/R = G/k, so the ratio is a direct read on how many
tables they count:

| gold definition | tables per question | P/R at our x2 budget |
|---|---:|---:|
| benchmark, after restatement completion | 4.50 | 0.764 |
| benchmark, tables named by a row/column binding | 3.24 | 0.572 |
| live | — | 0.563 |

Binding-only gold reproduces the live ratio to within 0.009, so
`complete_benchmark_labels.py` widened gold past what is scored — it was built to
close a gap between 2.57 labelled tables and an inferred organizer 3.17, and it
overshot. That is why local F2 reads 0.63 where live reads 0.486.

The practical consequence is a decision reversed. Sweeping the submission budget
against full gold picks x3 gated reports (F2 0.6392 against 0.6261 at x2); against
binding-only gold the optimum is x2, which is what ships. The apparent gain was
the over-labelling rewarding recall that is not scored.

Ranking comparisons are unaffected: sparse < 4B < 8B and fuse > replace hold under
both definitions. The mis-calibration moves levels, not order, so no earlier
accept or reject changes.

Docs is closed. The live split shows precision 0.9611 and recall 0.9767, and four
candidate fixes were refuted before spending a submission: pruning to the reports
that contributed a table (drops 63 reports, 9 of them gold), collapsing
consecutive years (807 reports affected, but precision >= 0.87 caps non-gold at
~390, so most must be gold), pruning reports whose OCR lacks the asked line item
(removes 33 gold, 0 non-gold), and expanding scope-silent questions to the sibling
statement (2 of 2,080 reports would change). Local instruments cannot measure docs
precision at all: the gate defines the search space, the search defines the gold,
and the gold then scores the gate.

## Fine-tuning the reranker: stopped at the pre-registered gate

HiREC reports that its edge over flat retrieval on standardized filings came from
a cross-encoder fine-tuned on financial tables, and we run an off-the-shelf one,
so a LoRA pass over our own labels was the strongest literature-backed idea
available. The plan gated it on data volume before any training: at least ~300
usable questions that share no report with the benchmark, so the held-out 233
stays honest.

`propose_multihop_labels.py` over the 779 unlabelled questions returned 579
proposed, 89 needs_review, 111 no_match, and `accept_proposals.py` kept 312 of
them (easy 172, medium 70, intermediate 43, hard 27). Of those, only **189 share
no report with the benchmark**, and their tier mix is 128 easy, 41 medium, 16
intermediate and **4 hard** — the benchmark touches 466 of 1,182 gated reports,
so report-blocking removes most of the hard questions along with them.

Flipping which side absorbs the leak does better: training on all 312 and
evaluating on the 128 benchmark questions disjoint from them keeps 27 hard in
training and 18 in evaluation. It was still refused. 312 questions is roughly
1,130 positives, thin for LoRA to beat a strong 8B, and shrinking the evaluation
set from 233 to 128 widens the bootstrap interval exactly where the effects we
are chasing are ~0.02. The gate existed to prevent talking ourselves past it.

The proposals are kept at `annotations/train/` — queue, proposals and accepted,
each carrying a `report_disjoint` flag — so the decision can be revisited without
recomputing anything.

## The instrument became the constraint

A session spent on retrieval closed roughly twenty lines and opened one. The
common cause is not that the ideas were bad. It is that the 233-record benchmark
resolves a paired cluster-bootstrap delta to about **±0.02**, and almost
everything left to try is smaller than that. What follows is recorded so none of
it is run twice.

Live Tables F2 over four submissions, all on the same candidate sets:

| submission | rule | Tables F2 |
|---|---|---:|
| `submission_v4` | v4 representation, unweighted RRF | 0.5118 |
| `submission_v4_tiered` | tier-conditional replace | **0.5221** |
| `submission_v4_w03` | weighted RRF, w=0.3 | 0.5167 |
| `submission_v4_w00` | model order alone, w=0 | 0.5192 |

### Measured and rejected

- **Listwise reranking**, Borda over both presentation orders: −0.0458.
- **Fusion weight.** A swept `w=0.3` beat the shipped `w=0.5` on the benchmark
  and lost live, 0.5167 against 0.5221. The curve it was fitted from is jagged
  and the inner cross-validation folds overlap 75%, so "stable in four of five
  folds" was never the evidence it appeared to be. The live number settles it.
- **Ensembling** the 4B and 8B rankings, four variants: none clears the interval.
- **Coverage and submodular selection.** Only 3 of 233 questions have a budget
  whose composition could be improved by trading a redundant table for a missing
  one, which caps the whole idea below the noise floor.
- **Slot reallocation** between reports inside a fixed budget: +0.0044.
- **Budget by named line-item count**, five formulas. A swept version gave
  +0.0178; the principled version, predicting gold as
  `1.2 x reports + 1.0 x items`, gave **−0.0124**. The swept gain was noise.
- **Adaptive budget on score confidence**, and **per-report retrieval depth**:
  both inside the interval.
- **Period metadata** as a ranking signal: gold tables are 1.08x more likely to
  match the asked period than non-gold, which is not a usable margin.
- **Statement-type staging.** Routing by statement type is worth +0.0673 with an
  oracle label and **−0.1009** with the classifier we can actually build. This is
  the standing proof that an oracle bound motivates nothing on its own.
- **The proposer as a retriever.** `propose_multihop_labels.py` searches raw OCR
  with no retriever in the loop, so it is an independent ranker; scored as one it
  reaches 0.5632 to 0.6874 against the shipped pipeline's 0.7050 on the same
  questions.
- **A dense head over frozen E5 embeddings** — see the commit; the gold-to-non-gold
  cosine gap is +0.0160 and no head recovers a signal that small.
- **A hard-only gold set** as a sharper instrument: 14 records, and their gold is
  systematically wider than the main benchmark's.

### Two claims withdrawn

The tier-conditional rule is live-validated at +0.0103 and the *mechanism* offered
for it was not. Sparse score dispersion is flat across tiers (0.0049 to 0.0054)
and the gain from `replace` is non-monotonic in gated report count, so "the model
should take over where the sparse signal is weakest" is a story the data does not
tell. The rule ships on its measured result alone.

A filter dropping tables whose cells are all small integers was proposed as junk
removal. Inspection of what it caught showed subsidiary-ownership tables and
reserve-ratio tables, whose values are legitimately percentages. Withdrawn.

### Decomposition is stage-dependent, and 566fd47 does not close it

`566fd47` rejected lexicon query decomposition at −0.0302 F2, CI [−0.0553,
−0.0059], with intermediate at −0.1053 and only hard gaining. It is easy to read
that as "decomposition does not work here". It measured something narrower: the
sub-queries were fused into the *retrieval* ranking, and its own diagnosis —
reciprocal rank weights every signal identically, so a merely good signal dilutes
better ones — is dilution at the fusion stage.

arXiv 2606.08577 reports the same effect as a general finding, and separates it
by stage: decomposition during initial retrieval frequently harms retrieval
through semantic dilution, yet substantially improves reranking by enabling
finer-grained constraint verification. Their framework keeps the monolithic query
at retrieval and uses sub-queries only at reranking. Under that reading `566fd47`
is a confirmation, not a refutation, and the reranking-stage version — `PER_ITEM`
in `kaggle/rerank_qwen_8b.py`, one query per named line item reduced by `max` —
has never been run.

One difference matters. Their constraints are conjunctive, so they aggregate by
min or product; ours is disjunctive at table level, because a table counts if it
supplies any one of the named items. The stage finding transfers; the aggregation
does not, and `max` is what the task requires.

The headroom is small. On the benchmark, F2 by named item count is 0.6923 at one
item (n=149), 0.6115 at two (n=70) and 0.6731 at three (n=13), so the optimistic
ceiling — two- and three-item questions rising to the one-item level, nothing
regressing — is **+0.0254**, which is the resolution limit. `build_queries`
returns a byte-identical string for zero- and one-item questions under both
settings, so exactly 83 of 233 benchmark questions can move at all. That is why
the run goes straight to the full corpus and is judged live on 506 questions
rather than gated here.

Note that the corpus-wide gradient quoted while this was being planned (0.653 /
0.471 / 0.384 for one, two and three items) does not reproduce on the benchmark,
where three-item scores above two-item.

### What is actually binding

Representation beats capacity: changing the candidate representation was worth
+0.0428, doubling the model from 4B to 8B +0.0088. And the benchmark flatters
progress — it captures 43.6% of the oracle gap against 29.9% on retriever-free
labels — though the oracle *level* generalises (0.8020 against 0.7846), so it
still bounds correctly even where it over-credits movement.

FinRank (arXiv 2608.07400) measures hard-negative discrimination as a task in its
own right. Every model family drops from 88-96% pairwise accuracy on random
negatives to 70-80% on curated hard negatives. Their hard negatives are 80%
cross-company; ours are near-duplicate tables inside a single filing, which is
strictly harder. Nobody has solved this, so a large local gain from a better
ranker is not the reasonable expectation.

The constraint is the ruler, not the ranker. `benchmark_hard_deferred.jsonl`
carries 93 `branch_unresolved` and 80 `needs_review` questions; resolving them is
the highest-value work left, because it is what would let any of the rejections
above be revisited with enough resolution to mean something.

## The reranker does not score the same pair twice

`scores_bench_v4.jsonl` and `scores_bench_d100.jsonl` overlap on 11,592
(question, table) pairs. The candidate text is byte-identical on all 11,592 —
checked, not assumed — and both were scored by Qwen3-Reranker-8B in int8 with the
same prompt. **138 of the scores agree.** Mean |delta| is 0.0098 and the maximum
is 0.389.

Rebuilding the ranking from the same 50 candidates but the other run's scores
moves benchmark F2 by **+0.0076, CI [-0.0063, +0.0235]**. Nothing changed but the
session.

The cause is that int8 is not batch-invariant: bitsandbytes decomposes outlier
features per batch, and the batches are packed from whatever candidate set the
run holds, so a deeper export repacks the shallower one's pairs into different
batches. `QUANTIZATION=fp16` drops that path and should shrink the drift, though
not to zero — fp16 matmuls are not batch-invariant either, since cuBLAS picks
kernels by shape. It has never been run here: it needs the T4 x2 Kaggle offers,
and the 1.5-2x speed figure is a single-card comparison rather than a measurement
of this model split across two cards, where one idles while the other computes.

### What it costs the record

The depth-100 result was measured across two sessions and is therefore two
things at once. Decomposed:

| | benchmark F2 | vs the row above |
|---|---:|---|
| depth-50 pool, depth-50 run | 0.6508 | — |
| depth-50 pool, depth-100 run | 0.6584 | **+0.0076**, CI [-0.0063, +0.0235] — drift |
| depth-100 pool, depth-100 run | 0.6650 | **+0.0066**, CI [-0.0006, +0.0150] — the candidates |

So depth 100 is worth +0.0066, not the +0.0100 recorded in `ASSESSMENT.md` or the
+0.0142 the end-to-end comparison reads. The mechanism agrees: depth 100 adds
9,158 candidates of which 58 are gold (0.6%), only **3** ever reach the submitted
budget, and of the +16 net gold tables gained inside budget, **14 come from
re-ordering the original 50** — that is, from the drift. Deeper retrieval is
nearly all ceiling and almost no floor: the extra gold sits at sparse ranks
51-100 and the reranker does not lift it.

Every cross-session comparison in this file inherits the same ~0.008. The
4B-versus-8B result (+0.0088) is no longer distinguishable from it. The
v3-versus-v4 representation result (+0.0428) is large enough to survive.
Same-session comparisons — every fusion weight, budget rule, tier switch and
selection policy, all scored offline from one cached score file — are unaffected,
which is most of what this file records.

### The rule that follows

A ranking comparison is valid only when both sides come from one session, or when
the drift is measured beside it. `scripts/compare_rerank_runs.py` does the second:
raw agreement, an A/A stratum of the questions whose prompt is identical under
both settings, and the treated stratum. For `PER_ITEM` the split is free —
`build_queries` returns a byte-identical prompt for questions naming zero or one
line item, 150 of the benchmark's 233 — so the noise floor is measured inside the
real run at no extra cost.

That the strata are not self-evidently safe is worth seeing: on the pure-drift
pair above, the A/A stratum reads +0.0028 while the treated stratum reads
+0.0163. Both are noise. A treated-stratum gain is only a result once it clears
the A/A stratum measured in the same pair of runs.

### Why this is the most useful thing in the file

The ~+/-0.02 resolution that closed most of the ideas above was treated as a
property of a 233-record benchmark, and so as something only more labels could
fix. A third of it is drift, and drift is fixable in an afternoon by scoring both
cells in one fp16 session. That reopens effects in the 0.01-0.025 band, which is
where `PER_ITEM` sits at a +0.0254 ceiling.

## The cap was break-even, and it measured the tail

Three live submissions truncating the shipped table list at 20, 16 and 12:

| | Tables F2 | precision | recall |
|---|---:|---:|---:|
| uncapped, `min(30, 2 x reports)` | 0.5221 | 0.3468 | 0.6129 |
| cap 20 | 0.5223 | 0.3490 | 0.6119 |
| cap 16 | 0.5214 | 0.3506 | 0.6101 |
| cap 12 | 0.5186 | 0.3534 | 0.6058 |

The prediction was +0.016 to +0.030 and it was wrong by that much. The identity
F2 = 5h/(4G+k) was applied to corpus means, but the metric is macro and the
questions being cut are exactly the ones with large G, so the per-question rule
is what holds: cutting k to c pays only when the share of hits lost is below
(k-c)/(4G+k). For a k=30, G~7.9 question cut to 12 that tolerates losing 29% of
hits, and roughly that is what was lost. Break-even by construction.

What the three points bought is the first measurement of the population the
benchmark cannot see. Implied gold rate of the tables removed: positions 21-30
0.035, 17-20 0.076, 13-16 0.116. The tail is thin, and it still does not pay to
cut it, because 4G swamps the k removed. The budget rule is right; precision
has to come from ranking gold higher, not from submitting less.

## Where gold sits in a report

Ordinal position of each table within its report, benchmark, shipped ranking:

| | n | first fifth | rest |
|---|---:|---:|---:|
| gold we submit | 482 | 0.60 | 0.40 |
| non-gold we submit | 614 | 0.34 | 0.66 |
| gold we miss | 273 | 0.59 | 0.41 |

Primary statements come first, notes after. Gold is a statement 60% of the time;
the non-gold we submit is a note 66% of the time. And the shipped INSTRUCTION
ends "count a note or segment breakdown restating it as yes" — it asks for
exactly the tables that fill the slots after the first gold. By tier, gold in
the first fifth is easy 0.57, medium 0.44, intermediate 0.62, hard 0.62; the
medium gold in notes is ownership shares, credit limits, term deposits, fair
values, tax components — items no statement carries.

A single log-odds prior of +1.08 for first-fifth tables, measured on what we
submit and not swept, scores +0.0183, CI [+0.0038, +0.0350], easy +0.021,
intermediate +0.033, hard +0.022 — and medium -0.010. Gating it on whether the
model already ranks a statement in its top three does not rescue medium. So the
prior is real and a global rule for it is wrong: the item decides whether gold is
a statement or a note. That is a decision the model can make if it sees the
position and is not told the answer, which is the next GPU run — `pairs_bench_v5`
carries a position line, INSTRUCTION_V2 removes the note clause and states the
statement-or-note rule by item type.

## The proposer against the benchmark: the labels look right

The one clean local check on the benchmark's labels is a labeller that never saw
our retriever. `propose_multihop_labels.py` run over the 233 benchmark questions
proposes on 203, and against binding gold: 80 identical, 109 overlapping, 14
disjoint. It proposes 5.17 tables a question against gold's 3.26, 464 tables gold
does not carry and 76 gold tables it cannot find.

The 76 are, on inspection, proposer failures and not label errors: "lãi vay"
against a row reading "Trong đó: Chi phí lãi vay"; "dự phòng" too generic to
match; hard questions whose gold spans the balance sheet and income statement of
five reports while the proposer follows one phrase. The 464 extras are the
restatement net the widened gold definition already rejected. So the audit does
not find the benchmark wrong. It also cannot find it right — a labeller weaker
than the one under audit only bounds the disagreement from one side. Whether the
organizers' gold is ours remains unmeasurable without submitting labels, which is
not a system and is not done.

## The drift is batch packing, and the position line does not pay

Two things were measured in one pass of Kaggle sessions, and the second only
means anything because of the first.

**The drift.** `scores_ctrl.jsonl` and `scores_ctrl_nb2.jsonl` are the same
model, prompt, quantization and pairs file, scored in two different sessions on
two different days. They agree on 11,580 of 11,592 pairs, mean |delta| 1.8e-05,
max 0.11. The rebuilt rankings are not merely close: F2 is 0.6676 either way and
every tier delta is exactly 0.0000.

So the earlier finding — 138 of 11,592 identical, F2 moved +0.0076 by changing
nothing — was not about sessions. Those two runs held different candidate pools,
depth 50 against depth 100, and candidates are sorted by length before batching
because a batch costs its longest member. Different pool, different neighbours in
the batch, and int8 decomposes outlier features per batch. The session was
innocent; the packing was not.

What survives of the rule: a treatment that changes candidate text or the number
of queries repacks the batches and carries roughly 0.008 F2 of noise on top of
whatever it does. A treatment that leaves the pairs file byte-identical carries
none. That is a sharper instrument than "score both cells in one session", and it
says the 4B-versus-8B comparison (+0.0088, different models, necessarily
different packing) is still unreadable while a same-pairs re-run needs no control
at all.

**The position line and the notes instruction.** `pairs_bench_v5.jsonl` is
`pairs_bench_v4.jsonl` with one line inserted per candidate, `Vị trí: bảng N/M
trong báo cáo`, and it was scored under `INSTRUCTION_V2`, which tells the model
that primary statements come first in a report, that the position line says which
those are, and that a note restating a statement item is not the answer when the
statement itself is a candidate. The motivation was measured: 60% of gold sits in
the first fifth of a report against 34% of our submitted non-gold, and the
shipped instruction explicitly counts restating notes as yes.

Against its own-session control:

| stratum | control F2 | treatment F2 | delta |
|---|---:|---:|---:|
| all 233 | 0.6676 | 0.6644 | -0.0032, CI [-0.0164, +0.0106] |
| easy | 0.7487 | 0.7487 | 0.0000 |
| medium | 0.7144 | 0.6936 | -0.0208 |
| intermediate | 0.6526 | 0.6478 | -0.0048 |
| hard | 0.5441 | 0.5559 | +0.0118 |

Agreement with the control is 34 of 11,592 pairs, mean |delta| 0.0229 — the
change reached the model, it just did not help. The split is the one the position
prior predicted before the run: hard questions gain, medium loses, and medium is
where gold is note-only. Rejected under the standing rule twice over — the
interval covers zero and a tier regresses.

The two factors were bundled deliberately, so this does not say which half
failed; it says the bundle is not worth another session. What it does establish
is that the model already knows where in a report to look, and telling it costs
more on the questions whose answers are somewhere else.

## The corpus labels itself: account codes and the tables that name the item

Two structures sit in every filing and the pipeline used neither.

**Circular 200 numbers the line items.** Every Vietnamese filer presents the same
primary statements with the same mandated `Mã số`, so `110` is "Tiền và các khoản
tương đương tiền" in every company and every year, and the balance sheet also
carries a `Thuyết minh` column naming the note that details each row. Reading the
146,246 tables for those columns yields 10,389 primary statements over 1,965
reports, 494 codes and 8,859 label variants from 167,306 observations — a lexicon
of the line-item vocabulary, including its OCR damage, built from no labels of
ours. Every one of the 1,764 line items named across the 1,012 questions resolves
into it: 49% as an exact label, 51% as the prefix of a longer one, none missing.

The first version read only three-digit codes and found 4,330 statements. The
income statement and the cash flow statement number their rows `01` to `70`, so
two thirds of the primary statements were invisible; the fix was one character in
a regular expression and it more than doubled the corpus this rests on.

**The hops that follow from it did not pay.** Resolving the item to a code and
submitting the statements that carry it gives 1.97 tables per question at
precision 0.480, against 4.70 tables at 0.4521 for the shipped ranking — tighter,
not better, and unioning it in was worth +0.0059. Following the `Thuyết minh`
number to the note table was worse: only 13% of the 53,228 note references
resolve to a heading we can find, and the notes that do resolve are gold at 0.05.
Note headings are not reliably above their table; a running company name or the
tail of a paragraph often is.

**What did pay is cruder and general.** A table whose own first column contains
the asked line item — as the item, or as the start of a longer label the question
abbreviated — is evidence by construction, and the ranker never sees most of
them: BM25 proposes fifty candidates chosen through one matched row and the
budget then cuts to five. Adding those tables to the submission, in the gated
reports only, moves the benchmark:

| expanded tiers | F2 | k | delta | easy | medium | intermediate | hard |
|---|---:|---:|---|---:|---:|---:|---:|
| none | 0.6676 | 4.70 | — | — | — | — | — |
| hard | 0.6965 | 5.63 | +0.0289 [+0.0166, +0.0423] | 0 | 0 | 0 | +0.1322 |
| hard, medium | 0.6996 | 5.93 | +0.0321 [+0.0174, +0.0488] | 0 | +0.0151 | 0 | +0.1322 |
| every tier | 0.6923 | 6.73 | +0.0247 [+0.0039, +0.0462] | -0.0271 | +0.0151 | -0.0001 | +0.1322 |

The tables it adds are gold at **0.370**, against a break-even of F2/5 = 0.133,
which is why the gain grows monotonically with how many are added and no cap
earns its place — 2, 4, 6, 8, 12 and uncapped read +0.0124, +0.0198, +0.0257,
+0.0261, +0.0285, +0.0289. Easy questions are the exception: they are answered by
one table and a second only costs precision, so the shipped gate is hard and
medium. That is the same tier-conditional shape as the fusion rule already in
production.

Two cautions. The benchmark's gold is the set of tables a row or column binding
was found in, which is close to the definition this expansion searches for, so
the benchmark cannot be a clean test of it — the live P/R ratio matching binding
gold to within 0.009 is the reason to expect the effect to survive anyway, since
it says the organizers count the same kind of table. And it is a recall play in a
season spent on precision: it pushes k from 5.87 towards 9 live, away from
synera's 3.45, which is only correct because F2 weights recall four to one and
0.370 is nearly three times the break-even.

Everything regenerates from `scripts/build_account_lexicon.py` and
`scripts/build_item_expansion.py`, and `run.py --expand` appends the result to
`relevant_tables` alone — evidence, the CSVs and the answer path are untouched,
so execution accuracy cannot move.

## The expansion live: recall past synera, and the benchmark overstated it sevenfold

Both cells submitted the same day from the same code, differing only in
`--expand`:

| | Tables F2 | P | R | MRR@5 |
|---|---:|---:|---:|---:|
| control | 0.5238 | 0.3488 | 0.6145 | 0.6171 |
| expanded | 0.5283 | 0.3259 | 0.6582 | 0.6212 |
| synera | 0.6437 | 0.6293 | 0.6532 | 0.6456 |

Three things follow.

**The benchmark overstated the effect by a factor of seven.** It predicted
+0.0321, CI [+0.0174, +0.0488]; live returned +0.0045. The circularity warned
about when the expansion was built is the whole of it: benchmark gold is the set
of tables a row or column binding was found in, and the expansion searches for
tables containing the item, so the instrument was scoring its own definition.
`gold_tables_binding` cannot be used to evaluate any lexical item-matching rule
again.

**The tables it added are gold at about 0.055.** Recall moved +0.0437 against
G = 3.33, so the 2,663 added tables carried roughly 147 gold — half the
break-even of F2/5 = 0.105. The aggregate identity says the change should have
lost; macro F2 says it won by +0.0045, because the additions land on hard and
medium questions whose own F2, and therefore whose own break-even, sits below the
corpus mean. The margin is thin enough that the result is a small real gain and
not a mandate to add more.

**Recall is no longer the gap.** At 0.6582 we retrieve more of the gold than
synera's 0.6532, with MRR@5 within 0.024 of theirs. Their entire lead is
precision, 0.6293 against 0.3259, and the shape of it is now unambiguous: they
submit about 3.45 tables per question and we submit 8.50, and we find the gold
either way. Every remaining point is in the ordering — putting the 2.19 gold
tables we already hold inside the first three or four positions — and none of it
is in retrieving more, which is the direction this project has spent most of its
submissions on.

The control also re-measured the baseline against the refactor: 0.5238 today
against 0.5221 in the shipped era, so the 46 questions the contents-page filter
moved are worth +0.0017 and the two eras are comparable.

## Coverage ordering loses, and so does every cap

Two ideas for buying precision were tested against both gold definitions before
spending a slot, and both failed.

**Promoting a carrier of each named item.** A question naming two items gets four
tables about the first and one about the second, so cutting to a budget drops the
second item — that was the theory, and the fix was the standard one, a greedy
coverage pass over the ranked candidates. It loses everywhere:

| budget | binding gold | | full gold | |
|---|---:|---:|---:|---:|
| | ranked | covered | ranked | covered |
| shipped | 0.6676 | 0.6608 | 0.6562 | 0.6471 |
| cap 5 | 0.6462 | 0.6371 | 0.6284 | 0.6176 |
| cap 4 | 0.6227 | 0.6123 | 0.6059 | 0.5941 |
| cap 3 | 0.6034 | 0.5877 | 0.5806 | 0.5649 |

F2 at a fixed budget depends on the set, not its order, so this says the table a
promotion displaces is gold more often than the carrier it promotes. The ranker
already handles multi-item questions better than a lexical coverage rule does,
and the gap widens as the budget tightens — the opposite of the prediction.

**Any cap at all.** Truncating the shipped order costs 0.021 at cap 5 and 0.064
at cap 3 under binding gold, 0.028 and 0.076 under full gold, which agrees with
the live cap experiments and with the identity: F2 = 5h/(4G+k) means dropping a
table pays only when its chance of being gold is below F2/5, and ranks 4 to 6 sit
around 0.20 against a break-even of 0.121.

That closes the arithmetic on selection. With h = 2.048 at k = 5.87, reaching
synera's 0.6437 needs h = 2.17 at k = 3.55 or h = 2.47 at k = 5.87 — more gold in
fewer tables, which no reordering and no truncation of the current candidates can
produce. Either the candidates carry gold we are not ranking, or they do not
carry it at all, and nothing local can tell the two apart: the expansion added
2.6 tables per question and bought only 0.147 gold, which is candidate generation
saturating.

So the next submission is a measurement rather than an attempt. Submitting the
whole 50-candidate pool returns the pool's recall directly, and that is the
ceiling on every reranking idea left. Above it, ordering is worth chasing; near
0.66, the reranker is finished and the work moves to candidate generation.

## Six hand-written priors, all worse than the model

Chasing the 0.22 of ordering headroom the pool-recall probe exposed, every prior
we could name was implemented and measured on the benchmark under both gold
definitions:

| prior | delta |
|---|---|
| promote a carrier of each named line item | -0.008 |
| promote statements carrying the item's account code | -0.050 |
| listwise generation over the candidates | -0.046 |
| promote tables corroborated by their neighbours' figures | -0.039 |
| drop candidates that never mention the asked year | refuted by break-even |
| submit above a probability threshold instead of a budget | -0.022 at best |

Each was motivated, several by a real marginal association — a table sharing a
figure with three neighbours is gold at 0.330 against 0.132 for one sharing none,
and gold sits in the first fifth of a report 60% of the time. The associations
hold. The reorderings lose, because F2 at a fixed budget depends on the set, and
the table a promotion displaces is gold more often than the table it promotes.

The reading is that the cross-encoder already exploits these cues — position,
wording, account structure, corroboration — so a hand-written feature does not
add to it, it competes with it. Two levers survive, and both change what the model
scores rather than how its output is reordered: putting the item-carrying tables
into the candidate set, and training the model on figure-linked supervision.

Every figure above rests on labels we wrote ourselves, and that instrument has
already overstated one lexical rule by a factor of seven. They are reasons to stop
pursuing something, never grounds for a claim.
