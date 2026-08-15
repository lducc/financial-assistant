# Assessment: the v4 pointwise run, the listwise experiment, and what to spend GPU on next

**Date:** 2026-08-14
**Branch:** `feat/cleanup`
**Measured on:** `annotations/benchmark.jsonl`, 233 records / 184 report clusters,
`--gold binding`, shipped budget (`table_budget`: two per gated report, capped at 30)

Three questions were open. All three are now answered, one of them against the
hypothesis that motivated the work.

| question | answer |
|---|---|
| Does the v4 candidate representation beat the v3 one that produced live 0.486? | Yes. +0.0301 F2, CI excludes zero, no tier regresses. **Submitted as `output/submission_v4`.** |
| Does listwise reranking help on top of it? | No. −0.0458 F2, CI excludes zero, every tier down. Rejected; no package built. |
| What should the next GPU run be? | Pointwise at depth 100 over the full corpus, not `windows_full`. Depth 100 is +0.0100 with the CI excluding zero, and it attacks the one loss reranking cannot reach. **Superseded by §8:** +0.0100 is +0.0066 of depth and +0.0076 of cross-session drift, so the next run is the `PER_ITEM` A/B in one fp16 session. |

---

## 1. Where the working tree came from

This machine's checkout was at `b3a6582` (2026-08-12): no `kaggle/`, no
`src/vifinqa/listwise.py`, no `scripts/apply_*`, empty `output/`. Nothing was
lost — `origin/feat/cleanup` held `33e7575`, "commit the whole output tree for
machine transfer", and `b3a6582` is a strict ancestor of it, so a fast-forward
restored every script and the full derived tree. Two files arrived out of band
and are the only new inputs:

| file | what it is |
|---|---|
| `output/rerank/scores_v4.jsonl` | Qwen3-Reranker-8B pointwise over all 1,012 questions on the v4 inventory representation |
| `output/rerank/orders_bench{,_rev}.jsonl` | the listwise run, 233 records in each presentation |

Integrity checks after the restore:

```
pytest                          54 passed  (52 existing + 2 new)
verify_benchmark.py             VALID, 957 bindings, 0 failures
audit_entity_resolution.py      0 unsupported resolutions, 2 unresolved questions
candidate-set reproduction      233/233 traces match every ranking file exactly
```

One environment gap: `pandas` is imported by
`test_emitted_query_reproduces_the_answer_through_pandas` but was not declared in
`pyproject.toml`, so the suite arrived at 51 passed / 1 failed on a clean
machine. Installed and declared.

### The handed-off scores are a different run from the committed bench scores

`scores_v4.jsonl` and the already-committed `scores_bench_v4.jsonl` cover the same
233 questions with **identical candidate sets** — so the representation and the
pairs export agree — but they are not the same GPU run:

| comparison | value |
|---|---|
| questions with an identical full ordering | 0 / 233 |
| questions agreeing on the top-1 table | 189 / 233 (81%) |
| mean top-5 overlap | 0.913 |
| max absolute score difference | 0.349 |

That is worth knowing on its own: **run-to-run variance for the same
representation is worth 0.0041 F2** (0.6508 against 0.6549). Any future delta
smaller than that is a re-roll of the same dice, not a result.

---

## 2. The benchmark table

Every row scores the same 233 cached retrievals; only the ordering differs, so the
comparisons are paired per question. Shares are of the 755 gold tables.

| ranking | F2 | recall | precision | MRR | hit | rank_miss | candidate_miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse (BM25, `report-coverage`) | 0.5637 | 0.6618 | 0.3779 | 0.6588 | 49.7% | 33.8% | 16.6% |
| v3 representation, 8B fuse — **this is live 0.486** | 0.6248 | 0.7330 | 0.4200 | 0.7510 | 57.2% | 26.2% | 16.6% |
| v4 representation, bench GPU run | 0.6508 | 0.7621 | 0.4413 | 0.7449 | — | — | 16.6% |
| **v4 representation, full GPU run — shipped** | **0.6549** | 0.7679 | 0.4417 | 0.7621 | 61.1% | 22.4% | 16.6% |
| listwise Borda over both passes | 0.6050 | 0.7087 | 0.4100 | 0.7263 | 56.4% | 27.0% | 16.6% |
| pointwise at depth 100 (bench only, not buildable) | 0.6650 | 0.7792 | 0.4495 | 0.7621 | 62.8% | 28.3% | **8.9%** |

`gate_miss` is 0.0% on every row: the document gate never loses a gold table on
any of the 233 records. That is consistent with Docs F2 0.9711 live and with the
entity audit finding zero unsupported resolutions, and it means document
retrieval is closed — no work there is worth doing.

### Reproduction gates do not quite hold

`docs/REBUILD.md` states 0.6562 for `ranking_bench_v4.json` and 0.6692 for
`ranking_bench_d100.json`. Measured here: **0.6508** and **0.6650** — both about
0.005 low, the same offset in both. Retrieval is not the cause; candidate sets
reproduce exactly, 233/233, against all three ranking files. A sweep over both
gold definitions and four budgets does not land on either published figure, so the
original numbers came from a protocol this repo no longer contains (they were
produced by inline commands, not a committed script). Every number in this
document was produced by `scripts/diagnose_retrieval.py` and
`scripts/compare_rankings.py` under one protocol, so the comparisons stand; the
two absolute figures in REBUILD should not be trusted, and have been corrected
there.

---

## 3. The three decisions

The rule is `docs/benchmark.md`'s, fixed in advance: accept only when the mean
improves, the paired cluster bootstrap CI on the delta excludes zero, and no
difficulty tier regresses.

### v4 over v3 — ACCEPT, and it is what shipped

| | delta F2 | CI95 |
|---|---:|---|
| overall | **+0.0301** | [0.0057, 0.0547] |

| slice | baseline | candidate | delta |
|---|---:|---:|---:|
| easy (63) | 0.7381 | 0.7487 | +0.0106 |
| medium (48) | 0.6649 | 0.7144 | +0.0495 |
| intermediate (71) | 0.5864 | 0.6126 | +0.0262 |
| hard (51) | 0.5005 | 0.5421 | +0.0416 |
| gold-150 slice (150) | 0.6482 | 0.6787 | +0.0305 |
| v3 slice (83) | 0.5825 | 0.6120 | +0.0295 |

31 questions improved, 12 regressed. Every tier up, and the two label sources
move together — which matters, because the v3 records were found by folded
substring search with no retriever in the loop and are the less flattering half.
A gain that appeared only on gold-150 would be suspected of tracking the public
submission its evidence was seeded from; this one does not.

Mechanically the gain is exactly what reranking is supposed to do: `rank_miss`
falls from 26.2% to 22.4% of gold tables while `candidate_miss` sits unmoved at
16.6%. The reranker promotes evidence already in hand and cannot do anything else.

### Listwise — REJECT

| | delta F2 | CI95 |
|---|---:|---|
| overall | **−0.0458** | [−0.0880, −0.0031] |

| slice | pointwise | listwise | delta |
|---|---:|---:|---:|
| easy (63) | 0.7566 | 0.6772 | −0.0794 |
| medium (48) | 0.7161 | 0.6684 | −0.0477 |
| intermediate (71) | 0.5912 | 0.5467 | −0.0445 |
| hard (51) | 0.5414 | 0.5372 | −0.0042 |
| gold-150 (150) | 0.6804 | 0.6022 | −0.0782 |
| v3 (83) | 0.5973 | 0.6101 | +0.0128 |

36 improved, 57 regressed. The v3 slice is the one place it gains, and at 83
records that sits inside the noise the same bootstrap assigns elsewhere; it is not
enough to rescue a result that is negative overall and negative in every tier.

**The aggregation is not what failed.** Scoring each presentation alone:

| pass | F2 |
|---|---:|
| forward order | 0.5842 |
| reversed order | 0.6046 |
| Borda over both | 0.6050 |

Borda beats both passes, which is what it was built to do, and it still lands
0.046 below the pointwise ranking it replaced. `mean_pass_agreement` is **0.4976**
over the 229 questions where both passes produced a usable order (3 produced
none). That is the middle of the pre-registered table rather than either end: the
model is not merely echoing the order it was shown — at 0.50 it has a real and
partly consistent opinion — but its opinion is worse than the pointwise
probability it displaced.

So the reading is neither "needs a smaller window" nor "position bias ruined it".
Reordering 20 candidates by generating a permutation is a harder task for this
model than scoring one candidate at a time, and 8B off the shelf is not good
enough at it. That is the case R6 (fine-tuning) was always meant to answer, and
it is now the only listwise route left. **`windows_full.jsonl` and
`windows_full_rev.jsonl` should not be run** — 3 to 19 GPU-hours to apply a
measured regression to all 1,012 questions.

### Depth 100 — ACCEPT the finding, but it is not buildable today

> **Corrected in §7.** This comparison puts a depth-50 run against a depth-100
> run, and cross-encoder scores are not reproducible across Kaggle sessions. The
> drift alone is worth +0.0076; the depth-100 candidates are worth **+0.0066**.
> Everything below is the sum of the two.

| | delta F2 | CI95 |
|---|---:|---|
| overall, against the shipped v4 ranking | **+0.0100** | [0.0014, 0.0206] |

No tier regresses (medium is flat at 0.0000, the rest gain 0.012–0.013), and the
v3 slice gains most (+0.0229 against +0.0029 on gold-150). 13 improved, 5
regressed. The interval is tight and barely excludes zero — this is a small, real
effect, not a large one.

Its importance is structural rather than in the point estimate. Depth 100 is the
only lever measured here that moves `candidate_miss`, which **halves from 16.6% to
8.9%**. That share is the ceiling on reranking: no ordering of a candidate list
can submit a table the list does not contain. The loss is concentrated exactly
where the benchmark says it always was —

| question size | n | F2 at depth 50 | candidate_miss at depth 50 |
|---|---:|---:|---:|
| 1 table | 80 | 0.7604 | 0.0% |
| 2 tables | 62 | 0.6809 | 0.0% |
| 3–5 tables | 61 | 0.6073 | 6.9% |
| 6+ tables | 30 | 0.4168 | 35.3% |

— because a single ranked depth of 50 is shared across as many as 30 gated
reports, leaving almost nothing per report on the largest questions.

`pairs_v4_d100.jsonl` (the full-corpus depth-100 export, 89 MB) is already on
disk. What does not exist is a GPU run over it: only `scores_bench_d100.jsonl`,
233 questions, exists. **No depth-100 submission can be built, and none was.**

---

## 4. What shipped

`output/submission_v4/package`, built by

```bash
.venv/bin/python run.py --ranking output/rerank/ranking_v4_fuse.json \
    --output-dir output/submission_v4
.venv/bin/python scripts/validate_submission.py output/submission_v4/package
```

from `ranking_v4_fuse.json`, which is `pairs_v4.jsonl` fused with the handed-off
`scores_v4.jsonl` by reciprocal rank (441 of 1,012 questions get a new top table).
`validate_submission.py` reports `VALID`; 1,012 rows, 5.87 tables per question
(min 2, max 30), no empty table list.

Against the package behind the live 0.486 (`output/submission_qwen8b`):

| | count of 1,012 |
|---|---:|
| identical document set | 1,012 |
| identical table set | 386 |
| identical table set *and* order | 265 |
| same top table | 664 |

Documents are untouched by construction — the reranker only permutes tables — so
Docs F2 should not move from 0.9711. The submitted table set changes on 626
questions. Live Tables F2 is not predicted here:
the benchmark and the organizers' gold are different label sets, and public covers
506 of 1,012 questions while ranking on execution accuracy. The benchmark says the
ordering is better by +0.0301 with the interval excluding zero; that is the claim,
and the live number is the test of it.

The listwise package was not built. The benchmark rejected the change, and a
package that applies a measured regression to 233 of 1,012 questions is worse than
the one that does not.

---

## 4b. The live result, and what it closed

`submission_v4` scored **Tables F2 0.5118** against the previous 0.486. The
benchmark predicted +0.0301 and live delivered +0.0258 — a transfer ratio of
0.86, and the only live A/B this instrument has ever been checked against. Docs
held at 0.9711 exactly, as predicted, because the reranker only permutes tables.

| metric | live |
|---|---:|
| TABLES_F2MACRO | 0.5118 |
| TABLES_PRECISION / RECALL / MRR5 | 0.3373 / 0.6016 / 0.5911 |
| DOCS_F2MACRO | 0.9711 |
| ANSWER_ACCURACY | 0.1561 |
| EXECUTION_ACCURACY | 0.1285 |

Live precision and recall pin the organizers' gold cardinality directly. With
`F2 = 5h/(4G+k)`, precision 0.3373 at k = 5.87 gives h = 1.98 hits and
**G = 3.29 gold tables per question** — against 3.24 in our binding gold, which is
why that definition and not the restatement-widened one reproduces live.

Three levers were tested against this and closed:

**Budget — at its optimum.** Sweeping the benchmark: 1 per report 0.5828, two
0.6549, three 0.6365, four 0.5818; every fixed budget is worse than every
report-proportional one. Live agrees independently: break-even marginal precision
is `h/(4G+k)` = 0.104, and going two→three per report yields 0.078. Not worth it.

**Slot reallocation — no.** With `--ranking`, `run.py` takes a flat top-k from the
reranked order, discarding the per-report reservation retrieval applied. Giving
those same k slots back to report structure: one-per-report-first 0.6565,
round-robin 0.6593, cap-two-per-report 0.6593, against flat 0.6549. The best is
+0.0044 — numerically identical to the round-robin experiment already recorded in
`select_report_coverage`'s docstring, whose CI was [−0.0065, +0.0152]. Starvation
is real and still lives beyond the budget, where reordering cannot reach it.

**`annotations/train/accepted.jsonl` is not an eval set.** 312 questions, disjoint
from the benchmark, discovered by folded row-label search with no retriever in
the loop — which made it look like the unbiased instrument the benchmark is not.
It is not, and the reason is worth recording. Its gold is *un-narrowed*: the
binding definition removes nothing on any of its 312 records (3.63 tables per
question either way), while on the benchmark it narrows 97 of 233 records from
4.50 to 3.24. Against the organizers' 3.29 the benchmark matches and this set is
10% over-wide, and over-wide gold counts every restatement as correct — which
flattens exactly the distinction a reranker makes. Measured: on this set the v3
and v4 rankings tie at 0.5815 and 0.5823, a delta of +0.0008, where live moved
+0.0258. It is good training data and a bad ruler. Narrowing it the way
`complete_benchmark_labels.py` narrowed the benchmark would fix that.

## 4e. Three live results, and what they establish about the answer path

| submission | Tables F2 | precision | recall | MRR@5 | Docs F2 | Answer | Execution |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 representation | 0.4860 | — | — | — | 0.9711 | 0.1601 | 0.1245 |
| v4 representation | 0.5118 | 0.3373 | 0.6016 | 0.5911 | 0.9711 | 0.1561 | 0.1285 |
| v4 + tier-conditional | **0.5221** | 0.3468 | 0.6129 | 0.6171 | 0.9711 | 0.1561 | 0.1285 |

The tier-conditional change delivered **+0.0103** live. The benchmark predicted
+0.0126 and the frozen test half +0.0087, so the true effect fell between the two
estimates — the instrument bracketed it. MRR@5 moved most (+0.0260), which is
what dropping the sparse prior on hard and intermediate questions should do:
`replace` puts the model's own first choice at rank 1.

Docs F2 has been 0.9711 across all three, to four decimals, because none of this
touches the gate.

**Table retrieval and the ranked metric are decoupled, and these three points
prove it on our own system.** Tables F2 rose 0.036 across the three; answer
accuracy fell 0.004 and execution accuracy rose 0.004. Between the second and
third submissions, 326 of 1,012 answers changed value and 428 changed query, and
the score landed on exactly the same 79 of 506 correct — equal numbers flipped
each way.

The mechanism is in `run.py`: evidence values come from `first_numeric_cell` on
the retrieved row, with no column chosen by the year asked and no check that the
row label is the metric. A better table hands that heuristic a different
arbitrary cell, not a better one. Until something binds the cell, table retrieval
improvements convert to noise in the answer, which is what the table above shows
happening three times.

This does not argue against finishing the retrieval work — it bounds what that
work can buy. Tables F2 is a scored metric in its own right and we are second
among real systems on it. Execution accuracy is the rank metric and it is
downstream of a heuristic that no ranking improvement can reach.

## 4d. Fusing two rerankers does not work; conditioning the fusion does

Four ways of ranking twice, all measured on the benchmark against the shipped
0.6549:

| variant | F2 |
|---|---:|
| same model, two independent runs, score-mean | 0.6540 |
| same model, two independent runs, rank-average | 0.6544 |
| two representations (v4 + v3), score-mean | 0.6422 |
| listwise second stage over the head | 0.6050 |
| multi-model ensemble | recorded earlier: recall 0.7932 → 0.7855, live 0.694 → 0.669 |

None beats the better single ranking. The mechanism is consistent: fusing the
reranker with *sparse* works because BM25 and a cross-encoder fail on different
questions, while fusing it with *another cross-encoder pass* interpolates, because
two passes of the same family fail on the same questions. Two runs of the same
model agree on only 81% of top-1 tables, but that disagreement is not zero-mean
noise around a better answer, so averaging just averages.

What does work is choosing **whether** to fuse, per question. `replace` drops the
sparse prior entirely. On an easy question — one table, one report, an exact
label match — that prior is real information and dropping it costs. On an
intermediate or hard question spanning many gated reports full of near-duplicate
rows the sparse rank is close to arbitrary and only dilutes the model. The
benchmark splits exactly where that argument predicts:

| policy (depth 50) | F2 | delta | CI95 | tier deltas |
|---|---:|---:|---|---|
| shipped, fuse everywhere | 0.6549 | — | — | — |
| replace on intermediate only | 0.6671 | +0.0122 | [−0.0023, +0.0275] | easy 0, medium 0, hard 0, **intermediate +0.040** |
| replace on hard+intermediate | 0.6676 | +0.0126 | [−0.0080, +0.0329] | easy 0, medium 0, hard +0.002, intermediate +0.040 |
| replace everywhere | 0.6604 | +0.0055 | [−0.0247, +0.0347] | easy +0.003, medium −0.038 |
| **replace on hard+intermediate, depth 100** | **0.6847** | **+0.0298** | **[+0.0079, +0.0537]** | easy +0.013, medium 0, hard +0.029, intermediate +0.065 |

Only the last clears the standing rule, and it needs the depth-100 scores. That
changes what the depth-100 run is for: not the +0.0100 it is worth alone, but the
thing that makes this combination decidable.

**Corrected in §7.** The last row compares scores from two Kaggle sessions; the
rows above it are all scored offline from one file and are unaffected. Holding
the policy fixed at replace-on-hard+intermediate and moving only the scores:

| | benchmark F2 | vs the row above |
|---|---:|---|
| depth-50 pool, depth-50 run | 0.6700 | — |
| depth-50 pool, depth-100 run | 0.6763 | +0.0062, CI [−0.0059, +0.0206] — drift |
| depth-100 pool, depth-100 run | 0.6847 | +0.0084, CI [−0.0017, **+0.0212**] — candidates |

The depth component does not clear zero. The +0.0298 above is measured against
plain fuse at depth 50, so it is the policy, the depth and the drift together;
none of the three is established on its own by it, and the combination is not
shipped.

**Stated plainly because it matters more than the headline:** the policy was
chosen after looking at tier deltas on this data. On the frozen test half — 65
records, never used to choose anything — the same policy is worth **+0.0087**,
against +0.0403 on dev. That gap is what selection-on-dev looks like, and the
small number is the expectation. The reason to act on it anyway is the mechanism,
which was not fitted, plus the fact that the live scorer is deterministic over 506
questions and settles a ±0.03 benchmark uncertainty outright.

Shipped as `--replace-tiers` on `apply_rerank_scores.py`, and **since superseded**
— see below.

### The tier switch does not survive being selected honestly

It reached live 0.5221, so it is not wrong. But it is a step function on a
classifier, and the subset `{hard, intermediate}` was chosen by eye from the
sixteen available. Selecting *both* families inside the cross-validation folds,
so neither gets to see the data it is scored on:

| family | candidates | out-of-fold F2 | vs no choice |
|---|---:|---:|---:|
| fixed w=0.5 | 1 | 0.6549 | — |
| **single global weight** | 10 | **0.6615** | **+0.0066** |
| tier subset × replace | 16 | 0.6569 | +0.0019 |

The tier family spends its extra freedom on overfitting: it barely beats making
no choice at all. What the folds picked settles it —

```
fold 0  w=0.3  |  {hard, intermediate}          fold 3  w=0.3  |  {intermediate}
fold 1  w=0.1  |  {easy, hard, intermediate}    fold 4  w=0.3  |  {easy, hard, intermediate}
fold 2  w=0.3  |  {easy, intermediate}
```

— the weight lands on 0.3 in four folds of five; the shipped subset is chosen in
one of five. One rule is stable under resampling and the other is not.

`--weight 0.3` replaces it. Not on score: the two sit inside the ±0.02 floor of
each other (CI on the difference [−0.0279, +0.0189]) and no live claim is made.
On assumptions. The weight is continuous, so no two near-identical questions
straddling a tier boundary get opposite treatment; it is one parameter chosen by
a standard procedure rather than one subset of sixteen chosen by eye; and it
drops the dependency on `classify_questions.py`, whose tier counts deviate from
the organizers' published ones on 162 of 1,012 questions. That dependency is a
liability on the private half with no demonstrated benefit on the public one.

Benchmark at w=0.3: F2 0.6714, recall 0.7866, MRR 0.7706, and no tier below the
tier-conditional version except medium by 0.008.

## 4c. Where the remaining headroom is

An oracle that reorders the candidates retrieval **already returns** — no deeper
pool, no larger budget:

| ordering | F2 | recall |
|---|---:|---:|
| sparse | 0.5637 | 0.6618 |
| v4 reranked (shipped) | 0.6549 | 0.7679 |
| **oracle over the depth-50 pool** | **0.8020** | 0.9295 |
| oracle over the depth-100 pool | 0.8255 | — |

**+0.147 F2 sits inside the ranking stage**, roughly +0.13 live at the observed
0.86 transfer. Deeper retrieval adds 0.023 on top of that, so depth 100 — worth a
measured +0.0100 with the real GPU scores — is a rounding error beside it. Every
zero-shot lever around the ranker is now spent: budget optimal, reallocation
+0.0044 with the interval covering zero, listwise −0.0458.

That leaves fine-tuning, which is what `rank_miss` (22.4% of gold, the largest
addressable state) has always meant. The data exists and the split is clean:
train on `accepted.jsonl`'s 311 usable questions, measure on the 233 benchmark
questions they are disjoint from.

```
scripts/export_rerank_training.py   311 questions, 1,069 positives,
                                    4,959 hard negatives, 6,028 rows
                                    64 gold tables (5.6%) lie outside the
                                    pool and are excluded, not counted
kaggle/train_reranker.py            LoRA, group-softmax on the yes-minus-no
                                    margin — the same quantity inference ranks on
kaggle/rerank_qwen_8b.py            ADAPTER_PATH re-scores the pool with the
                                    adapter attached; prompt unchanged, so a
                                    tuned run stays comparable to every
                                    committed score file
```

## 5. What to do next, in order

1. **Fine-tune the reranker.** Export is built and the split is clean. Run
   `kaggle/train_reranker.py` at 4B first — it is the cheap test of whether tuning
   moves anything at all — then re-score `pairs_v4.jsonl` with `ADAPTER_PATH` set,
   fuse, and compare against `ranking_v4_fuse.json` on the benchmark under the
   standing rule. This is where +0.147 of oracle headroom lives.
2. **Run pointwise 8B over `pairs_v4_d100.jsonl`** if GPU time is spare. Measured
   +0.0100 — corrected to **+0.0066** in §7, the rest being cross-session drift;
   the only lever that touches `candidate_miss`, and the only one that raises the
   oracle ceiling (0.8020 → 0.8255). It raises the ceiling far more than the
   floor: 3 of the 9,158 candidates it adds reach the submitted budget.
3. **Do not run `windows_full*.jsonl`.** Listwise is measured negative.
4. **Narrow `accepted.jsonl`'s gold** if it is ever to be used as a ruler rather
   than as training data — see §4b.
5. **Execution accuracy remains the largest untouched score.** Live 0.1285 against
   the leader's 0.660, and it is the primary rank metric — out of scope by
   standing decision, and stated here so the size of it is on the record.
   Correlation across the public board puts Docs F2 as the main execution driver
   (r ≈ 0.63) and Tables F2 as weak (r ≈ 0.19), so further table-retrieval work
   has a low ceiling on the metric that actually ranks.

## 6. How to reproduce any number here

```bash
.venv/bin/python scripts/diagnose_retrieval.py --refresh                       # depth-50 traces
.venv/bin/python scripts/diagnose_retrieval.py --ranking output/rerank/ranking_v4_fuse.json
.venv/bin/python scripts/compare_rankings.py \
    --baseline output/rerank/ranking_8b_fuse.json \
    --candidate output/rerank/ranking_v4_fuse.json
```

`--ranking` on `diagnose_retrieval.py` and `scripts/compare_rankings.py` are new
in this pass; both re-sort cached candidates rather than retrieving again, so a
ranking can never introduce a table retrieval did not find, and every row above
scores one retrieval.

## 7. Where this ends, and what supersedes §5

§5 was written before the session's remaining ideas were measured. All of them
are now closed — listwise, four fusion-weight and four ensembling variants,
coverage selection, slot reallocation, five budget formulas, per-report depth,
period metadata, statement-type staging, the proposer-as-retriever, a hard-only
gold set, and a dense head over frozen E5 embeddings. Each is recorded with its
interval in `research-history.md` under "The instrument became the constraint".

The ordering in §5 changes on one point. `PER_ITEM=1` — one reranker query per
named line item, reduced by `max` — moves ahead of fine-tuning, because it has
independent support that fine-tuning does not: arXiv 2606.08577 finds
decomposition harms at the retrieval stage and helps at the reranking stage, which
makes this repo's own `566fd47` (−0.0302, fused into *retrieval*) a confirmation
rather than a refutation. It is run over all 1,012 questions rather than gated on
the benchmark first, because the benchmark's optimistic ceiling for it is +0.0254
against a ±0.02 resolution and only 83 of 233 questions can move.

Everything in §5 that was not superseded still stands, including item 5: table
retrieval has a low ceiling on the metric that actually ranks.

The honest summary of the position: the ranker is not obviously the problem, and
the ruler is. FinRank (arXiv 2608.07400) shows every model family losing 13 to
20.5 points going from random to hard negatives, and our negatives are
near-duplicate tables from inside one filing. The 173 deferred questions in
`annotations/benchmark_hard_deferred.jsonl` are the highest-value remaining work,
because a benchmark that resolves better than ±0.02 is what would make any further
ranking decision possible.

## 8. The reranker is not reproducible, and it re-reads several rows above

Corrects §5 item 2, the depth-100 section, and the tier-conditional table.

`scores_bench_v4.jsonl` and `scores_bench_d100.jsonl` share 11,592
(question, table) pairs. The candidate text is byte-identical on all of them and
both were scored by the same model in int8 with the same prompt. **138 of the
scores agree**, mean |delta| 0.0098, max 0.389. Rebuilding the ranking from the
same 50 candidates but the other run's scores moves benchmark F2 by **+0.0076,
CI [-0.0063, +0.0235]**.

§1 already noticed this and put it at 0.0041 F2, unpaired, as a caveat on future
deltas. Paired per question it is +0.0076, and the point of §8 is that the caveat
was never applied: the depth-100 result on which §5's GPU plan rests is a
cross-session comparison and spends most of that on drift.

int8 is not batch-invariant — bitsandbytes decomposes outlier features per batch,
and batches are packed from whatever candidate set a run holds — so a deeper
export repacks the shallower one's pairs. `QUANTIZATION=fp16` is exact and
1.5-2x faster on Turing; it has never been used here only because it needs a
T4 x2, which Kaggle offers.

What it changes:

- **Depth 100 is +0.0066, not +0.0100.** The rest was drift. Under the
  tier-conditional policy the depth component is +0.0084, CI [-0.0017, +0.0212],
  which does not clear zero.
- **Depth 100 raises the ceiling far more than the floor.** It adds 9,158
  candidates of which 58 are gold (0.6%); 3 reach the submitted budget; and of
  the +16 net gold tables gained inside budget, 14 come from re-ordering the
  original 50 rather than from any new candidate.
- **4B versus 8B (+0.0088) is no longer distinguishable from drift.** v3 versus
  v4 (+0.0428) is.
- Everything scored offline from a single score file — every fusion weight,
  budget rule, tier switch, selection policy and the listwise result — is
  unaffected. That is most of §5.

`scripts/compare_rerank_runs.py` is the instrument: raw agreement, an A/A stratum
of the questions whose prompt is identical under both settings, and the treated
stratum. On the pure-drift pair above the A/A stratum reads +0.0028 and the
treated stratum +0.0163, both noise — which is why the treated stratum has to
clear the A/A stratum measured in the same pair before it counts.

The consequence is larger than any single correction. The ~±0.02 resolution that
closed most of this project's ideas was treated as a property of a 233-record
benchmark, fixable only with more labels. A third of it is drift, removable by
scoring both cells in one fp16 session. That reopens the 0.01-0.025 band, which
is where `PER_ITEM` sits at a +0.0254 ceiling — so the next run is that A/B, not
a bigger model and not more depth.
