# ViFinQA table-retrieval benchmark

192 questions, 493 gold tables, 613 verified cell bindings. Every table ID,
row, column, and raw cell string resolves in the raw OCR corpus.

This measures table retrieval only: which tables a system must return. It says
nothing about answers, Pandas, or units.

## Files

| Path | What it is |
|---|---|
| `annotations/benchmark.jsonl` | the benchmark; one record per question |
| `annotations/benchmark_manifest.json` | counts, distributions, corpus hash, file hash |
| `annotations/gold_150.jsonl` | seeded source set, kept as-is |
| `annotations/v3/labels_*.jsonl` | independently discovered set, plus deferrals with reasons |
| `annotations/v3/queue.jsonl` | the remaining 230-question annotation queue |
| `data/derived/question_tiers.jsonl` | difficulty tier for all 1,012 questions |

Rebuild with `python3 scripts/build_benchmark.py`.

## Composition

| Tier | Questions | from gold-150 | from v3 |
|---|---:|---:|---:|
| easy | 63 | 46 | 17 |
| medium | 48 | 39 | 9 |
| intermediate | 54 | 42 | 12 |
| hard | 27 | 23 | 4 |

By question size: 76 need one table, 59 need two, 50 need three to five, 7 need
six or more. 160 independent report clusters — that count, not the question
count, is what confidence intervals are computed over.

Tiers follow the organizer's four difficulty levels and are assigned by
`scripts/classify_questions.py` using the same question parser the retrieval gate
uses. Against the organizer's published counts the absolute deviation is 162 of
1,012; the tiers are not tuned to close that gap. They hold up against evidence
they never saw: mean gold tables per question runs 1.00 / 2.08 / 3.50 / 6.35
across the four tiers.

## Known bias, stated because it changes how you read recall

78% of records come from gold-150, whose candidate evidence was seeded from
public submission 2333 before every operand was relocated in raw source. Of its
420 gold tables, 181 (43%) also appear in that submission's candidates, so the
rest were found independently — but tables no retriever ever surfaced are still
under-represented.

The `v3` records exist to counter that. They were discovered by folded substring
search over raw OCR with no retriever involved, and `source` on every record says
which set it came from, so any result can be split both ways. Where the two
disagree, treat the v3 slice as the less flattering and more trustworthy read.

Multi-hop records carry `provenance.branch_resolution`: the computation that
decided which year or company the question selects, written out so the label can
be rechecked without redoing the search.

## Verify it

```
python3 scripts/verify_benchmark.py
```

Re-derives every claim from raw OCR: schema completeness, unique IDs, table IDs
that parse and exist, each bound cell still holding its exact recorded string,
gold tables and gold reports agreeing, and the corpus hash matching the manifest.
Exits non-zero on any failure, so it can gate a change. Current status: `VALID`,
613 bindings, 0 failures.

## Diagnose a system with it

```
python3 scripts/diagnose_retrieval.py --table-mode report-coverage
```

Every missed gold table is attributed to the stage that lost it, which points at
different work:

| State | Meaning | Where the fix lives |
|---|---|---|
| `gate_miss` | the report never entered the document gate | question parser, gate |
| `candidate_miss` | report gated, table never reached ranked depth | candidate generation, matching |
| `rank_miss` | retrieved but below the submitted budget | ranking, budget |
| `hit` | submitted | — |

Current system (`report-coverage`, three tables per gated report):

| Slice | n | F2 | fully covered | hit | rank_miss | candidate_miss | gate_miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 192 | 0.546 | 0.65 | 0.59 | 0.20 | 0.21 | 0.00 |
| easy | 63 | 0.672 | 0.94 | 0.92 | 0.08 | 0.00 | 0.00 |
| medium | 48 | 0.546 | 0.56 | 0.72 | 0.28 | 0.00 | 0.00 |
| intermediate | 54 | 0.478 | 0.56 | 0.58 | 0.20 | 0.22 | 0.00 |
| hard | 27 | 0.387 | 0.33 | 0.39 | 0.21 | 0.41 | 0.00 |
| 1 table | 76 | 0.667 | 0.93 | 0.93 | 0.07 | 0.00 | 0.00 |
| 6+ tables | 7 | 0.130 | 0.00 | 0.11 | 0.07 | 0.82 | 0.00 |

Reading: the document gate never loses a table — every gold report is gated, on
all 192 records. Losses split evenly between ranking and candidate generation,
but they sit in different places. Ranking loses tables on small questions
(medium: 28% rank_miss, zero candidate_miss). Candidate generation collapses on
large ones (six-plus tables: 82% candidate_miss), because a single ranked depth
of 50 is shared across up to 30 gated reports, leaving almost no depth per
report. Raising depth would convert much of that into rank_miss.

## Compare two systems with it

```
python3 scripts/cross_validate_retrieval.py --labels annotations/benchmark.jsonl --folds 5
```

Retrieves once to depth 50, caches, then scores selection policies offline over
five folds blocked by connected report groups, with a paired cluster bootstrap
against a fixed top-5 baseline. Because retrieval is cached and policies are
scored offline, sweeping is cheap enough that cross-validation is the default
rather than a ceremony.

**Decision rule, fixed in advance.** Accept a change only when the
cross-validated mean improves, the bootstrap CI on the delta excludes zero, and
no difficulty tier regresses. Report the interval, never the point estimate
alone.

**Cross-encoder drift comes from batch packing, not from the session.** Two
Kaggle sessions that scored `pairs_bench_v4.jsonl` with the same model, prompt
and settings agree on 11,580 of 11,592 pairs, mean |delta| 1.8e-05, and the
rebuilt rankings score identically: F2 delta 0.0000 on every tier. The earlier
reading — 138 of 11,592 identical, F2 +0.0076 — compared runs whose candidate
pools differed (depth 50 against depth 100), so the length-sorted batches packed
differently and int8 outlier handling saw different neighbours.

The consequence is narrower than "nothing reproduces" but it still binds: any
treatment that changes the candidate text or the number of queries repacks the
batches, so it carries that noise on top of its effect, and 0.008 F2 is the
scale of it. Re-running the identical configuration is the one thing that is
free of it, which is what makes a same-session control worth its GPU hours only
when the two cells differ in the factor under test.

So report the drift beside the effect:

```
python3 scripts/compare_rerank_runs.py --pairs output/rerank/pairs_bench_v4.jsonl \
    --control <baseline scores> --treatment <scores under test>
```

It prints the raw agreement, an A/A stratum of the questions whose prompt is
identical under both settings, and the treated stratum. The treated stratum has
to beat the A/A stratum before any of the difference is the method. Passing
`--aa-items 99` makes the whole set an A/A, which is how to measure drift on
purpose.

## What it can and cannot resolve

Resampling gives a CI half-width of roughly 0.023 F2 overall, so effects above
about 0.05 are decidable. Per tier the intervals are much wider — hard has 27
records, and nothing subtle about hard questions is measurable yet.

The queue in `annotations/v3/queue.jsonl` holds 230 further questions, weighted
toward hard and intermediate, each excluded from reports already used so new
records form new clusters. Roughly 10% of drawn questions resist labelling from
source alone and are recorded as deferred with a reason rather than guessed.

## What the live leaderboard can and cannot resolve

The public score is a mean over 1,012 questions and per-question F2 has a
standard deviation of 0.271, so the standard error of that mean is **0.0085** and
two independent configurations are indistinguishable inside about **±0.017**.

Paired changes resolve better but not by much. The item expansion altered 343
questions with a paired delta standard deviation of 0.245, giving a standard
error of 0.0045 on the corpus mean — and it measured +0.0045, exactly one
standard error. So it is not distinguishable from zero, and neither is the
weighted-RRF choice that ships (0.5221 against 0.5167 at w=0.3) nor any cap
result (0.0002 to 0.0035).

Three live results have ever cleared the bar: reranking over sparse (+0.043),
8B over 4B (+0.031), and the v3-to-v4 representation (+0.023). All three were
mechanism-driven rather than swept.

**The rule that follows.** Accept a live change at a margin above 0.02, or not at
all. A sequence of sub-0.02 accepts is a sequence of coin flips, and the private
phase allows five submissions — the cost of having tuned on noise is paid there,
where it cannot be measured.

Re-submitting a package measures nothing: the scorer is deterministic, so the
same zip returns the same score. The 0.0085 is sampling error over questions,
which is exactly the quantity that governs transfer to a different question set.
