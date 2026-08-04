# E003 — Stratified manual proxy benchmark

**Type:** confirmatory evaluation-infrastructure experiment  
**Hypothesis:** A 100–150 question proxy set stratified by operation, temporal
span, entity count, expected evidence-table count, scope, statement family, and
OCR quality will expose different bottlenecks than aggregate retrieval metrics.

## Prediction

A pure keyword/metadata baseline will be strong for single-report lookups but
will have materially lower evidence sufficiency for multi-period and derived
metric questions. Annotation will also show that high OCR-risk reports need a
separate error stratum.

## Method

1. Build a reproducible question sampler from the 1,012 public prompts using
   only question-level features; do not infer answers from leaderboard results.
2. Manually label question slots, required metric roles, gold reports/tables,
   row/column bindings, units, an operation graph, executable reference query,
   and answer.
3. Double-annotate at least 30 questions and reconcile differences.
4. Freeze the proxy set before comparing retrieval or reasoning variants.

## Metrics / decision rule

- Coverage across every chosen stratum must be reported.
- Two independent annotations must be retained for the double-annotated subset.
- No model is compared until the frozen set contains at least 100 valid labels.

