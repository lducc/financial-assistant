# Additional literature — strategy map for ViFinQA

## FinQA: retrieve facts, then generate an executable DSL

The original [FinQA paper](https://arxiv.org/pdf/2109.00122) is the closest
structural precedent.  It separates a fact retriever from a program generator,
turns table rows into natural-language fact sentences, and uses a small DSL of
arithmetic/table operations.  It reports that retrieval input length matters:
the row-fact approach beat a generic sliding-window baseline.  It also
distinguishes execution accuracy from symbolic program accuracy, because a
correct number can come from a wrong program by chance.

**Transfer:** keep our row-level retriever, add a compact ViFinQA DSL/IR, and
score both execution and canonical program validity.  Do not copy FinQA’s
report/table assumptions into dashboard line IDs without calibration.

## DATER / versatile decomposers: simplify evidence and questions

[DATER](https://arxiv.org/pdf/2301.13808) decomposes both a large table into
sub-evidence and a complex question into subquestions.  Its important
engineering idea is “parsing–execution–filling”: parse an operation, execute
the numeric step externally, then fill the result into the next step.  This is
stronger than asking a model for a long free-form chain of thought.

**Transfer:** apply E009’s operation template to each subquestion and preserve
the union of grounded rows.  Never let a model’s subtable pruning delete a row
without retaining the original evidence closure for validation.

## SQL/query decomposition

[Training Table QA via SQL Query Decomposition](https://arxiv.org/pdf/2402.13288)
supports restricted algebraic intermediate steps as supervision.  The lesson
for ViFinQA is not to introduce a full SQL database; it is to train/evaluate
the model on a small operation vocabulary and let a deterministic executor
handle arithmetic and filtering.

## Seek-and-solve

[Seek and Solve](https://arxiv.org/abs/2409.05286) explicitly separates finding
relevant information from solving.  This matches our two-stage architecture:
the first stage should expose evidence and uncertainty; the second should not
silently retrieve new facts while calculating.

## Chain-of-Table

[Chain-of-Table](https://openreview.net/pdf?id=4L0xnS4GQM) evolves an intermediate
table through operations rather than keeping all reasoning in prose.  A
ViFinQA adaptation would use immutable raw CSV plus derived temporary views;
each view must retain a parent table ID and operation provenance so the final
submission still points to the organizer’s original line-addressed table.

## Prioritized strategy list

1. Exact metadata/report gating and line-addressed evidence closure.
2. Row-fact retrieval with a compact Markdown view and raw CSV sidecar.
3. E009 typed operation planner and decomposed subquestions.
4. Restricted program generation/execution with unit and grounding checks.
5. Multi-candidate program sampling and canonical voting.
6. Abstention and human-review routing for unresolved table/operation cases.

The first four are viable without a large teacher.  Self-consistency is a
later ablation; it cannot repair missing evidence.
