# E012 — Deterministic double-annotation subset

## Question

Can we select a reproducible 30-question subset from the frozen E003 queue
that preserves operation/complexity diversity for independent annotation and
reconciliation?

## Method

1. Use only the frozen queue's precomputed `primary_stratum`; never inspect an
   answer, evidence, or system output.
2. Round-robin deterministically across sorted primary strata, with IDs sorting
   within each stratum, until selecting 30 distinct records.
3. Give the identical subset to two annotators using the E010 contract.
4. Report subset strata and operation coverage.  Agreement is not measured
   until both reviewers return completed files.

## Acceptance criteria

- Exactly 30 distinct queue IDs are selected.
- Every operation hint represented in the 120-record queue is represented in
  the selected subset.
- Re-running on unchanged input produces identical IDs and summary.
