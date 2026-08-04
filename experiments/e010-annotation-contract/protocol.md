# E010 — Proxy annotation contract and validator

## Question

Can manual labels for the frozen E003 queue be made machine-checkable before
they are used to score retrieval, grounding, or execution?

## Method

1. Define an explicit annotation contract for evidence reports/tables,
   row-column bindings, operation graph, Pandas query, numeric answer, unit,
   confidence, and failure tags.
2. Add a validator that permits the untouched queue in `unannotated` state but
   rejects a claimed `complete` record unless all required fields are present
   and internally consistent.
3. Create tests for an incomplete queue record, a valid completed record, and
   invalid evidence/table identifiers.

## Acceptance criteria

- The original 120-record queue validates only with incomplete labels allowed.
- A synthetic complete record validates.
- Missing evidence, missing query, non-finite answer, and a table whose report
  prefix is absent from `gold_reports` are rejected.

## Scope boundary

This is annotation infrastructure.  It must neither infer labels nor turn
organizer feedback into pseudo-gold.
