# Table Retrieval: Next Approaches

Current submission mode: `report-coverage`. It keeps baseline ranking for a
single report, while reserving one table for each document-gated report on
multi-report questions. `role-coverage` and `evidence-slots` remain
diagnostic-only.

1. **Slot obligations** — expand year-only slots into entity × report-year × scope × metric × operand-role. Reserve one unique table per obligation, then fill remaining positions by baseline order.
2. **Report-gate repair** — improve entity, year, and scope resolution before table ranking; document selection is already strong, but table report coverage still misses.
3. **Row/cell-aware ranking** — rank matched rows with header, period, and unit context, then select their owning tables. Do not concatenate repeated table context.
4. **Dense candidate union** — only if development labels show a material absent-from-sparse-top-50 rate. Use multilingual E5 to union dense and sparse top-50 candidates with provenance.
5. **Hybrid selection** — use baseline relevance for single-table questions and slot coverage for multi-year or multi-entity questions.
6. **Grounded answers** — after table retrieval stabilizes, bind source cells and execute validated arithmetic. Do not guess answers from unlabelled rows.

## Decision gates

- Keep an approach only if it improves v2 development slot recall@5 and passes MRR/nDCG, latency, source-binding, and package checks.
- Freeze code, model revision, labels, corpus/index hash, and decision rule before sealed evaluation.
- Gold-150 is legacy audit only. Gold-250/350 candidate labels never select methods.
