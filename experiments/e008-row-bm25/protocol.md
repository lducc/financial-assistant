# E008 — Row-centric BM25 baseline

**Type:** confirmatory deterministic retrieval baseline  
**Hypothesis:** Ranking row text before aggregating to tables provides more
specific metric grounding than whole-table ranking and yields an auditable row
binding for later typed compilation.

## Method

Use the E005b metadata filter, apply BM25 to individual raw rows, retain each
table's best-scoring row, and emit table ID, row index, raw row cells, score,
candidate counts, and latency. Verify a known VNM net-revenue row.

## Decision rule

- Every query emits ranked row/table traces.
- The integration fixture locates the expected VNM revenue table and row.
- No quality claim is made until comparison against E003 gold bindings.

