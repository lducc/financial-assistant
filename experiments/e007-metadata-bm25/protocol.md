# E007 — Metadata-filtered BM25 table baseline

**Type:** confirmatory deterministic retrieval baseline  
**Hypothesis:** Applying explicit ticker/year/scope metadata before BM25 ranking
will make table retrieval tractable and materially reduce candidate tables while
retaining an auditable fallback for unresolved metadata.

## Method

1. Use E005b metadata only when explicit; record filter/fallback stage.
2. Parse literal tables from candidate reports and rank their raw row text with
   per-query BM25.
3. Emit top-k table IDs, page, score, preview, candidate counts, and latency.
4. Evaluate only structural metrics until E003 annotations are complete.

## Decision rule

- Every question emits a trace without crashing.
- Candidate set sizes and fallback frequency are reported.
- No Recall/F2 or claim of retrieval quality is made without gold evidence.

