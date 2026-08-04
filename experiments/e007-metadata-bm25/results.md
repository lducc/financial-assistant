# E007 result — smoke test passed; relevance remains unmeasured

The integration fixture retrieved the known VNM 2023 net-revenue statement
table at source line 237. A 30-question real-corpus smoke run completed without
crash: all queries used the strict ticker/year/scope filter, had nonzero BM25
top scores, and averaged 7.4 candidate reports / 522.3 candidate tables.
Median per-query ranking latency was 185 ms after report discovery.

This establishes an auditable deterministic baseline. It does **not** establish
Recall, F2, evidence sufficiency, or answer accuracy because E003 annotations
are not yet complete. A full 1,012-question rerun should use caching/indexing
rather than reparse tables per query.

