# E008b — Metadata-aware BM25 query cleaning

**Type:** confirmatory targeted repair of E008  
**Hypothesis:** Removing already-filtered ticker/year tokens and generic
question/unit terms from the BM25 query will prevent ubiquitous table headers
from outranking metric-bearing rows.

## Decision rule

The VNM net-revenue fixture must return a row containing `Doanh thu thuần` from
the main income-statement table within top-k. The same 30-question smoke run
must remain crash-free; this remains a grounding diagnostic, not accuracy.

