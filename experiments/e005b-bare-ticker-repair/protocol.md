# E005b — Exact bare-ticker repair

**Type:** confirmatory targeted repair of E005  
**Hypothesis:** Most E005 unresolved entities are explicit bare ticker symbols
that were omitted by the initial parenthesis-only rule. Matching only uppercase
tokens in the supplied 100-ticker map will recover these safely.

## Prediction

Entity coverage will rise substantially without fuzzy matching. Any additional
candidate must be an exact uppercase token in `code_stock.csv`; ordinary
Vietnamese abbreviations and unlisted acronyms remain unresolved.

## Method and decision rule

1. Keep all E005 extraction unchanged.
2. Add exact bare uppercase tokens only if they are a known ticker.
3. Mark the source as `explicit_bare_ticker` in the audit trace.
4. Compare coverage to E005; do not call it accuracy before proxy labels exist.

