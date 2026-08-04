# E005 — Conservative question metadata parser

**Type:** confirmatory deterministic-baseline experiment  
**Hypothesis:** Exact ticker recognition plus alias-normalized company matching
and explicit year/scope extraction can sharply reduce the report search space
without introducing unsafe hard exclusions.

## Prediction

Most questions will expose at least one reporting year and a resolvable ticker
or company name. `công ty mẹ` will be a high-precision separate-scope signal,
while questions with no explicit scope must retain both report scopes.

## Method

1. Load the provided ticker/company map and normalize Vietnamese text only for
   matching; preserve the original question text.
2. Extract exact tickers in parentheses, then exact normalized company aliases.
3. Extract all four-digit reporting years and scope cues.
4. Emit candidates and a confidence/audit trace for every question.
5. Measure coverage and ambiguity; do not claim accuracy without proxy labels.

## Decision rule

- A hard filter is allowed only for an explicit ticker, explicit year, or a
  high-precision explicit scope cue.
- Unresolved entity/scope becomes a soft ranking feature, never a dropped
  question or report.
- Record all ambiguities for proxy annotation.

