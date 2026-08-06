# E016 — Public top-submission forensics

## Research question

What concrete retrieval, table-addressing, and answer-generation strategy is
used by the current public leaderboard leader, and which parts are safe to
adopt under the ViFinQA 14B limit?

## Scope and safety

- Inspect only the anonymously downloadable public artifact for submission
  2333 (`synera`) and local copies already obtained from the public leaderboard.
- Do not expose signed download URLs, private test data, or hidden gold labels.
- Treat artifact fields as implementation evidence, not proof that the method
  is valid for the private phase.

## Measurements

1. Count submitted records and per-question CSV rows.
2. Check whether `pandas_query` selects a candidate by `candidate_id`.
3. Check whether the selected candidate has a precomputed `answer_value` and
   whether the submitted answer equals it.
4. Compare the submitted `relevant_tables` suffix with the candidate CSV's
   `table_id` to detect an ordinal-versus-OCR-line convention mismatch.
5. Record scorer warnings, especially prediction/gold length mismatch.

## Acceptance

The audit must be deterministic, skip malformed rows without crashing, and
report counts plus concrete examples. Conclusions must separate observed
artifact behavior from hypotheses about the hidden scorer.
