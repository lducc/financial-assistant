# Audit report - 023 compliant docs balanced

## Objective

Improve both document precision and recall over submission 022 while retaining a reproducible pipeline that complies with the competition model/data rules.

Official score used as the starting point for 022:

- DOCS_PRECISION: 0.9623
- DOCS_RECALL: 0.8469
- DOCS_MRR5: 0.9713

No official leaderboard score is available yet for 023.

## What was integrated

Only general rules and aliases from the supplied pseudo-GT 21 analysis were integrated. No per-question predictions, hidden answers, table IDs, or closed-model labels were copied.

The final pipeline uses full ticker-year Cartesian coverage, union entity extraction, default consolidated scope, per-year mixed scope, unspecified-scope fallback, and one report per atom. Every manual alias is checked against text in an official BTC report owned by the target ticker.

## Local result

- Questions: 1,012
- Selected documents: 3,000
- Mean documents per question: 2.9644
- Empty document lists: 0
- Entity sets changed versus 022: 30 questions
- Document lists changed versus 022: 317 questions
- Catalog reports: 1,973
- Local-model calls: 0

The supplied pseudo-GT analysis reported 2,744 documents. The higher count here is mainly intentional full-year coverage and broad handling of two exceptional questions, not duplicate reports.

## Manual double-checks

The changed entity set was reviewed twice. Important regressions now covered by tests include:

- Questions 4/8/499: `Chung khoan FPT` resolves to FTS, not FPT.
- Question 594: VCB is an investee; GEE remains the subject.
- Questions 740/749: metric wording no longer hides GEG or EIB.
- Question 949: ACB, MBB, EIB, and BID are all retained.
- Question 952: DTK 2022-2023 use separate reports and 2024-2025 use consolidated reports.

## Validation

- 9 tests passed.
- Submission contains exactly 1,012 rows in official question order.
- Every document ID exists in the official catalog.
- No ID includes the source-file suffix `_extracted`.
- No duplicate document appears within a question.
- No ticker/year atom mismatch was found.
- 47 atoms use an official report without an explicit scope suffix.
- 5 atoms use the opposite scope because exact and unspecified reports do not exist.
- ZIP root contains exactly `submission.json` and `data/`.

## Known risks

Question 412 has three entities but no explicit year. The recall-oriented build selects every available year for those entities, producing 33 documents. This may lower precision for one macro-averaged query.

Question 464 names no companies and asks for a corpus-wide screen in 2015-2016. The build selects available reports for the complete official company registry, producing 139 documents. This is defensible for recall but may not match the organizer annotation policy.

Full-year coverage should increase recall when BTC annotates every year named in a question. It can reduce precision where a later annual report already contains the prior-year comparative column. The leaderboard result is required to measure that tradeoff.

This is a docs-only probe. `ANSWER_ACCURACY` and `EXECUTION_ACCURACY` are expected to be zero and must not be used to judge this experiment.