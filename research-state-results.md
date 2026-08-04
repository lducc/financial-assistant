# Research-state addendum — 2026-08-04

This addendum records results after the initial protocol state was committed.
It should be read with `research-state.yaml` until the next state refresh.

## Completed

- E001: dataset inventory and integrity audit — passed.
- E002: literal HTML table-markup audit — passed with OCR/metadata warnings.

## Next

1. E003: create a stratified manual proxy annotation set.
2. E004: exercise the submission validator with generated fixtures and validate
   the organizer's table-position convention when a safe test submission is
   available.
3. E005: build a conservative metadata parser that keeps unknown scope values.

## Results to preserve

- 1,012 questions; IDs unique and contiguous.
- 1,973 reports; 146,246 literal HTML tables.
- Scope distribution: 957 consolidated, 954 separate, 7 aggregated, 55 unknown.
- 1,794 reports carry possible mojibake markers; questions do not.
- Eight PRT explanatory reports have no literal HTML tables.

