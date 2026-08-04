# Research-state addendum — E005/E005b

## Completed

- E005 baseline: year coverage 1,011/1,012; entity coverage 605/1,012.
- E005b exact bare-ticker repair: entity coverage 983/1,012, with 29 unresolved.

## Current decision

Use explicit ticker, year, and `công ty mẹ` scope as deterministic candidate
metadata. Keep all unresolved/ambiguous entities as soft retrieval cases until
the manual proxy benchmark provides accuracy evidence.

## Next safe work

1. E004: execute the submission validator against safe synthetic fixtures.
2. E003: begin manual annotation of the frozen queue; double-annotate 30.
3. Build normalized table rows and the metadata-filtered lexical retrieval
   baseline, using the exact organizer OCR artifact when available.

