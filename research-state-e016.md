# Research-state addendum — E016 (2026-08-06)

## Completed

- E016 public top-submission forensics passed.
- The current top public solution (submission 2333, `synera`) uses document
  retrieval followed by evidence CSV candidate selection. Every checked answer
  is read from a precomputed evidence column, and every submitted table suffix
  is one less than the candidate row's table ID.

## Decision

Adopt the *safe* part of this pattern for the ≤14B design: generate a bounded
candidate evidence table, but require the model to emit a typed program whose
arithmetic is executed over raw columns. Never rely on a hidden precomputed
answer column as the sole correctness mechanism. Normalize table IDs through an
explicit OCR-line/HTML-ordinal conversion layer and test it before submission.

## Next

1. Implement the table-address conversion test fixture (HTML ordinal, OCR line,
   and one-based/zero-based variants).
2. Add a candidate-evidence schema with provenance and unit checks to E015.
3. Evaluate the 14B multi-candidate program agent on the reconciled proxy set.
