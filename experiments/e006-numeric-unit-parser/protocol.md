# E006 — Vietnamese numeric and unit parser

**Type:** confirmatory deterministic-baseline experiment  
**Hypothesis:** A transparent parser for Vietnamese financial numeric forms,
parentheses negatives, missing-value dashes, decimal/thousands separators, and
percentage points can meet a high round-trip reliability threshold before table
normalization.

## Method

Implement unit-tested parsing for source cells and headers. Preserve raw text,
return `None` for missing/ambiguous values, and separate numeric value from
unit-scale metadata. Percentages remain percentage points, matching the
organizer's answer convention.

## Decision rule

- Every curated parser fixture must pass.
- Ambiguous strings must be flagged rather than silently coerced.
- Do not claim corpus-level 99% accuracy until a manually audited sample exists.

