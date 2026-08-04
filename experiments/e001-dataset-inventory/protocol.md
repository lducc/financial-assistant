# E001 — Pinned dataset inventory and integrity audit

**Type:** confirmatory infrastructure experiment  
**Hypothesis:** The published ViFinQA revision can be inventoried reproducibly,
and filename-derived metadata will reveal report coverage and scope anomalies
that must be handled before retrieval.

## Prediction

The snapshot contains 1,012 questions and approximately 1,973 reports from 100
companies over 2015–2025. Filename scopes will include nonstandard forms beyond
`consolidated` and `separate`, so strict two-class filename parsing will lose
reports.

## Method

1. Record the exact remote revision and local file counts.
2. Parse report paths into ticker, year, report identifier, and filename scope.
3. Check question identifiers for uniqueness and contiguous coverage.
4. Count missing ticker-year-scope combinations and malformed paths.
5. Write machine-readable inventory JSON and a human-readable audit report.

## Metrics / decision rule

- Required: all 1,012 question IDs are unique; every report yields a stable
  report identifier.
- Report the scope distribution and all unclassified names.
- Stop downstream retrieval if question IDs are not unique or report identity is
  not deterministic.

