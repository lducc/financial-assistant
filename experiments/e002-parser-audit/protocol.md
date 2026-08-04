# E002 — OCR/table-markup extraction audit

**Type:** confirmatory infrastructure experiment  
**Hypothesis:** Literal `<table>...</table>` blocks can provide a stable
first-pass table identity, but table content quality and encoding anomalies will
make raw table ordinals insufficient as semantic evidence.

## Prediction

Most reports contain at least one HTML table; a nontrivial fraction includes
administrative or malformed tables. The audit will expose the table-count
distribution, malformed blocks, row/column irregularity, and text-decoding
signals needed for a cataloging baseline.

## Method

1. Extract literal table blocks with an HTML parser, retaining report ID,
   document-wide ordinal, page context, and raw HTML.
2. Parse rows/cells while preserving raw values and table ordinals.
3. Measure table presence, row/column distributions, malformed markup, and
   candidate statement-table indicators.
4. Produce a deterministic catalog with no semantic correction.

## Metrics / decision rule

- Extraction coverage: share of reports with stable table identities.
- Structural validity: share of tables with at least one row and one cell.
- Record—not silently fix—encoding and markup defects.
- The table-position submission convention remains unresolved until organizer
  documentation or a controlled submission clarifies it.

