# E006c — On-demand evidence materialization

**Type:** confirmatory engineering experiment  
**Hypothesis:** A source-line-addressed table can be deterministically
re-extracted from a report, preserved as a submission CSV, and accompanied by
non-destructive numeric annotations without pre-materializing the entire corpus.

## Method

Use `report_id|start_line` to locate the literal HTML table in a supplied OCR
artifact. Write ragged rows to a rectangular CSV, preserve raw cells, and emit a
JSON sidecar containing parse status/Decimal values. Test a known VNM statement
table end-to-end.

## Decision rule

- The selected source line must resolve to exactly one literal table.
- CSV rows preserve original cells; no source cell is overwritten.
- Numeric conversion failures are recorded in sidecar metadata, not coerced.

