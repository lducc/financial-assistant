# E003 annotation guide

For each frozen queue record, retain the question unchanged and replace only
its `annotation` object.

- `gold_reports`: exact report IDs, one per necessary report.
- `gold_tables`: dashboard-format `report_id|table_start_line`; every table's
  report prefix must occur in `gold_reports`.
- `row_column_bindings`: enough row/column detail for a second reviewer to
  reproduce the extraction.
- `table_units`: units as printed in the evidence.
- `operation_graph`: a small JSON description of the calculation (for example
  `{"op":"difference","left":"2023","right":"2022"}`).
- `pandas_query`: must calculate from the evidence variable(s) and assign
  `result`; no constant answer expressions.
- `numeric_answer`: numeric, in exactly the unit requested by the question.
- `confidence`: `high`, `medium`, or `low`; use `needs_review` rather than
  guessing when evidence is uncertain.

Run the validator with `--allow-incomplete` while the queue is being filled.
Before scoring a system, require every retained record to validate without that
flag.  A second annotator should independently complete the preselected
double-annotation subset before reconciliation.
