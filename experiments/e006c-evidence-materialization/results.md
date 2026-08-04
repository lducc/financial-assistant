# E006c result — passed

The integration fixture resolved
`VNM_financial_statements_2023_consolidated|237` to exactly one literal source
table, materialized it as a 20-row CSV, and wrote a JSON sidecar containing raw
cells plus numeric parse status/value annotations. Raw source cells are retained
unchanged in both artifacts.

The component is intentionally on-demand: it avoids eagerly producing 146,246
evidence CSVs and creates only the CSVs referenced by a candidate submission.
The final submission must still use line IDs regenerated from the exact
organizer OCR artifact.

