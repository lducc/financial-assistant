# E000 reconciliation — semantic rule is clear; artifact alignment is not yet proven

The organizer has explicitly defined `relevant_tables` as
`report_id|line_number_where_the_table_starts_in_the_corresponding_OCR_file`.
That rule is adopted for the submission catalog.

However, their illustrative ID
`VNM_financial_statements_2023_consolidated|350` does **not** map to a literal
`<table>` start in the pinned Hugging Face snapshot
`0450088ab22ec946f04f097586967ca405955b3b`: source line 350 is ordinary OCR
text, and the visible statement tables start at lines such as 166, 189, 208,
and 237.

Therefore, do not submit line IDs from this mirror until its byte-for-byte
relationship to the organizer's evaluation package is confirmed. The catalog
implementation is correct for a supplied OCR artifact; final IDs must be
regenerated from the exact artifact used by the dashboard.

