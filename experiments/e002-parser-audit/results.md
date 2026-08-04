# E002 results — passed with an OCR-quality warning

The audit extracted 146,246 document-wide literal HTML table identities. Every
table had at least one parsed row and one cell, and every table had preceding
page context. Eight PRT explanatory reports had no literal HTML table, which is
compatible with their document type and must be retained as catalog records.

Possible mojibake markers occurred in 1,794 reports and in no questions. The
next parser must preserve raw cells, record a decoding flag, and evaluate any
repair variant against manually annotated evidence; global silent rewriting is
not justified.

`report_id|document_wide_html_ordinal` is stable as an internal catalog key,
but remains unvalidated as the competition's `relevant_tables` position.

