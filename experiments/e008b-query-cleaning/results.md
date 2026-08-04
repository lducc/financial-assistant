# E008b result — passed

After removing ticker/year tokens already applied as metadata filters and
generic question/unit words from the BM25 query, the VNM fixture retrieves the
`Doanh thu thuần` row from the main income-statement table at source line 237.

The 30-question smoke run completed without crash or zero-score result. It
averaged 7.4 candidate reports and 5,400.7 candidate rows per question, with a
median ranking latency of 179.5 ms after report discovery. Gold row-binding
accuracy remains unmeasured pending E003 annotations.

