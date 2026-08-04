# E010 results — annotation contract and validator

The focused validator tests passed.  The untouched E003 queue also validated
with `--allow-incomplete`: 120 records, all marked `unannotated`, zero schema
errors.  This is deliberately not a scoring pass.

The contract blocks a completed record with missing evidence, a malformed or
cross-report table ID, a non-finite numeric answer, or a program that does not
assign `result`.  The next human-in-the-loop action is to fill and independently
review the designated E003 records; only then can E005b/E008b/E009 be measured.
