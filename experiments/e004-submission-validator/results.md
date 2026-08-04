# E004 result — passed

The static validator passed a valid package fixture and rejected fixtures for:

- constant Pandas output;
- missing evidence CSV;
- table reference absent from an optional line-addressed catalog;
- ZIP with an extra parent directory.

It also checks unique/full question IDs, finite numeric answers, table-ID syntax,
table/document consistency, evidence variable uniqueness, `data/`-relative CSV
paths, and evidence-variable use in the Pandas query.

This is only a local structural gate. It cannot prove dashboard compatibility or
that the Hugging Face OCR mirror is the organizer's exact line-address source.

