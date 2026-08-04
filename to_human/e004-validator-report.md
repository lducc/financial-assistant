# ViFinQA submission preflight is ready

Run the local preflight before every candidate ZIP:

```text
python scripts/validate_submission_v2.py submission.zip \
  --questions data/raw/vifinqa/questions/questions.jsonl \
  --catalog data/derived/table_catalog/tables.jsonl
```

Do not use the mirror-derived catalog for final dashboard submission until the
exact organizer OCR artifact has been reconciled. The validator remains useful
for ZIP, ID, evidence, and executable-query integrity regardless.

