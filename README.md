# ViFinQA table retrieval

This is a small, deterministic retrieval project for Vietnamese financial-report
tables. It produces document IDs, table IDs, and CSV evidence only. Answer
execution is explicitly out of scope.

The production path is:

```text
question -> company/year/scope document gate -> contextual BM25 tables -> top 5 CSV evidence -> strict package validation -> ZIP
```

## Local data and install

Keep the organizer data locally (it is ignored by Git):

```text
data/raw/vifinqa/
  code_stock.csv
  questions/questions.jsonl
  financial_statements/<TICKER>/<YEAR>/<REPORT_ID>/<REPORT_ID>_extracted.txt
```

Install the only test dependency in an environment you control:

```bash
python3 -m pip install -r requirements.txt
```

Run the compact checks:

```bash
python3 -m pytest tests -q
```

## Submission

```bash
python3 run.py --data-root data/raw/vifinqa --output-dir output/run
```

The command keeps raw OCR untouched and writes `output/run/submission.zip`.
The ZIP contains `submission.json` and one `data/tables/*.csv` file for every
retrieved table. The strict validator runs before ZIP creation and rejects bad
IDs, documents/table mismatches, duplicate table IDs, missing evidence CSVs,
and invalid package schema. The fixed top-5 contextual-BM25 baseline is the
default; multi-year role coverage is opt-in with `--table-mode role-coverage`.

Public scores and the retained pilot measure retrieval evidence, not answer
correctness. The pilot is not organizer ground truth; see
[`docs/research-history.md`](docs/research-history.md).

## Layout

```text
run.py                         production command
src/docs.py                    document gate and packaging
src/vifinqa/tables.py          OCR table parsing and CSV evidence
src/vifinqa/retrieval.py       contextual BM25 table retrieval
scripts/validate_submission.py strict package validator
scripts/rebuild_gold_150.py      rebuild 150 retrieval labels from E027 evidence
scripts/evaluate_table_retrieval.py retrieval evaluator
scripts/validate_pilot.py      source-cell binding checker
annotations/pilot_v1/          independent pilot labels
tests/                         compact regression suite
```
