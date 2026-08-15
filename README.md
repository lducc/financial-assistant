# ViFinQA table retrieval

This is a small, deterministic retrieval project for Vietnamese financial-report
tables. It produces document IDs, table IDs, CSV evidence, and conservative
single-report numeric extraction.

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

Install the package in an environment you control. It has to be installed,
not just cloned: `scripts/` and `tests/` import `vifinqa`, and an editable
install is what puts it on the path.

```bash
python3 -m pip install -e .
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
and invalid package schema. The default is contextual BM25 with adaptive
report coverage: for multi-report questions it reserves the best table from
each document-gated report, then fills the remaining slots by relevance.
`baseline`, `role-coverage`, and `evidence-slots` remain explicit modes via
`--table-mode`.

For a single document-gated report, the submission also emits a numeric answer
from the highest-ranked matched OCR row and a matching executable evidence
expression. Multi-report arithmetic remains conservative until its operands
can be jointly validated.

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
