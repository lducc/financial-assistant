# ViFinQA Stage 2: product documentation

## 1. Product summary

The system answers Vietnamese financial questions by retrieving source tables
from the organizer corpus, selecting evidence tables, and executing a grounded
numerical program. Every evidence CSV is copied from an organizer table and keeps
its document and table provenance. The pipeline diagram is in
[README.md](README.md).

## 2. Data

The only data are the organizer's questions, company mapping, and financial
statement reports, kept at `data/raw/vifinqa/` (layout in
[README.md](README.md#reproduce)). The organizer data access link (Google Drive,
OneDrive, or equivalent) goes in the delivery email and stays out of the repository.

## 3. Models and checkpoints

| Model | Revision | Use |
| --- | --- | --- |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` | dense table index |
| `Qwen/Qwen3-Reranker-8B` | `77d193c791ed757ca307ee72715aa132723da912` | fact/table reranking |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | grounded answer programs |

All are public Hugging Face checkpoints, used without fine-tuning. Put their links
in the delivery email.

## 4. Reproduction

Every command, from `pip install -e .` through the answer notebook, is in
[README.md](README.md#reproduce). The answer notebook writes the ZIP only after
all 1,012 records validate. Committed packages are in `artifacts/`.

## 5. Submission compliance

Evidence CSVs are subsets of organizer data, traceable through `relevant_docs`,
`relevant_tables`, and `evidence`. Queries compute results from those CSVs at
execution time, so every value comes from a CSV. Check a package with:

```bash
python scripts/validate_submission.py path/to/package
python scripts/verify_execution.py --package path/to/package
```

Deliver this document, the source repository, the organizer-data link, the
checkpoint links, and the final ZIP's SHA-256.

## 6. Known limitations

Malformed programs are retried with validation feedback, then rejected. A run is
complete only when every question has a validated, source-grounded execution record.
