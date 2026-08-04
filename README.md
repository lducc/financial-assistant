# ViFinQA evidence-closure research

This repository is a reproducible research and engineering workspace for
Road2AI Stage 2 / ViFinQA. The central question is whether accounting-aware
evidence closure and verified typed Pandas compilation can improve Vietnamese
financial table QA when organizer labels are unavailable.

The public dataset snapshot is intentionally local-only at `data/raw/vifinqa`.
Its pinned source revision is recorded in `research-state.yaml`.

## First experiment sequence

1. E001 — inventory and integrity audit of the pinned dataset.
2. E002 — audit OCR/table markup and establish stable table identifiers.
3. E004 — validate competition submissions before any submission is made.
4. E005–E009 — build the deterministic retrieval and execution baseline.

Run `python scripts/audit_dataset.py --dataset-root data/raw/vifinqa` after
the audit tooling is added. All generated results belong under `data/results/`
and are deliberately excluded from version control.

