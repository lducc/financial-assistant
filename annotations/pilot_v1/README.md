# Independent pilot v1

`queue.jsonl` contains 30 question-only annotation tasks. It is retained for
review workflow context; it is not a label set or organizer ground truth.

`agent_labels.jsonl` contains twelve source-cell audits independent of the
removed candidate-seeded benchmark. They pass raw-cell validation. They are
single-agent reviews, not double-reviewed labels and must not select models.

User reviewed labels 183, 301, 222, 126, 724, and 771 on 2026-08-09. Labels
832, 823, 125, 810, 887, and 1011 still need user or independent review before promotion.

Run `python3 scripts/validate_pilot.py` to verify source bindings against local
raw OCR. Add independent reviewer files before treating this pilot as an
evaluation set.
