# E016 results — public leader artifact (submission 2333)

Run date: 2026-08-06. Artifact: public `synera` submission from the public
phase leaderboard. The audit is read-only and uses only the downloaded public
submission and its evidence CSVs.

## Verified measurements

- Submitted records: **1,012**.
- Evidence CSVs found: **1,012 / 1,012**.
- Query families: **359** `candidate_id` selectors, **651** `source_id`
  selectors, and **2** `candidate_id.isin(...).sum()` selectors.
- All **1,012 / 1,012** answers equal a value already present in the evidence
  CSV (`answer_value` or `computed_answer`), including the two sums whose
  selected `answer_value` cells sum to the submitted answer.
- For all **1,012 / 1,012** records, the submitted table suffix is exactly
  **one less** than the evidence row's `table_id` (`submitted - csv = -1`).
- Public scorer warning: `questions.jsonl` length mismatch, `gold=506`
  and `pred=1012`.

## Interpretation

The leader is best understood as a high-recall document retriever plus a
candidate-evidence generator and deterministic value selector. It is not
evidence that a ≤14B model must perform free-form arithmetic at answer time.
The table metric collapse is consistent with a systematic ordinal/line-number
conversion bug, while answer values remain correct because the candidate CSV
already contains normalized/computed values.

This is an observed artifact property, not a claim about hidden gold labels or
private-phase validity. The public scorer's 506-vs-1012 warning must be treated
as a separate validity risk.

## Reproduction

```text
python scripts/audit_top_submission_values.py \
  data/results/top_synera_2333/submission_unpacked
```

The detailed machine-readable output is retained at
`data/results/top_synera_2333/audit_v3.json` and the value audit was rerun
after the protocol commit.
