# E009 results — rule-based operation template planner

## Verified result

The focused synthetic suite passed, including the false-positive guard that
maps `Tổng tài sản năm 2023 là bao nhiêu?` to `lookup`, not `aggregate`.
The repaired full runner emitted one structurally valid plan for each of the
1,012 public questions.

| Operation | Plans |
| --- | ---: |
| lookup | 379 |
| ratio_or_percent | 207 |
| selector | 126 |
| growth_or_change | 108 |
| difference | 78 |
| aggregate | 66 |
| average | 48 |

The generated local artifact is `data/derived/template_plans/plans.jsonl`.
Unit hints were: billion VND 373, percent points 294, million VND 216,
unspecified 92, shares/count 22, thousand VND 15.

## Interpretation

This proves schema/coverage and makes the intended calculation visible before
code generation.  It does **not** establish operation classification accuracy,
correct evidence selection, or executable-answer accuracy.  Labels from E003
are still required to measure semantic quality and tune the cue rules.

## Repair note

The original full-run script has a summary-only list-summing defect after its
focused tests pass.  `run_template_planner_v2.py` is the verified runner; both
produce the same plan records and the v2 runner uses boolean structural counts.
