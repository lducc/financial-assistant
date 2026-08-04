# E014 — Independent heuristic disagreement audit

## Question

Where do the E003 queue sampler's original operation hints and the E009
planner's later operation templates disagree on the same frozen questions?

## Method

1. Map only equivalent label names (`extremum_or_selector` → `selector`).
2. Join the frozen E003 queue to E009 plans by question ID.
3. Report the confusion matrix and list disagreement IDs, retaining the two
   independent cues rather than choosing a winner.
4. Mark disagreements as higher-priority reconciliation candidates for E012.

## Acceptance criteria

- Every E003 record joins to exactly one E009 plan.
- The audit is deterministic and reports both agreement and disagreement.
- No agreement percentage is called accuracy without human labels.
