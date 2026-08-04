# E009 — Rule-based operation template planner

## Question

Can a conservative, inspectable Vietnamese rule set transform every ViFinQA
question into a computation *shape* that is useful to the later IR compiler,
without claiming that it has identified the correct financial row or table?

## Hypothesis

A priority-ordered lexical planner can classify the high-level operation
(`lookup`, `difference`, `growth_or_change`, `ratio_or_percent`, `average`,
`selector`, or `aggregate`) and return typed operand roles for all 1,012
questions.  This is a coverage and structural-validity experiment, not an
accuracy measurement; operation labels need manual/proxy annotation before
they can be treated as correct.

## Method

1. Reuse E005b metadata for entity, year, and scope constraints.
2. Apply specific operation cues before generic ones.  In particular, phrases
   such as `tổng tài sản` must remain a lookup; only additive cues may yield
   `aggregate`.
3. Emit a JSON plan with the triggering cues, input constraints, expected
   output-unit hint, and operand roles.
4. Unit-test representative synthetic questions, including the `tổng tài sản`
   false-positive guard.
5. Run the planner over the complete public question set and report structural
   coverage plus the operation distribution.

## Acceptance criteria

- Every question yields one schema-valid plan with a known operation.
- The synthetic guard classifies `Tổng tài sản ... là bao nhiêu?` as `lookup`.
- No result is described as operation accuracy without labels.

## Falsification and next decision

If plans are invalid or generic aggregate wording dominates ordinary metric
lookups, revise the cue priority/rules.  If structural coverage passes, use
the plans only as a constrained input to the future compiler and validate
their semantic accuracy on the E003 annotation queue.
