# Research log addendum — 2026-08-04 E009

1. Committed the E009 protocol before running the experiment.
2. Implemented a priority-ordered Vietnamese rule planner with explicit
   operand roles and constraints inherited from E005b.
3. Focused tests passed.  The first full run exposed a summary-only type error;
   a v2 runner repaired the count and reran the unchanged 1,012-question input.
4. Full run verified 1,012 valid plans.  No semantic accuracy claim was made.
