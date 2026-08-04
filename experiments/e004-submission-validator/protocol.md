# E004 — End-to-end submission validator

**Type:** confirmatory engineering experiment  
**Hypothesis:** A local validator can reject the known structural failure modes
before dashboard submission: malformed ZIP layout, missing evidence, missing
question IDs, invalid line-addressed table references, non-finite answers, and
Pandas programs that do not reference evidence.

## Method

Create synthetic valid and invalid submission packages. Validate both a
directory and ZIP package, optionally against the question IDs and a
line-addressed catalog. No live dashboard submission is used.

## Decision rule

- Accept every valid fixture.
- Reject each deliberately malformed fixture with a specific error class.
- Treat a passing local check as packaging validation only, never score
validation or proof that a local catalog matches the organizer artifact.

