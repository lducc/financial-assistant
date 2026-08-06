# E015 — 14B-grounded multi-candidate program agent

## Research question

Under the maximum-14B constraint, does generating multiple *typed and
evidence-grounded* Pandas-program candidates, then selecting only executable
and unit-consistent candidates, offer a viable replacement for a single free
form answer prompt?

## Boundaries

- Select a base model only after a compliance manifest confirms <=14B, public
  availability by the organizer cutoff, licence, and local inference policy.
- Do not use closed APIs or a large-teacher trace unless explicitly permitted.
- The LLM receives a compact Markdown presentation of retrieved tables; Pandas
  executes only against the matching raw evidence CSV(s).
- A candidate cannot emit a literal answer, invent an unlisted table/variable,
  or change requested unit silently.

## Candidate flow

1. Metadata-gated hybrid retrieval yields a bounded evidence closure.
2. A <=14B proposer receives question, typed E009 plan, and Markdown evidence.
3. Generate N diverse typed program candidates.
4. Static validator checks variables, referenced rows/columns, operator family,
   and output unit; sandboxed Pandas executes survivors.
5. Canonicalize equivalent programs and majority-vote only among valid outputs;
   otherwise abstain/flag for review.

## Evaluation plan

Use reconciled E003 labels only.  Report evidence closure, executable-program
rate, program agreement, answer accuracy, unit failures, and abstentions by
operation/complexity stratum.  Compare single-candidate against N-candidate
selection using the same retrieved evidence.

## Falsification

Reject the approach if multi-candidate selection merely increases executable
but ungrounded programs, or if it does not improve answer accuracy on the
frozen proxy set after reconciliation.
