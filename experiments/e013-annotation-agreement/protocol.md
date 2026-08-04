# E013 — Double-annotation agreement report

## Question

Can two completed E012 reviewer files be compared reproducibly at the levels
that matter for ViFinQA: evidence, operation, unit, and final number?

## Method

1. Validate each reviewer record against E010 before comparison.
2. Join reviews by question ID and flag missing, duplicate, invalid, and
   excluded records.
3. Report exact agreement for report/table sets, operation graph, and unit.
4. Compare numeric answers using the organizer-reported relative tolerance of
   0.02% (`abs(a-b)/max(abs(b), epsilon) <= 0.0002`), retaining exact numeric
   differences for reconciliation.
5. Emit disagreement rows; do not automatically choose either reviewer.

## Acceptance criteria

- Synthetic identical annotations agree on every measure.
- A numeric difference just within tolerance agrees and one beyond does not.
- Table, operation, unit, and missing-ID disagreements are exposed.

## Scope boundary

This is an agreement/reconciliation aid.  It does not validate a human label
against the hidden organizer gold or resolve disagreements automatically.
