# Findings — E016

The apparent top strategy is a two-stage system: retrieve the correct report,
then construct a per-question candidate table and select a normalized value
deterministically. Its public artifact contains no evidence of model-generated
arithmetic for the 1,012 checked records; all answers equal precomputed values
in the supplied CSVs. The artifact also emits a uniform one-line table-ID
offset, explaining its zero public table-retrieval scores despite strong
document retrieval.

For our submission, use candidate tables only as grounded evidence and compile
typed arithmetic over raw values. A table-address adapter is a first-class
component, not a formatting detail. The 506-gold/1012-pred warning means the
public score should be treated as provisional until the organizer confirms
length handling.
