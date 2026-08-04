# Finding addendum — visible mojibake is not sufficient evidence

A dashboard, terminal, or chat surface may render Vietnamese differently from
the UTF-8 file.  Repairing on that appearance alone risks corrupting genuine
text, including normal Vietnamese sequences containing `Ã`.  The ingestion
architecture should retain raw strings and normalize only for narrowly tested
matching operations, never for evidence addressing or submission table IDs.
