# Internal retrieval labels

`gold_150.jsonl` is reconstructed from preserved E027 source evidence and the
official question text. It restores 150 exact document/table/cell bindings for
retrieval evaluation. Original answer programs, operation graphs, and taxonomy
were deleted. Do not treat this file as answer ground truth.

`gold_150_candidate_submissions.jsonl` separately preserves matching candidate
answers, tables, evidence, and `pandas_query` strings. These are predictions,
not executable gold answers.

`pilot_v1` contains independent source-cell audits for retrieval diagnostics,
not organizer ground truth or answer labels. See its README and
`docs/research-history.md` for trust boundaries.
