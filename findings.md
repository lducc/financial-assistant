# Findings

## Current understanding

ViFinQA is an end-to-end retrieval and executable-Pandas task, not a
single-table QA benchmark. The public package supplies questions and OCR
reports but no gold evidence, answers, or programs. This makes an auditable
proxy benchmark and parser audit prerequisites for credible model comparisons.

## Supported findings

None yet. The initial dataset counts are an inventory observation, not an
evaluated system result.

## Negative results

None yet.

## Lessons and constraints

- Preserve the dataset revision and do not treat leaderboard feedback as labels.
- Do not claim novelty for generic hybrid retrieval or generic agentic RAG.
- Track raw source text separately from any decoding/normalization repair.
- Protocols must precede results in git history.

## Open questions

- Whether report-table position means HTML-table ordinal, page-local ordinal,
  or another organizer-defined identifier.
- The prevalence and pattern of encoding/OCR corruption in reports and questions.
- The distribution of table counts and statement types required by questions.

