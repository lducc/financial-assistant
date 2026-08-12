# V2 evidence annotation protocol

Scope: question-only queue plus raw ViFinQA reports. Reviewers must not read Gold labels, retrieval traces, predictions, submissions, or another review file.

For every required entity, report year, scope, metric, and operand role, emit one slot. Each alternative is one complete interchangeable evidence set: report IDs, table IDs as `report_id|start_line`, and exact raw-cell coordinates. Do not merge partial alternatives. Preserve OCR cell text; write `unknown` for unverified period, scope, or unit.

Canonical fields for calibration: use `operation: "lookup"` unless question explicitly asks to calculate, compare, change, or combine two named values. A requested value expressed in a unit (`triệu đồng`, `%`, `nghìn tỷ`) is still a lookup. For lookup slots use `operand_role: "value"` exactly. Set `metric` to the complete requested Vietnamese noun phrase, retaining discriminative qualifiers such as `tổng`, person, counterparty, or investment name; exclude only issuer, report scope, and period. Do not replace it with a shorter row-label guess.

Calibration restart canonical slot key: use the selected source report's uppercase ticker for `entity`, its year and scope for `report_year` and `scope`, and the exact selected source-table row-label cell (trim surrounding whitespace; collapse internal whitespace) for `metric`. Set `slot_id` exactly to `{entity}|{report_year}|{scope}|{metric}|{operand_role}`. Do not paraphrase question text into these fields. If one lookup needs more than one raw value, create one slot per row-label/operand value. This rule overrides the preceding metric wording for calibration restart.

Each JSONL record must use schema `2.0`, include its queue `id` and `question_hash`, `operation`, `complete: true`, numeric confidence, and a reviewer-specific protocol hash. Use `agent-audited proxy` only; never call these organizer or human ground truth.

Calibration set: first 20 development queue records. Stop after those records. The adjudicator resolves every disagreement after both reviews are complete. Gates: 100% coordinate validity, median table-set Jaccard at least 0.80, slot exact agreement at least 0.75, operation agreement at least 0.80. If any gate fails, discard this calibration set, revise protocol once, and restart it.
