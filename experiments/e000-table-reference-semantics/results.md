# E000 result — organizer table-reference semantics resolved

Organizer clarification on 2026-08-04: the table position in
`relevant_tables` is the **1-based line number at which the table begins in the
corresponding OCR report**, for example
`VNM_financial_statements_2023_consolidated|350`.

This invalidates the prior internal `report_id|document_wide_html_ordinal`
submission convention. The ordinal remains an internal analysis key only. The
new catalog emits `report_id|start_line` while retaining ordinal and page for
diagnostics.

Related organizer constraints recorded from the same clarification:

- submit all 1,012 records even though public scoring currently uses 506;
- numeric answer tolerance is relative and at most 0.02%;
- return percentages as percentage points (for example, `90`, not `0.9`);
- private ranking uses retrieval, answer accuracy, and execution metrics;
- empty `relevant_tables` is valid but loses private retrieval credit;
- Pandas must derive the answer from submitted evidence, not return a constant.

