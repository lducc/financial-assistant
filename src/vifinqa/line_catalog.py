"""Organizer-compatible table identifiers for ViFinQA OCR reports.

The organizer has clarified that `relevant_tables` uses the 1-based source line
where the literal HTML table begins, not the document-wide HTML ordinal. Both
identifiers are retained: the ordinal remains useful for internal analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .catalog import PAGE_RE, TABLE_RE, ReportIdentity, parse_table_rows


@dataclass(frozen=True)
class SubmissionTableRecord:
    report_id: str
    ticker: str
    year: int
    scope: str
    source_path: str
    table_ordinal: int
    start_line: int
    submission_table_id: str
    page: int | None
    row_count: int
    column_count: int

    def as_dict(self) -> dict:
        return asdict(self)


def extract_submission_table_records(text: str, identity: ReportIdentity) -> list[SubmissionTableRecord]:
    """Extract catalog records in one pass, preserving exact source line starts."""

    records: list[SubmissionTableRecord] = []
    pages = list(PAGE_RE.finditer(text))
    page_index = 0
    current_page: int | None = None
    cursor = 0
    current_line = 1
    for ordinal, match in enumerate(TABLE_RE.finditer(text), start=1):
        current_line += text.count("\n", cursor, match.start())
        cursor = match.start()
        while page_index < len(pages) and pages[page_index].start() < match.start():
            current_page = int(pages[page_index].group(1))
            page_index += 1
        rows = parse_table_rows(match.group(0))
        records.append(
            SubmissionTableRecord(
                report_id=identity.report_id,
                ticker=identity.ticker,
                year=identity.year,
                scope=identity.scope,
                source_path=identity.source_path,
                table_ordinal=ordinal,
                start_line=current_line,
                submission_table_id=f"{identity.report_id}|{current_line}",
                page=current_page,
                row_count=len(rows),
                column_count=max((len(row) for row in rows), default=0),
            )
        )
    return records

