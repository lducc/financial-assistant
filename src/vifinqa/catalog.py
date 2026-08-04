"""Stable, lossless first-pass cataloging for ViFinQA OCR reports.

This module deliberately does not attempt semantic correction. It records literal
HTML-table order, page context, and raw cells so later experiments can compare
normalization strategies without losing the source representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterable


PAGE_RE = re.compile(r"=====\s*PAGE\s+(\d+)\s*=====", re.IGNORECASE)
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
SCOPE_RE = re.compile(r"_(consolidated|separate|aggregated)(?:_\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class ReportIdentity:
    report_id: str
    ticker: str
    year: int
    scope: str
    source_path: str


@dataclass(frozen=True)
class TableRecord:
    table_id: str
    report_id: str
    table_ordinal: int
    page: int | None
    raw_html: str
    rows: list[list[str]]

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    @property
    def valid_structure(self) -> bool:
        return bool(self.rows and self.column_count)

    def as_dict(self, *, include_html: bool = False) -> dict:
        value = asdict(self)
        value["column_count"] = self.column_count
        value["valid_structure"] = self.valid_structure
        if not include_html:
            value.pop("raw_html")
        return value


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell_parts is not None:
            text = " ".join("".join(self._cell_parts).split())
            self._row.append(text)
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def parse_report_identity(report_path: Path, dataset_root: Path) -> ReportIdentity:
    """Derive a stable identity from the published folder layout.

    The `table_ordinal` is a literal document-wide HTML-table ordinal. It is a
    research catalog key, not an assertion about the organizer's submission
    convention.
    """

    relative = report_path.relative_to(dataset_root)
    parts = relative.parts
    try:
        marker = parts.index("financial_statements")
        ticker, year_text, report_dir = parts[marker + 1 : marker + 4]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unexpected report path: {relative}") from exc

    report_id = report_path.stem.removesuffix("_extracted")
    scope_match = SCOPE_RE.search(report_dir)
    scope = scope_match.group(1).lower() if scope_match else "unclassified"
    return ReportIdentity(
        report_id=report_id,
        ticker=ticker,
        year=int(year_text),
        scope=scope,
        source_path=relative.as_posix(),
    )


def parse_table_rows(raw_html: str) -> list[list[str]]:
    parser = _TableParser()
    parser.feed(unescape(raw_html))
    parser.close()
    return parser.rows


def extract_tables(report_text: str, identity: ReportIdentity) -> list[TableRecord]:
    tables: list[TableRecord] = []
    for ordinal, match in enumerate(TABLE_RE.finditer(report_text), start=1):
        preceding_text = report_text[: match.start()]
        pages = list(PAGE_RE.finditer(preceding_text))
        page = int(pages[-1].group(1)) if pages else None
        raw_html = match.group(0)
        tables.append(
            TableRecord(
                table_id=f"{identity.report_id}|{ordinal}",
                report_id=identity.report_id,
                table_ordinal=ordinal,
                page=page,
                raw_html=raw_html,
                rows=parse_table_rows(raw_html),
            )
        )
    return tables


def iter_report_paths(dataset_root: Path) -> Iterable[Path]:
    return sorted((dataset_root / "financial_statements").rglob("*_extracted.txt"))

