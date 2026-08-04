"""Lossless on-demand conversion of source OCR tables into evidence artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
from pathlib import Path

from .catalog import TABLE_RE, parse_table_rows
from .numeric import parse_number
from .numeric_v2 import detect_unit


@dataclass(frozen=True)
class MaterializedTable:
    table_id: str
    start_line: int
    rows: list[list[str]]
    unit_kind: str
    unit_scale_to_vnd: str | None


def extract_rows_at_line(report_text: str, start_line: int) -> list[list[str]]:
    """Return the literal table beginning at the requested 1-based source line."""

    current_line, cursor = 1, 0
    for match in TABLE_RE.finditer(report_text):
        current_line += report_text.count("\n", cursor, match.start())
        cursor = match.start()
        if current_line == start_line:
            return parse_table_rows(match.group(0))
    raise KeyError(f"No literal HTML table begins at source line {start_line}")


def annotate_rows(rows: list[list[str]]) -> list[list[dict]]:
    annotated: list[list[dict]] = []
    for row in rows:
        annotated_row = []
        for cell in row:
            parsed = parse_number(cell)
            annotated_row.append(
                {
                    "raw": cell,
                    "numeric": str(parsed.value) if parsed.value is not None else None,
                    "numeric_status": parsed.status,
                    "is_percent": parsed.is_percent,
                }
            )
        annotated.append(annotated_row)
    return annotated


def materialize(report_path: Path, start_line: int, table_id: str, csv_path: Path, sidecar_path: Path) -> MaterializedTable:
    report_text = report_path.read_text(encoding="utf-8")
    rows = extract_rows_at_line(report_text, start_line)
    if not rows:
        raise ValueError(f"Table {table_id} is empty")
    column_count = max(len(row) for row in rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([row + [""] * (column_count - len(row)) for row in rows])
    unit = detect_unit(" ".join(cell for row in rows[:3] for cell in row))
    record = MaterializedTable(
        table_id=table_id,
        start_line=start_line,
        rows=rows,
        unit_kind=unit.kind,
        unit_scale_to_vnd=str(unit.scale_to_vnd) if unit.scale_to_vnd is not None else None,
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = asdict(record)
    sidecar["annotated_rows"] = annotate_rows(rows)
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return record

