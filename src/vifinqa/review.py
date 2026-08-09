"""Source-binding checks for the independently labelled pilot."""

from pathlib import Path
import re

from .retrieval import load_reports
from .tables import extract_rows_at_line


TABLE_ID = re.compile(r"^([^|\s]+)\|([1-9]\d*)$")


def source_report_catalog(raw_root: Path) -> dict[str, object]:
    """Index raw reports by the IDs used in pilot table bindings."""
    root = Path(raw_root).resolve()
    return {report.identity.report_id: report for report in load_reports(root)}


def validate_source_bindings(
    record: dict, raw_root: Path, reports: dict[str, object] | None = None,
) -> list[str]:
    """Validate annotated raw cells without evaluating answer formulas."""
    annotation = record.get("annotation")
    if not isinstance(annotation, dict) or annotation.get("status") != "complete":
        return []
    root = Path(raw_root).resolve()
    reports = reports or source_report_catalog(root)
    errors = []
    for index, binding in enumerate(annotation.get("row_column_bindings", [])):
        prefix = f"row_column_bindings[{index}]"
        if not isinstance(binding, dict):
            errors.append(f"{prefix} must be an object")
            continue
        match = TABLE_ID.fullmatch(str(binding.get("table", "")))
        if not match:
            errors.append(f"{prefix} invalid table ID")
            continue
        report_id, line = match.groups()
        report = reports.get(report_id)
        if report is None:
            errors.append(f"{prefix} missing source report: {report_id}")
            continue
        try:
            rows = extract_rows_at_line(report.path.read_text(encoding="utf-8"), int(line))
        except (OSError, ValueError) as error:
            errors.append(f"{prefix} cannot extract {binding['table']}: {error}")
            continue
        row, column = binding.get("row"), binding.get("column")
        if isinstance(row, bool) or isinstance(column, bool) or not isinstance(row, int) or not isinstance(column, int):
            errors.append(f"{prefix} row and column must be integers")
        elif row < 0 or column < 0 or row >= len(rows) or column >= len(rows[row]):
            errors.append(f"{prefix} cell out of range: {binding['table']} row={row} column={column}")
        elif rows[row][column] != binding.get("raw"):
            errors.append(f"{prefix} raw cell mismatch: {binding['table']} row={row} column={column}")
    return errors
