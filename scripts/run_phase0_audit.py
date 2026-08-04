#!/usr/bin/env python3
"""Linear-time implementation of the locked E001/E002 audit.

Kept separate from the exploratory audit script so the recorded result includes
the exact audited method. It scans each report once for page and table markers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.catalog import iter_report_paths, parse_report_identity, parse_table_rows


PAGE_RE = re.compile(r"=====\s*PAGE\s+(\d+)\s*=====", re.IGNORECASE)
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
MOJIBAKE_RE = re.compile(r"[ÃÂÄÆ]")


def read_questions(path: Path) -> list[dict]:
    values: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), int) or not isinstance(row.get("question"), str):
            raise ValueError(f"Invalid question record at line {line_number}")
        values.append(row)
    return values


def audit(root: Path) -> dict:
    questions = read_questions(root / "questions" / "questions.jsonl")
    question_ids = [row["id"] for row in questions]
    scopes: Counter[str] = Counter()
    table_count = invalid_count = missing_page_count = 0
    no_table_reports: list[str] = []
    report_ids: set[str] = set()
    report_mojibake = 0
    rows_min = columns_min = None
    rows_max = columns_max = 0
    samples: list[dict] = []

    for path in iter_report_paths(root):
        identity = parse_report_identity(path, root)
        if identity.report_id in report_ids:
            raise ValueError(f"Duplicate report ID: {identity.report_id}")
        report_ids.add(identity.report_id)
        scopes[identity.scope] += 1
        text = path.read_text(encoding="utf-8")
        report_mojibake += bool(MOJIBAKE_RE.search(text))
        pages = list(PAGE_RE.finditer(text))
        page_index, current_page = 0, None
        matches = list(TABLE_RE.finditer(text))
        if not matches:
            no_table_reports.append(identity.report_id)
        for ordinal, match in enumerate(matches, 1):
            while page_index < len(pages) and pages[page_index].start() < match.start():
                current_page = int(pages[page_index].group(1))
                page_index += 1
            parsed_rows = parse_table_rows(match.group(0))
            row_count = len(parsed_rows)
            column_count = max((len(row) for row in parsed_rows), default=0)
            valid = bool(row_count and column_count)
            table_count += 1
            invalid_count += not valid
            missing_page_count += current_page is None
            rows_min = row_count if rows_min is None else min(rows_min, row_count)
            columns_min = column_count if columns_min is None else min(columns_min, column_count)
            rows_max = max(rows_max, row_count)
            columns_max = max(columns_max, column_count)
            if len(samples) < 5:
                samples.append(
                    {
                        "table_id": f"{identity.report_id}|{ordinal}",
                        "report_id": identity.report_id,
                        "table_ordinal": ordinal,
                        "page": current_page,
                        "row_count": row_count,
                        "column_count": column_count,
                        "valid_structure": valid,
                    }
                )

    result = {
        "audit_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "questions": {
            "count": len(questions),
            "unique_ids": len(set(question_ids)),
            "min_id": min(question_ids, default=None),
            "max_id": max(question_ids, default=None),
            "contiguous_from_one": sorted(question_ids) == list(range(1, len(question_ids) + 1)),
            "mojibake_marker_count": sum(bool(MOJIBAKE_RE.search(row["question"])) for row in questions),
        },
        "reports": {
            "count": len(report_ids),
            "unique_report_ids": len(report_ids),
            "scope_counts": dict(sorted(scopes.items())),
            "mojibake_marker_count": report_mojibake,
            "without_literal_html_tables": len(no_table_reports),
            "without_table_examples": no_table_reports[:20],
        },
        "tables": {
            "count": table_count,
            "structurally_invalid": invalid_count,
            "missing_page_context": missing_page_count,
            "min_rows": rows_min or 0,
            "max_rows": rows_max,
            "min_columns": columns_min or 0,
            "max_columns": columns_max,
            "sample_records": samples,
        },
    }
    return result


def render(result: dict) -> str:
    q, r, t = result["questions"], result["reports"], result["tables"]
    scope_rows = "\n".join(f"| {scope} | {count} |" for scope, count in r["scope_counts"].items())
    return f"""# ViFinQA E001/E002 audit\n\nGenerated: `{result['generated_at']}`\n\n| Check | Value |\n| --- | ---: |\n| Questions | {q['count']} |\n| Unique question IDs | {q['unique_ids']} |\n| IDs contiguous from 1 | {q['contiguous_from_one']} |\n| Reports | {r['count']} |\n| Unique report IDs | {r['unique_report_ids']} |\n| Literal HTML tables | {t['count']} |\n| Reports without literal tables | {r['without_literal_html_tables']} |\n| Structurally invalid literal tables | {t['structurally_invalid']} |\n| Tables without page context | {t['missing_page_context']} |\n\n## Filename-derived scope\n\n| Scope | Reports |\n| --- | ---: |\n{scope_rows}\n\n## Data-quality signals\n\n| Source | Records containing a possible mojibake marker |\n| --- | ---: |\n| Questions | {q['mojibake_marker_count']} |\n| Reports | {r['mojibake_marker_count']} |\n\nThe catalog key `report_id|document_wide_html_ordinal` is stable for research, but is **not** evidence that the organizer expects that ordinal in a submission.\n"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "results" / "e001_e002")
    args = parser.parse_args()
    result = audit(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "audit.md").write_text(render(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

