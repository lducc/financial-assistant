#!/usr/bin/env python3
"""Run the E001/E002 lossless inventory and HTML-table audit."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.catalog import extract_tables, iter_report_paths, parse_report_identity


MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Æ")


def read_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value.get("id"), int) or not isinstance(value.get("question"), str):
                raise ValueError(f"Invalid question record at line {number}")
            questions.append(value)
    return questions


def has_mojibake(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def audit(dataset_root: Path) -> dict:
    questions = read_questions(dataset_root / "questions" / "questions.jsonl")
    question_ids = [question["id"] for question in questions]
    report_paths = list(iter_report_paths(dataset_root))

    scope_counts: Counter[str] = Counter()
    report_ids: set[str] = set()
    report_mojibake = 0
    reports_without_tables: list[str] = []
    total_tables = 0
    malformed_tables = 0
    page_missing = 0
    table_row_counts: list[int] = []
    table_column_counts: list[int] = []
    examples: list[dict] = []

    for report_path in report_paths:
        identity = parse_report_identity(report_path, dataset_root)
        if identity.report_id in report_ids:
            raise ValueError(f"Duplicate report identifier: {identity.report_id}")
        report_ids.add(identity.report_id)
        scope_counts[identity.scope] += 1
        text = report_path.read_text(encoding="utf-8")
        report_mojibake += int(has_mojibake(text))
        tables = extract_tables(text, identity)
        total_tables += len(tables)
        if not tables:
            reports_without_tables.append(identity.report_id)
        for table in tables:
            malformed_tables += int(not table.valid_structure)
            page_missing += int(table.page is None)
            table_row_counts.append(len(table.rows))
            table_column_counts.append(table.column_count)
            if len(examples) < 5:
                examples.append(table.as_dict(include_html=False))

    return {
        "audit_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset_root),
        "questions": {
            "count": len(questions),
            "unique_ids": len(set(question_ids)),
            "min_id": min(question_ids, default=None),
            "max_id": max(question_ids, default=None),
            "contiguous_from_one": sorted(question_ids) == list(range(1, len(question_ids) + 1)),
            "mojibake_marker_count": sum(has_mojibake(q["question"]) for q in questions),
        },
        "reports": {
            "count": len(report_paths),
            "unique_report_ids": len(report_ids),
            "scope_counts": dict(sorted(scope_counts.items())),
            "mojibake_marker_count": report_mojibake,
            "without_literal_html_tables": len(reports_without_tables),
            "without_table_examples": reports_without_tables[:20],
        },
        "tables": {
            "count": total_tables,
            "structurally_invalid": malformed_tables,
            "missing_page_context": page_missing,
            "min_rows": min(table_row_counts, default=0),
            "max_rows": max(table_row_counts, default=0),
            "min_columns": min(table_column_counts, default=0),
            "max_columns": max(table_column_counts, default=0),
            "sample_records": examples,
        },
    }


def render_markdown(result: dict) -> str:
    questions = result["questions"]
    reports = result["reports"]
    tables = result["tables"]
    scopes = "\n".join(f"| {name} | {count} |" for name, count in reports["scope_counts"].items())
    return f"""# ViFinQA E001/E002 audit\n\nGenerated: `{result['generated_at']}`\n\n## Integrity\n\n| Check | Value |\n| --- | ---: |\n| Questions | {questions['count']} |\n| Unique question IDs | {questions['unique_ids']} |\n| Contiguous IDs from 1 | {questions['contiguous_from_one']} |\n| Reports | {reports['count']} |\n| Unique report IDs | {reports['unique_report_ids']} |\n| Literal HTML tables | {tables['count']} |\n| Reports without literal tables | {reports['without_literal_html_tables']} |\n| Structurally invalid literal tables | {tables['structurally_invalid']} |\n| Tables without page context | {tables['missing_page_context']} |\n\n## Filename-derived scope\n\n| Scope | Reports |\n| --- | ---: |\n{scopes}\n\n## Source-corruption signals\n\nThe marker count is diagnostic only; it must not trigger silent data repair.\n\n| Source | Records containing a marker |\n| --- | ---: |\n| Questions | {questions['mojibake_marker_count']} |\n| Reports | {reports['mojibake_marker_count']} |\n\n## Interpretation\n\nTable IDs in this audit have the form `report_id|document_wide_html_ordinal`. They are stable catalog keys, but not yet a claim that this ordinal matches the competition's required `relevant_tables` position.\n"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "results" / "e001_e002")
    args = parser.parse_args()
    result = audit(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "audit.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

