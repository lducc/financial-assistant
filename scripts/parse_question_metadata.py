#!/usr/bin/env python3
"""Execute E005: parse safe metadata from every public ViFinQA question."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.metadata import load_companies, parse_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "derived" / "question_metadata")
    args = parser.parse_args()
    companies = load_companies(args.companies)
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for row in questions:
        parsed = parse_question(row["question"], companies)
        records.append({"id": row["id"], **parsed})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "questions.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {
        "experiment": "E005",
        "generated_at": datetime.now(UTC).isoformat(),
        "question_count": len(records),
        "company_map_count": len(companies),
        "entity_resolution": {
            "resolved": sum(not row["unresolved_entity"] for row in records),
            "unresolved": sum(row["unresolved_entity"] for row in records),
            "candidate_count_distribution": dict(sorted(Counter(row["entity_count"] for row in records).items())),
        },
        "year_coverage": {
            "with_year": sum(bool(row["years"]) for row in records),
            "without_year": sum(not row["years"] for row in records),
        },
        "scope_distribution": dict(sorted(Counter(row["scope"] or "unspecified" for row in records).items())),
        "output_path": str(output),
        "warning": "Coverage is not accuracy; validate against frozen proxy annotations before applying hard filters beyond explicit metadata.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

