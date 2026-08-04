#!/usr/bin/env python3
"""Execute E005b and compare conservative bare-ticker coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.metadata import load_companies
from vifinqa.metadata_v2 import parse_question_with_bare_tickers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "derived" / "question_metadata_v2")
    args = parser.parse_args()
    companies = load_companies(args.companies)
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [{"id": row["id"], **parse_question_with_bare_tickers(row["question"], companies)} for row in questions]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "questions.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    source_counts = Counter(source["source"] for row in records for source in row["entity_candidates"])
    summary = {
        "experiment": "E005b",
        "generated_at": datetime.now(UTC).isoformat(),
        "question_count": len(records),
        "entity_resolution": {
            "resolved": sum(not row["unresolved_entity"] for row in records),
            "unresolved": sum(row["unresolved_entity"] for row in records),
            "candidate_count_distribution": dict(sorted(Counter(row["entity_count"] for row in records).items())),
            "candidate_source_counts": dict(sorted(source_counts.items())),
        },
        "year_coverage": sum(bool(row["years"]) for row in records),
        "scope_distribution": dict(sorted(Counter(row["scope"] or "unspecified" for row in records).items())),
        "output_path": str(output),
        "warning": "Exact known ticker coverage is not entity-linking accuracy; validate on the frozen proxy set.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

