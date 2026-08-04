#!/usr/bin/env python3
"""E009 repair: full planner run with boolean structural coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.planner import build_plan


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "derived" / "template_plans")
    args = parser.parse_args()
    metadata = {row["id"]: row for row in read_jsonl(args.metadata)}
    plans = [{"id": row["id"], **build_plan(row["question"], metadata.get(row["id"], {}))} for row in read_jsonl(args.questions)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "plans.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plans), encoding="utf-8")
    summary = {
        "experiment": "E009",
        "generated_at": datetime.now(UTC).isoformat(),
        "question_count": len(plans),
        "valid_plan_count": sum(bool(row["operation"] and row["operand_roles"]) for row in plans),
        "operation_distribution": dict(sorted(Counter(row["operation"] for row in plans).items())),
        "unit_hint_distribution": dict(sorted(Counter(row["output_unit_hint"] for row in plans).items())),
        "output_path": str(output),
        "warning": "This reports structural coverage only, not operation classification accuracy.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
