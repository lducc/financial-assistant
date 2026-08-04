#!/usr/bin/env python3
"""Measure E011 adoption without changing raw questions or reports."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.text_repair import mojibake_score, repair_text


def audit(values: list[str]) -> dict:
    repaired = [repair_text(value) for value in values]
    return {
        "strings": len(values),
        "raw_marker_score": sum(mojibake_score(value) for value in values),
        "repaired_marker_score": sum(mojibake_score(value) for value, _, _ in repaired),
        "adopted": sum(adopted for _, adopted, _ in repaired),
        "codec_counts": dict(Counter(codec for _, adopted, codec in repaired if adopted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--report-limit", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = [json.loads(line)["question"] for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    reports = sorted(args.reports_root.rglob("*_extracted.txt"))[: args.report_limit]
    # Fixed first 80 lines/report avoids an unbounded full-corpus materialization.
    report_lines = [line for path in reports for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:80]]
    summary = {"experiment": "E011", "questions": audit(questions), "report_sample": {"reports": len(reports), **audit(report_lines)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
