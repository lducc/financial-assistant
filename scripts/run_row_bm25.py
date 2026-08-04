#!/usr/bin/env python3
"""Run E008 row-centric BM25 and save row/table grounding traces."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import load_reports
from vifinqa.row_retrieval import retrieve_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "results" / "e008_row_bm25")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    reports = load_reports(args.dataset_root)
    traces = [{"id": row["id"], "question": row["question"], "metadata": row["hard_filter"], **retrieve_rows(row["question"], row, reports, args.top_k)} for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "traces.jsonl"
    trace_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in traces), encoding="utf-8")
    summary = {
        "experiment": "E008",
        "generated_at": datetime.now(UTC).isoformat(),
        "questions_run": len(traces),
        "mean_candidate_reports": round(statistics.fmean(row["candidate_report_count"] for row in traces), 3) if traces else 0,
        "mean_candidate_rows": round(statistics.fmean(row["candidate_row_count"] for row in traces), 3) if traces else 0,
        "mean_candidate_tables": round(statistics.fmean(row["candidate_table_count"] for row in traces), 3) if traces else 0,
        "median_latency_ms": round(statistics.median(row["latency_ms"] for row in traces), 3) if traces else 0,
        "filter_stage_counts": dict(sorted(Counter(row["filter_stage"] for row in traces).items())),
        "zero_score_count": sum(row["zero_score"] for row in traces),
        "trace_path": str(trace_path),
        "warning": "No gold row bindings were used; this is not a retrieval accuracy result.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

