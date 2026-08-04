#!/usr/bin/env python3
"""Create a deterministic, question-only manual annotation queue for E003."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re


YEAR_RE = re.compile(r"\b20\d{2}\b")
TICKER_RE = re.compile(r"\(([A-Z]{2,5})\)")


def classify(question: str) -> dict:
    lower = question.lower()
    if any(word in lower for word in ("tăng trưởng", "tăng bao nhiêu", "giảm bao nhiêu", "so với năm")):
        operation = "growth_or_change"
    elif any(word in lower for word in ("tỷ lệ", "roe", "roa", "biên", "phần trăm")):
        operation = "ratio_or_percent"
    elif any(word in lower for word in ("trung bình", "bình quân")):
        operation = "average"
    elif any(word in lower for word in ("cao nhất", "lớn nhất", "thấp nhất", "nhỏ nhất")):
        operation = "extremum_or_selector"
    elif any(word in lower for word in ("tổng", "cộng dồn", "tổng cộng")):
        operation = "aggregate"
    elif any(word in lower for word in ("chênh lệch", "khác biệt")):
        operation = "difference"
    elif any(word in lower for word in ("bao nhiêu", "là gì", "giá trị")):
        operation = "lookup"
    else:
        operation = "other"
    years = YEAR_RE.findall(question)
    tickers = TICKER_RE.findall(question)
    if "công ty mẹ" in lower or "riêng" in lower:
        scope = "separate_hint"
    elif "hợp nhất" in lower or "tập đoàn" in lower:
        scope = "consolidated_hint"
    else:
        scope = "unspecified"
    if "%" in question or "phần trăm" in lower or "tỷ lệ" in lower:
        unit = "percent_hint"
    elif "tỷ đồng" in lower:
        unit = "billion_vnd_hint"
    elif "triệu đồng" in lower:
        unit = "million_vnd_hint"
    elif "nghìn đồng" in lower:
        unit = "thousand_vnd_hint"
    else:
        unit = "unspecified"
    complexity = "compositional" if operation != "lookup" or len(set(years)) > 1 or len(set(tickers)) > 1 else "single_lookup"
    return {
        "operation_hint": operation,
        "years": sorted(set(years)),
        "tickers": sorted(set(tickers)),
        "scope_hint": scope,
        "unit_hint": unit,
        "complexity_hint": complexity,
        "stratum": f"{operation}|{complexity}|{scope}|{unit}",
    }


def stable_key(identifier: int, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def select(records: list[dict], target: int, seed: str) -> list[dict]:
    by_operation: dict[str, deque[dict]] = {}
    for operation in sorted({row["features"]["operation_hint"] for row in records}):
        candidates = [row for row in records if row["features"]["operation_hint"] == operation]
        candidates.sort(key=lambda row: (row["features"]["stratum"], stable_key(row["id"], seed)))
        by_operation[operation] = deque(candidates)
    selected: list[dict] = []
    while len(selected) < target and any(by_operation.values()):
        for operation in sorted(by_operation):
            if len(selected) >= target:
                break
            if by_operation[operation]:
                selected.append(by_operation[operation].popleft())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--seed", default="vifinqa-e003-v1")
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/proxy_queue"))
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [{"id": row["id"], "question": row["question"], "features": classify(row["question"])} for row in questions]
    selected = select(records, args.target, args.seed)
    for row in selected:
        row["annotation"] = {
            "annotator": None,
            "question_slots": None,
            "required_metric_roles": None,
            "gold_reports": None,
            "gold_tables": None,
            "row_column_bindings": None,
            "table_units": None,
            "operation_graph": None,
            "pandas_query": None,
            "numeric_answer": None,
            "failure_tags": [],
            "confidence": None,
            "status": "unannotated",
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / "annotation_queue.jsonl"
    queue_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    feature_counts = {
        field: dict(sorted(Counter(row["features"][field] for row in selected).items()))
        for field in ("operation_hint", "complexity_hint", "scope_hint", "unit_hint", "stratum")
    }
    manifest = {
        "experiment": "E003",
        "selection": "deterministic round-robin across question-only operation hints",
        "source_questions": len(questions),
        "selected_questions": len(selected),
        "seed": args.seed,
        "feature_counts": feature_counts,
        "queue_path": str(queue_path),
        "label_status": "unannotated; no model comparison is permitted",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

