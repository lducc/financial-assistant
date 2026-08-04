#!/usr/bin/env python3
"""Create the corrected E003 annotation queue with explicit stratum coverage."""

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
    elif any(word in lower for word in ("chênh lệch", "khác biệt", "bé hơn", "lớn hơn")):
        operation = "difference"
    elif any(word in lower for word in ("tổng cộng", "cộng dồn")):
        operation = "aggregate"
    else:
        operation = "lookup"
    years = sorted(set(YEAR_RE.findall(question)))
    tickers = sorted(set(TICKER_RE.findall(question)))
    if "công ty mẹ" in lower or "báo cáo riêng" in lower:
        scope = "separate_hint"
    elif "báo cáo hợp nhất" in lower or "công ty con" in lower:
        scope = "consolidated_hint"
    else:
        scope = "unspecified"
    if "%" in question or "phần trăm" in lower or "tỷ lệ" in lower:
        unit = "percent_hint"
    elif "tỷ đồng" in lower or "nghìn tỷ" in lower or "trăm tỷ" in lower:
        unit = "billion_vnd_hint"
    elif "triệu đồng" in lower:
        unit = "million_vnd_hint"
    elif "nghìn đồng" in lower:
        unit = "thousand_vnd_hint"
    else:
        unit = "unspecified"
    if operation == "lookup" and len(years) <= 1 and len(tickers) <= 1:
        complexity = "single_lookup"
    elif len(years) > 1:
        complexity = "multi_period"
    elif len(tickers) > 1:
        complexity = "multi_entity"
    else:
        complexity = "derived_or_compositional"
    return {
        "operation_hint": operation,
        "years": years,
        "tickers": tickers,
        "scope_hint": scope,
        "unit_hint": unit,
        "complexity_hint": complexity,
        "stratum": f"{operation}|{complexity}|{scope}|{unit}",
        "primary_stratum": f"{operation}|{complexity}",
    }


def stable_key(identifier: int, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def select(records: list[dict], target: int, seed: str) -> list[dict]:
    buckets: dict[str, deque[dict]] = {}
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_stratum[record["features"]["stratum"]].append(record)
    for stratum, candidates in by_stratum.items():
        candidates.sort(key=lambda row: stable_key(row["id"], seed))
        buckets[stratum] = deque(candidates)
    selected: list[dict] = []
    while len(selected) < target and any(buckets.values()):
        for stratum in sorted(buckets):
            if len(selected) >= target:
                break
            if buckets[stratum]:
                selected.append(buckets[stratum].popleft())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--seed", default="vifinqa-e003-v2")
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
        for field in ("operation_hint", "complexity_hint", "scope_hint", "unit_hint", "primary_stratum")
    }
    manifest = {
        "experiment": "E003",
        "selection": "deterministic round-robin across question-only full strata",
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

