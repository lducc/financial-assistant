#!/usr/bin/env python3
"""Find candidate evidence rows by searching raw OCR text, for blind annotation.

Annotation must not inherit the retriever's opinion: if candidates arrive ranked
by the same BM25 the benchmark later scores, the labels encode what that ranker
already finds and recall is measured against itself. This tool does plain
diacritic-folded substring matching over the raw report text of the gated
reports, ordered by document position, and never scores or ranks.

Usage:
    python3 scripts/search_evidence.py --id 42 "tien va cac khoan tuong duong tien"
    python3 scripts/search_evidence.py --report VJC_..._2018_separate "loi nhuan sau thue"
"""

import argparse
import json
from pathlib import Path
import unicodedata

from vifinqa.retrieval import report_tables
from vifinqa.tables import parse_report_identity

ROOT = Path(__file__).resolve().parents[1]


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")


def report_paths(dataset_root: Path, report_ids: list[str]) -> list[Path]:
    wanted = set(report_ids)
    return [
        path for path in sorted((dataset_root / "financial_statements").rglob("*_extracted.txt"))
        if path.stem.removesuffix("_extracted") in wanted
    ]


def search(dataset_root: Path, report_ids: list[str], needles: list[str], context: int) -> list[dict]:
    folded_needles = [fold(needle) for needle in needles]
    hits = []
    for path in report_paths(dataset_root, report_ids):
        identity = parse_report_identity(path, dataset_root)
        for table in report_tables(str(path), identity):
            for index, row in enumerate(table.rows):
                text = fold(" ".join(row))
                if all(needle in text for needle in folded_needles):
                    hits.append({
                        "table": table.table_id,
                        "row": index,
                        "page": table.page,
                        "title": table.title[:90],
                        "periods": list(table.periods),
                        "unit": table.unit[:60],
                        "header_rows": [list(header) for header in table.headers[:2]],
                        "cells": list(row),
                        "neighbors": [
                            " | ".join(table.rows[other])[:110]
                            for other in range(max(0, index - context), min(len(table.rows), index + context + 1))
                            if other != index
                        ],
                    })
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("needles", nargs="+", help="folded substrings that must all appear in the row")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--queue", type=Path, default=ROOT / "annotations" / "v3" / "queue.jsonl")
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument("--id", type=int, help="queue question ID; searches that question's gated reports")
    parser.add_argument("--report", action="append", default=[], help="explicit report ID; repeatable")
    parser.add_argument("--context", type=int, default=1, help="neighbouring rows to show for label context")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    reports = list(args.report)
    if args.id is not None:
        tiers = {
            json.loads(line)["id"]: json.loads(line)
            for line in args.tiers.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        record = tiers[args.id]
        reports.extend(record["gated_report_ids"])
        print(json.dumps({
            "id": args.id, "question": record["question"], "tier": record["tier"],
            "tickers": record["tickers"], "years": record["years"], "scope": record["scope"],
            "gated_reports": record["gated_report_ids"],
        }, ensure_ascii=False, indent=2))
    if not reports:
        raise SystemExit("provide --id or at least one --report")

    hits = search(args.dataset_root, reports, args.needles, args.context)
    print(json.dumps({"matches": len(hits), "shown": min(len(hits), args.limit)}, ensure_ascii=False))
    for hit in hits[: args.limit]:
        print(json.dumps(hit, ensure_ascii=False))


if __name__ == "__main__":
    main()
