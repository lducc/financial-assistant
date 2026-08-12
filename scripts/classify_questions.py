#!/usr/bin/env python3
"""Classify every ViFinQA question into the organizer's four difficulty tiers.

The organizer slides publish tier counts over the full question set (Easy 361,
Medium 235, Intermediate 200, Hard 216). Those counts are the only external
ground truth we hold about question structure, so they are used here to check a
classifier we can then apply to all 1,012 questions, including the 862 that carry
no retrieval labels.

Tier definitions, taken from the slides:

* Easy         - read one numeric value from one table.
* Medium       - two values and one arithmetic operation.
* Intermediate - more than two values with repeated or grouped operations.
* Hard         - multi-hop, where an intermediate result selects the next table,
                 company, or period.

Entities, years, and scope come from `docs.parse_question`, the same parser the
retrieval pipeline gates on, so the tiers describe what the system actually sees.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docs import load_companies, load_reports as load_doc_reports, parse_question, retrieve_docs


PUBLISHED_TIERS = {"easy": 361, "medium": 235, "intermediate": 200, "hard": 216}
SELECTOR_CUES = ("cao nhat", "thap nhat", "lon nhat", "nho nhat", "nhieu nhat", "it nhat", "dan dau")
DEPENDENT_CUES = (
    "nam ma", "vao nam ma", "o nam ma", "tai nam ma", "ngay sau", "ngay truoc",
    "doanh nghiep co", "cong ty co", "trong so cac", "trong nhom", "trong so ",
)
GROUPED_CUES = ("trung binh", "binh quan", "trung vi", "tong cong", "tong hop")
PAIRED_CUES = ("chenh lech", "tang truong", "so voi", "ty le", "ty trong", "bien loi nhuan", " gap ")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")


def classify(question: str, tickers: list[str], years: list[int]) -> str:
    text = f" {fold(question)} "
    entities, periods = max(1, len(tickers)), max(1, len(years))
    selector = any(cue in text for cue in SELECTOR_CUES)
    dependent = any(cue in text for cue in DEPENDENT_CUES)
    # Multi-hop: the question first picks a company, year, or row, then asks for a
    # value determined by that choice.
    if selector and (dependent or periods > 2 or entities > 2):
        return "hard"
    operands = entities * periods
    if operands > 2 or any(cue in text for cue in GROUPED_CUES):
        return "intermediate"
    if operands == 2 or selector or any(cue in text for cue in PAIRED_CUES):
        return "medium"
    return "easy"


def diagnostics(parsed, docs: list[str]) -> dict:
    return {
        "tickers": parsed.tickers,
        "years": parsed.years,
        "scope": parsed.scope,
        "entity_resolved": bool(parsed.tickers),
        "year_resolved": bool(parsed.years),
        "gated_reports": len(docs),
        "gated_report_ids": docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in (args.dataset_root / "questions" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    companies = load_companies(args.dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")

    records = []
    for question in questions:
        parsed = parse_question(question["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, doc_reports)
        records.append({
            "id": question["id"],
            "question": question["question"],
            "tier": classify(question["question"], parsed.tickers, parsed.years),
            **diagnostics(parsed, docs),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )

    counts = Counter(record["tier"] for record in records)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["tier"]].append(record)
    summary = {
        "questions": len(records),
        "tiers": {tier: counts[tier] for tier in PUBLISHED_TIERS},
        "published_tiers": PUBLISHED_TIERS,
        "absolute_deviation": sum(abs(counts[tier] - published) for tier, published in PUBLISHED_TIERS.items()),
        "by_tier": {
            tier: {
                "questions": len(items),
                "entity_resolved": round(sum(item["entity_resolved"] for item in items) / len(items), 4),
                "year_resolved": round(sum(item["year_resolved"] for item in items) / len(items), 4),
                "mean_gated_reports": round(sum(item["gated_reports"] for item in items) / len(items), 2),
                "max_gated_reports": max(item["gated_reports"] for item in items),
                "zero_gated_reports": sum(1 for item in items if not item["gated_reports"]),
                "mean_entities": round(sum(max(1, len(item["tickers"])) for item in items) / len(items), 2),
                "mean_years": round(sum(max(1, len(item["years"])) for item in items) / len(items), 2),
            }
            for tier, items in sorted(grouped.items())
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
