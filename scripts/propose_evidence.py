#!/usr/bin/env python3
"""Collect raw-text evidence candidates for a queue batch, for human adjudication.

Discovery is diacritic-folded substring matching over the gated reports' raw OCR
(`scripts/search_evidence.py`), never the retriever's ranking, so the resulting
labels do not encode what our own BM25 already finds.

Candidates are proposals, not labels. A row is emitted when its text contains the
question's metric phrase; the reviewer decides which rows are evidence. Questions
where exactly one row matches in each gated report are marked `unambiguous` and
still require review, but they are the cheap ones.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_evidence import fold, search

from vifinqa.jsonl import load_jsonl

# Question boilerplate: interrogatives, units, period wording, and scope wording.
# What survives is the metric phrase the statement row should carry.
BOILERPLATE = (
    r"la bao nhieu.*$", r"^tinh ", r"^cho biet ", r"bao nhieu.*$", r"\?$",
    r"\b(?:trieu|ty|nghin|tram) dong\b", r"\bphan tram\b", r"\bdon vi\b",
    r"\bcuoi nam \d{4}\b", r"\bdau nam \d{4}\b", r"\bnam \d{4}\b", r"\bngay \d{1,2}[/ ]",
    r"\bthang \d{1,2}\b", r"\bnam \d{4}\b", r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    r"\bcong ty me\b", r"\bhop nhat\b", r"\brieng\b", r"\btap doan\b",
    r"\bctcp\b", r"\bcong ty co phan\b", r"\btong cong ty\b", r"\bngan hang tmcp\b",
)
STOP_HEAD = ("cua", "tai", "vao", "den", "trong", "voi", "theo", "o")


def metric_phrase(question: str, tickers: list[str]) -> str:
    text = fold(question)
    for ticker in tickers:
        text = text.replace(fold(ticker), " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z0-9%\s]", " ", text)
    words = [word for word in text.split() if word]
    while words and words[0] in STOP_HEAD:
        words.pop(0)
    # Company names follow "cua"; the metric is what precedes it.
    if "cua" in words:
        words = words[: words.index("cua")]
    return " ".join(words[:12])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--queue", type=Path, default=ROOT / "annotations" / "v3" / "queue.jsonl")
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "v3" / "candidates.jsonl")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    queue = load_jsonl(args.queue)
    tiers = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.tiers.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    batch = [item for item in queue if item["batch"] == args.batch][: args.limit]

    records, status = [], Counter()
    for number, item in enumerate(batch, 1):
        meta = tiers[item["id"]]
        phrase = metric_phrase(item["question"], meta["tickers"])
        needles = phrase.split()
        hits = []
        # Every token must appear in the row, in any order, because OCR reorders and
        # splits cells. Relax from the front: Vietnamese noun phrases lead with the
        # generic head ("số dư", "tổng giá trị") and carry the discriminative words
        # at the tail, so dropping leading tokens widens the search the useful way.
        while needles and not hits:
            hits = search(args.dataset_root, meta["gated_report_ids"], needles, context=1)
            if not hits:
                needles = needles[1:]
        by_report = Counter(hit["table"].partition("|")[0] for hit in hits)
        state = (
            "no_match" if not hits
            else "unambiguous" if all(count == 1 for count in by_report.values())
            and len(by_report) == len(meta["gated_report_ids"])
            else "ambiguous"
        )
        status[state] += 1
        records.append({
            "id": item["id"],
            "question": item["question"],
            "tier": item["tier"],
            "tickers": meta["tickers"],
            "years": meta["years"],
            "scope": meta["scope"],
            "gated_reports": meta["gated_report_ids"],
            "metric_phrase": phrase,
            "matched_phrase": " ".join(needles),
            "status": state,
            "candidates": hits[: args.max_candidates],
            "candidate_count": len(hits),
        })
        if number % 10 == 0:
            print(f"proposed {number}/{len(batch)}", flush=True, file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "batch": args.batch, "questions": len(records), "status": dict(status),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
