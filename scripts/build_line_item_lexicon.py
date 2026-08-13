#!/usr/bin/env python3
"""Collect the corpus's own line-item vocabulary for query decomposition.

Splitting a question on connectives produces fragments that no table contains
("tổng tài sản cuối 2016 phần trăm"), and those fragments retrieve worse than the
whole question. Row labels do not have that problem: they are the exact strings
the statements use. Matching a question against them decomposes it into the
figures it names and, on the way, bridges the gap between how a question says a
line item ("lãi vay") and how the statement writes it ("chi phí lãi vay").

Two frequencies are recorded per phrase. Corpus frequency, in reports, keeps
phrases that are real accounting vocabulary rather than one report's wording.
Question frequency drops scope boilerplate: "của công ty mẹ" appears inside row
labels and in most questions, so it decomposes nothing.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import load_reports, report_tables, tokenize


def row_label(row: tuple[str, ...]) -> str:
    """The longest of a row's leading cells, which is where the label sits."""
    return " ".join(tokenize(max(row[:2], key=len) if len(row) > 1 else row[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "derived" / "line_items.json")
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--min-reports", type=int, default=10)
    parser.add_argument("--max-question-share", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    reports = load_reports(args.dataset_root)
    report_frequency: Counter = Counter()
    for number, report in enumerate(reports, 1):
        seen: set[str] = set()
        for table in report_tables(str(report.path), report.identity):
            for row in table.rows:
                if not row:
                    continue
                label = row_label(row)
                if args.min_tokens <= len(label.split()) <= args.max_tokens and not label.replace(" ", "").isdigit():
                    seen.add(label)
        report_frequency.update(seen)
        report_tables.cache_clear()
        if args.progress_every and number % args.progress_every == 0:
            print(f"indexed {number}/{len(reports)} reports", flush=True, file=sys.stderr)

    frequent = {label for label, count in report_frequency.items() if count >= args.min_reports}

    questions = [
        json.loads(line)
        for line in (args.dataset_root / "questions" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    question_frequency: Counter = Counter()
    for question in questions:
        text = " ".join(tokenize(question["question"]))
        question_frequency.update(label for label in frequent if label in text)

    limit = args.max_question_share * len(questions)
    lexicon = {
        label: {"reports": report_frequency[label], "questions": question_frequency[label]}
        for label in frequent
        if question_frequency[label] <= limit
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lexicon, ensure_ascii=False), encoding="utf-8")

    dropped = sorted(
        (label for label in frequent if question_frequency[label] > limit),
        key=lambda label: -question_frequency[label],
    )
    print(json.dumps({
        "output": str(args.output),
        "labels_seen": len(report_frequency),
        "frequent_labels": len(frequent),
        "lexicon": len(lexicon),
        "dropped_as_boilerplate": dropped[:12],
        "questions_matching_any": sum(
            1 for question in questions
            if any(label in " ".join(tokenize(question["question"])) for label in lexicon)
        ),
        "questions": len(questions),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
