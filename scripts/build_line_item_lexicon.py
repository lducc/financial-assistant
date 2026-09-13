# Build the corpus-derived line-item vocabulary.
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import load_reports, report_tables, tokenize

ROOT = Path(__file__).resolve().parents[1]
MIN_TOKENS, MAX_TOKENS = 2, 8
MIN_REPORTS = 10
MAX_QUESTION_SHARE = 0.15


def row_label(row: tuple[str, ...]) -> str:
    return " ".join(tokenize(max(row[:2], key=len) if len(row) > 1 else row[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "derived" / "line_items.json")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    reports = load_reports(args.dataset_root)
    report_frequency: Counter[str] = Counter()
    for number, report in enumerate(reports, 1):
        seen = set()
        for table in report_tables(str(report.path), report.identity):
            for row in table.rows:
                if not row:
                    continue
                label = row_label(row)
                token_count = len(label.split())
                if (MIN_TOKENS <= token_count <= MAX_TOKENS
                        and not label.replace(" ", "").isdigit()):
                    seen.add(label)
        report_frequency.update(seen)
        report_tables.cache_clear()
        if args.progress_every and number % args.progress_every == 0:
            print(f"indexed {number}/{len(reports)} reports", file=sys.stderr, flush=True)

    frequent = {
        label for label, count in report_frequency.items()
        if count >= MIN_REPORTS
    }
    questions = load_jsonl(args.dataset_root / "questions" / "questions.jsonl")
    question_frequency: Counter[str] = Counter()
    question_texts = [" ".join(tokenize(record["question"])) for record in questions]
    for text in question_texts:
        question_frequency.update(label for label in frequent if label in text)

    limit = MAX_QUESTION_SHARE * len(questions)
    lexicon = {
        label: {
            "reports": report_frequency[label],
            "questions": question_frequency[label],
        }
        for label in sorted(frequent)
        if question_frequency[label] <= limit
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(lexicon, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    dropped = sorted(
        (label for label in frequent if question_frequency[label] > limit),
        key=lambda label: (-question_frequency[label], label),
    )
    matched = sum(any(label in text for label in lexicon) for text in question_texts)
    print(json.dumps({
        "output": str(args.output),
        "labels_seen": len(report_frequency),
        "frequent_labels": len(frequent),
        "lexicon": len(lexicon),
        "dropped_as_boilerplate": dropped[:12],
        "questions_matching_any": matched,
        "questions": len(questions),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
