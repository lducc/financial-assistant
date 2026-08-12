#!/usr/bin/env python3
"""Precompute corpus-wide document frequency for BM25 row scoring.

BM25 in `vifinqa.retrieval` scores rows inside one question's gated candidate
set, so its document frequency comes from a few thousand rows of a single
company. A term that is rare in that slice but common across the corpus gets a
large weight, which makes ranking depend on the gate rather than on the term.
This builds the statistics once over every parsed row so scoring uses a stable
corpus-level IDF.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.retrieval import (
    contextual_row, load_reports, report_tables, tokenize, unicode_tokenize,
)


def build(dataset_root: Path, progress_every: int) -> dict:
    reports = load_reports(dataset_root)
    frequency: dict[str, Counter] = {
        "row": Counter(), "context": Counter(), "unicode_row": Counter(), "unicode_context": Counter(),
    }
    totals = Counter()
    lengths = Counter()
    for number, report in enumerate(reports, 1):
        for table in report_tables(str(report.path), report.identity):
            for row in table.rows:
                row_text = " ".join(row)
                context_text = contextual_row(table, row_text)
                for field, text, tokenizer in (
                    ("row", row_text, tokenize),
                    ("context", context_text, tokenize),
                    ("unicode_row", row_text, unicode_tokenize),
                    ("unicode_context", context_text, unicode_tokenize),
                ):
                    tokens = tokenizer(text)
                    frequency[field].update(set(tokens))
                    totals[field] += 1
                    lengths[field] += max(1, len(tokens))
        # Parsed tables are cached per report; release them before the next one.
        report_tables.cache_clear()
        if progress_every and number % progress_every == 0:
            print(f"indexed {number}/{len(reports)} reports", flush=True)
    return {
        "documents": dict(totals),
        "average_length": {field: lengths[field] / totals[field] for field in totals},
        # Singletons are a third of the vocabulary and never change a ranking.
        "document_frequency": {
            field: {token: count for token, count in counts.items() if count > 1}
            for field, counts in frequency.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "derived" / "row_idf.json")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    statistics = build(args.dataset_root, args.progress_every)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(statistics, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "documents": statistics["documents"],
        "vocabulary": {field: len(counts) for field, counts in statistics["document_frequency"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
