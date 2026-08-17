#!/usr/bin/env python3
"""Re-render the candidate text of an exported pairs file.

The shipped representation is a summary: the title, the one row BM25 matched with
its numbers, the other row labels with no numbers and cut at 600 characters, then
the headers. The whole table is a median 971 characters against 656 for that
summary and exceeds the 1024-token window on 0.2% of candidates, so the model is
reading a lossy description of something that would have fitted uncompressed.

That matters because the judgement is whether the table holds the asked item as a
row with a value for the period — checkable for the matched row, guesswork for
every other one.

`--mode full` renders the table as its rows. `--mode codes` adds the mandated
`Mã số` values, which are three digits and survive the OCR damage that wrecks
Vietnamese labels. Candidates, ordering and identifiers are untouched, so this
changes only what the model reads.
"""

import argparse
import json
from pathlib import Path

from vifinqa.jsonl import load_jsonl, write_jsonl
from vifinqa.statements import CODE, rows as parse_rows

ROOT = Path(__file__).resolve().parents[1]


def render(table: str, title: str, mode: str, budget: int) -> str:
    lines = [title] if title else []
    codes = []
    for row in parse_rows(table):
        cells = [cell for cell in row if cell]
        if not cells:
            continue
        lines.append(" | ".join(cells))
        codes += [cell for cell in row[1:3] if CODE.match(cell.strip())]
    if mode == "codes" and codes:
        lines.insert(1, "Mã số: " + ", ".join(dict.fromkeys(codes)))
    text = "\n".join(lines)
    return text[:budget]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "codes"), default="full")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/derived/table_catalog/tables.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--budget", type=int, default=3500, help="characters, about a 1024-token window")
    args = parser.parse_args()

    catalog = {record["submission_table_id"]: record for record in load_jsonl(args.catalog)}
    cache, records, missing = {}, list(load_jsonl(args.pairs)), 0
    for record in records:
        for candidate in record["candidates"]:
            entry = catalog.get(candidate["table_id"])
            if entry is None:
                missing += 1
                continue
            source = entry["source_path"]
            if source not in cache:
                cache[source] = (args.corpus / source).read_text(encoding="utf-8", errors="replace").split("\n")
            lines = cache[source]
            if entry["start_line"] > len(lines):
                missing += 1
                continue
            title = candidate["text"].split("\n")[0]
            candidate["text"] = render(lines[entry["start_line"] - 1], title, args.mode, args.budget)

    write_jsonl(args.output, records)
    lengths = [len(c["text"]) for r in records for c in r["candidates"]]
    print(json.dumps({
        "questions": len(records), "candidates": len(lengths), "kept_original_text": missing,
        "median_chars": sorted(lengths)[len(lengths) // 2],
        "over_budget": sum(1 for n in lengths if n >= args.budget),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
