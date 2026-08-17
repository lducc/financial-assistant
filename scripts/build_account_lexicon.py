#!/usr/bin/env python3
"""Read the corpus's own account codes into a lexicon and a table structure index.

Two artefacts, both derived from the filings and from no labels at all:

`account_lexicon.json` maps each `Mã số` account code to every row label the
corpus writes against it, with counts. That is the line-item vocabulary of
Vietnamese financial statements together with its OCR damage, which is the
discrimination the reranker has been asked to make from the question text alone.

`table_structure.jsonl` says what each table is — a primary statement, a note, or
neither — which account codes it carries, which notes its rows point at, and, for
a note, the number its heading opens with. The statement-to-note pointer and the
note heading are the same string, so the second and third gold tables of a
question are reachable from the first by a join rather than by a ranker.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from vifinqa.jsonl import load_jsonl, write_jsonl
from vifinqa.statements import statement_rows, table_heading

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/derived/table_catalog/tables.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "data/derived/account_lexicon.json")
    parser.add_argument("--structure", type=Path, default=ROOT / "data/derived/table_structure.jsonl")
    args = parser.parse_args()

    by_report = defaultdict(list)
    for record in load_jsonl(args.catalog):
        by_report[record["source_path"]].append(record)

    lexicon = defaultdict(Counter)
    structure = []
    for source, tables in sorted(by_report.items()):
        lines = (args.corpus / source).read_text(encoding="utf-8", errors="replace").split("\n")
        for record in sorted(tables, key=lambda r: r["start_line"]):
            line = record["start_line"]
            table = lines[line - 1] if line <= len(lines) else ""
            codes, notes = [], {}
            for label, code, reference in statement_rows(table):
                lexicon[code][label] += 1
                codes.append(code)
                if reference:
                    notes[code] = reference
            number, heading = table_heading(lines, line - 1)
            structure.append({
                "id": record["submission_table_id"],
                "report_id": record["report_id"],
                "kind": "statement" if codes else "note" if heading else "other",
                "codes": sorted(set(codes)),
                "note_by_code": notes,
                "note_number": number,
                "heading": heading,
            })

    write_jsonl(args.structure, structure)
    args.lexicon.write_text(
        json.dumps(
            {code: dict(labels.most_common()) for code, labels in sorted(lexicon.items())},
            ensure_ascii=False, indent=1,
        ) + "\n",
        encoding="utf-8",
    )

    kinds = Counter(row["kind"] for row in structure)
    linked = sum(1 for row in structure if row["note_by_code"])
    print(json.dumps({
        "reports": len(by_report),
        "tables": len(structure),
        "kinds": dict(kinds),
        "statements_pointing_at_notes": linked,
        "codes": len(lexicon),
        "label_variants": sum(len(labels) for labels in lexicon.values()),
        "observations": sum(sum(labels.values()) for labels in lexicon.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
