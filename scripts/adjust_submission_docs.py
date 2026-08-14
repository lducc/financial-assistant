#!/usr/bin/env python3
"""Rewrite `relevant_docs` in a built submission, leaving tables and answers alone.

Docs F2 and Tables F2 are scored independently, so the document list can be
varied without disturbing anything else in a package — which makes it possible
to test a document hypothesis on a submission that is already being spent on a
retrieval change.

Two edits are available, and both are shaped by what `src/docs.py` will accept:
every submitted table's report must appear in `relevant_docs`, so the smallest
legal document set is the set of reports that contributed a table.

`--add` widens the list. It takes the probe output naming (question, report)
pairs where the evidence says a report we never gated does hold the line item —
adding those costs precision only where we are already known to be wrong.

`--prune-empty` narrows it to reports that contributed a table. Measured against
the 8B ranking this dropped 63 reports of which 9 were gold, so it is kept for
completeness rather than recommended.

Nothing here re-runs retrieval; the package is read, its documents rewritten,
and it is repacked.
"""

import argparse
import json
from pathlib import Path
import shutil
import zipfile


def load_package(path: Path) -> tuple[Path, list[dict]]:
    """The package directory and its rows, accepting either a directory or a zip."""
    if path.suffix == ".zip":
        raise SystemExit("point at the package directory, not the zip; the zip is rebuilt from it")
    root = path / "package" if (path / "package").is_dir() else path
    submission = root / "submission.json"
    if not submission.is_file():
        raise SystemExit(f"no submission.json under {root}")
    return root, json.loads(submission.read_text(encoding="utf-8"))


def report_of(table_id: str) -> str:
    return table_id.partition("|")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="built submission directory")
    parser.add_argument("destination", type=Path, help="directory to write the variant into")
    parser.add_argument(
        "--add", type=Path,
        help="JSON list of [question_id, report_id, ...] pairs to add to relevant_docs",
    )
    parser.add_argument(
        "--prune-empty", action="store_true",
        help="keep only reports that contributed a submitted table",
    )
    args = parser.parse_args()

    root, rows = load_package(args.source)

    additions: dict[int, set[str]] = {}
    if args.add:
        for entry in json.loads(args.add.read_text(encoding="utf-8")):
            additions.setdefault(int(entry[0]), set()).add(entry[1])

    widened = narrowed = 0
    for row in rows:
        docs = list(row["relevant_docs"])
        required = {report_of(table) for table in row.get("relevant_tables", [])}
        if args.prune_empty:
            kept = [doc for doc in docs if doc in required]
            narrowed += len(docs) - len(kept)
            docs = kept or docs
        for doc in sorted(additions.get(int(row["id"]), ())):
            if doc not in docs:
                docs.append(doc)
                widened += 1
        # The validator rejects a table whose report is missing, so the required
        # set is restored last no matter what the edits above did.
        docs.extend(sorted(required - set(docs)))
        row["relevant_docs"] = docs

    if args.destination.exists():
        shutil.rmtree(args.destination)
    shutil.copytree(root, args.destination / "package")
    target = args.destination / "package" / "submission.json"
    target.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    archive = args.destination / "submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((args.destination / "package").rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(args.destination / "package").as_posix())

    total = sum(len(row["relevant_docs"]) for row in rows)
    print(json.dumps({
        "output": str(archive),
        "questions": len(rows),
        "documents": total,
        "documents_per_question": round(total / max(1, len(rows)), 3),
        "added": widened,
        "pruned": narrowed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
