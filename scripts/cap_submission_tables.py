#!/usr/bin/env python3
"""Cap the submitted table list, leaving the answer path untouched.

The budget is `min(30, 2 x gated reports)`, and the 30 has never been measured.
190 of 1,012 questions reach 9 or more tables and they account for 47.8% of the
5,938 tables submitted; the benchmark holds 26 such questions and its largest
budget is 16, so no local instrument can see the population that produces half
our precision loss.

The identity settles what a cap is worth without labels. F2 = 5h/(4G+k), so
dropping a table pays whenever its chance of being gold is below F2/5 = 0.104 at
our live F2. Measured on the benchmark, that chance is 0.161 at position 8, 0.231
at 9, 0.192 at 10, and unmeasurable after. Break-even for the tables a cap
removes:

    cap 20    230 dropped    pays if their gold rate is below 0.260
    cap 16    418 dropped    pays if below 0.190
    cap 12    714 dropped    pays if below 0.155
    cap 10    940 dropped    pays if below 0.143
    cap  8  1,320 dropped    pays if below 0.132

Truncating rather than re-running keeps this a clean test of one thing. The
ordering is unchanged, so a cap only ever removes trailing tables; `evidence`,
`pandas_query` and the CSVs are copied byte for byte, so answer and execution
accuracy cannot move and the whole delta belongs to the tables metrics. The
submission contract allows it: evidence names a `data/` path that exists, not a
table that was submitted.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

from validate_submission import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="a built package directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cap", type=int, required=True, help="most tables to submit per question")
    args = parser.parse_args()

    rows = json.loads((args.source / "submission.json").read_text(encoding="utf-8"))
    before = sum(len(row["relevant_tables"]) for row in rows)
    for row in rows:
        row["relevant_tables"] = row["relevant_tables"][:args.cap]
    after = sum(len(row["relevant_tables"]) for row in rows)

    package = args.output_dir / "package"
    if package.exists():
        shutil.rmtree(package)
    shutil.copytree(args.source, package)
    (package / "submission.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    errors = validate(package)
    if errors:
        raise SystemExit("validation failed:\n" + "\n".join(errors))

    csv_paths = {package / item["csv_path"] for row in rows for item in row["evidence"]}
    zip_path = args.output_dir / "submission.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        archive.write(package / "submission.json", "submission.json")
        for path in sorted(csv_paths | {p.with_suffix(".json") for p in csv_paths}):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())

    print(json.dumps({
        "cap": args.cap,
        "tables_before": before,
        "tables_after": after,
        "dropped": before - after,
        "questions_truncated": sum(1 for row in rows if len(row["relevant_tables"]) == args.cap),
        "mean_tables": round(after / len(rows), 2),
        "package": str(package),
        "zip": str(zip_path),
    }, indent=2))


if __name__ == "__main__":
    main()
