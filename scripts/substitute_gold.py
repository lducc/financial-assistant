#!/usr/bin/env python3
"""Submit our own labels for the benchmark questions, to score the labels.

Every decision here rests on the 233-record benchmark, and the benchmark rests on
gold we wrote ourselves. Live reads 22% below it — 0.5221 against 0.6676 — and
nothing local can say whether that is the public split being harder or our gold
being wrong 22% of the time. The scorer can. Replacing the shipped table list
with our gold on the benchmark questions, and nothing else, turns the delta into
a direct read on label agreement:

    perfect agreement, random half public   ->  about +0.11 F2
    78% agreement                           ->  about +0.06

Precision and recall move separately, and their asymmetry says which way we are
wrong. P near 1 with R short means the organizers count tables we do not, so
gold should widen; R near 1 with P short means we count tables they do not, so
it should narrow. Built once for each definition, so the pair also says which
one they use.

The other 779 questions, the docs, the answers and the queries are copied byte
for byte, so nothing but the tables metrics can move, and the shipped package
remains the reference.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from vifinqa.jsonl import load_jsonl
from vifinqa.scoring import gold_tables_for

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

from validate_submission import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="a built package directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--gold", choices=("binding", "full"), default="binding")
    args = parser.parse_args()

    gold = {r["id"]: gold_tables_for(r["annotation"], args.gold) for r in load_jsonl(args.benchmark)}
    rows = json.loads((args.source / "submission.json").read_text(encoding="utf-8"))
    replaced = 0
    for row in rows:
        if row["id"] in gold and gold[row["id"]]:
            row["relevant_tables"] = list(gold[row["id"]])
            replaced += 1

    package = args.output_dir / "package"
    if package.exists():
        shutil.rmtree(package)
    shutil.copytree(args.source, package)
    (package / "submission.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "gold": args.gold,
        "questions_replaced": replaced,
        "mean_tables_on_them": round(sum(len(gold[i]) for i in gold if gold[i]) / replaced, 2),
        "zip": str(zip_path),
    }, indent=2))


if __name__ == "__main__":
    main()
