#!/usr/bin/env python3
"""Add interchangeable tables to benchmark labels.

A figure usually appears more than once in a report: the balance sheet carries
the total, a note repeats it in its breakdown, the cash-flow statement restates
it. Our annotation recorded one canonical table per bound value, while the
organizers count every table that carries it — which is why local recall reads
about 21 points above the live score.

The addition is deliberately conservative: same report, same exact raw cell
string, and a row label sharing content words with the bound row. Matching on the
number alone would sweep in coincidences (account codes, years, repeated small
integers), so a label match is required as well.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.answers import parse_ocr_number
from vifinqa.evaluation_v2 import sha256_text
from vifinqa.retrieval import load_reports, report_tables

# Words that carry no discriminating power in a row label.
LABEL_STOPWORDS = {
    "cac", "khoan", "va", "cua", "trong", "cho", "tai", "theo", "so", "tong", "cong",
    "gia", "tri", "khac", "nam", "cuoi", "dau", "ngan", "dai", "han", "phai", "thu", "tu",
}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")


def label_tokens(label: str) -> set[str]:
    return {token for token in fold(label).replace(".", " ").split() if token.isalpha() and len(token) > 2} - LABEL_STOPWORDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--min-shared-label-tokens", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = [
        json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    reports = {report.identity.report_id: report for report in load_reports(args.dataset_root)}

    added_counter = Counter()
    for record in records:
        annotation = record["annotation"]
        existing = set(annotation["gold_tables"])
        additions: set[str] = set()
        for binding in annotation["row_column_bindings"]:
            raw = binding["raw"].strip()
            value = parse_ocr_number(raw)
            # A bare code or a small integer repeats everywhere; require a real figure.
            if not raw or value is None or abs(value) < 1000:
                continue
            wanted = label_tokens(binding["row_label"])
            if len(wanted) < args.min_shared_label_tokens:
                continue
            report_id = binding["table"].partition("|")[0]
            report = reports.get(report_id)
            if report is None:
                continue
            for table in report_tables(str(report.path), report.identity):
                if table.table_id in existing or table.table_id in additions:
                    continue
                for row in table.rows:
                    if not any(cell.strip() == raw for cell in row):
                        continue
                    shared = wanted & {token for cell in row for token in label_tokens(cell)}
                    if len(shared) >= args.min_shared_label_tokens:
                        additions.add(table.table_id)
                        break
        if additions:
            added_counter[record["id"]] = len(additions)
            annotation["gold_tables"] = sorted(existing | additions)
            annotation["gold_reports"] = sorted({t.partition("|")[0] for t in annotation["gold_tables"]})
            record["taxonomy"]["table_count"] = len(annotation["gold_tables"])
        report_tables.cache_clear()

    total_before = sum(len(r["annotation"]["gold_tables"]) - added_counter.get(r["id"], 0) for r in records)
    total_after = sum(len(r["annotation"]["gold_tables"]) for r in records)
    summary = {
        "records": len(records),
        "records_extended": len(added_counter),
        "gold_tables_before": total_before,
        "gold_tables_after": total_after,
        "mean_before": round(total_before / len(records), 3),
        "mean_after": round(total_after / len(records), 3),
        "largest_additions": added_counter.most_common(8),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        args.output.write_text(text, encoding="utf-8")
        # The manifest pins the file hash, so it has to move with the labels or
        # verify_benchmark reports drift.
        manifest_path = args.output.parent / "benchmark_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["gold_tables"] = total_after
            manifest["table_count_distribution"] = dict(sorted(Counter(
                len(record["annotation"]["gold_tables"]) for record in records
            ).items()))
            manifest["interchangeable_tables_added"] = total_after - total_before
            manifest["benchmark_sha256"] = sha256_text(text)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
