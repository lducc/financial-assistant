"""Write one passage per corpus table for the dense index."""

import argparse
import json
from pathlib import Path
import sys

from vifinqa.retrieval import carries_figures, load_reports, report_tables

ROOT = Path(__file__).resolve().parents[1]
LABELS = 900  # characters of line-item labels
SCOPE = {"consolidated": "hợp nhất", "separate": "công ty mẹ riêng lẻ"}


def passage(table, identity, companies: dict) -> str:
    labels = dict.fromkeys(
        row[0].strip() for row in table.rows if row and row[0].strip()
    )
    listed = "; ".join(labels)[:LABELS]
    name = companies.get(identity.ticker, "")
    where = (f"{name} ({identity.ticker}) năm {identity.year} "
             f"báo cáo {SCOPE.get(identity.scope, identity.scope)}").strip()
    headers = " | ".join(" | ".join(row) for row in table.headers)
    parts = (where, table.title, f"Các chỉ tiêu: {listed}" if listed else "",
             headers, " | ".join(table.periods), table.unit)
    return "\n".join(part for part in dict.fromkeys(p.strip() for p in parts) if part)


def load_company_names(path: Path) -> dict:
    import csv
    names = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].strip():
                names[row[0].strip()] = row[1].strip()
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/vifinqa")
    parser.add_argument("--output", type=Path, default=ROOT / "output/dense/tables.jsonl")
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    companies = load_company_names(args.data_root / "code_stock.csv")
    reports = load_reports(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    with args.output.open("w", encoding="utf-8") as handle:
        for number, report in enumerate(reports, 1):
            try:
                tables = report_tables(str(report.path), report.identity)
            except Exception:
                continue
            for table in tables:
                if not carries_figures(table):
                    skipped += 1
                    continue
                handle.write(json.dumps({
                    "report_id": table.report_id,
                    "table_id": table.table_id,
                    "text": passage(table, report.identity, companies),
                }, ensure_ascii=False) + "\n")
                written += 1
            if number % args.progress_every == 0:
                print(f"  {number}/{len(reports)} reports, {written} tables",
                      file=sys.stderr, flush=True)
    print(f"{written} passages written, {skipped} tables carried no figures")
    print(args.output)


if __name__ == "__main__":
    main()
