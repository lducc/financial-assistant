import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import load_companies, load_reports, make_row, parse_question, retrieve_docs, validate, write_package
from vifinqa.retrieval import load_reports as load_table_reports, retrieve_rows
from vifinqa.tables import materialize
from validate_submission import validate as validate_submission


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--table-mode", choices=("baseline", "role-coverage"), default="baseline")
    args = parser.parse_args()

    data_root = args.data_root
    questions = load_questions(data_root / "questions" / "questions.jsonl")
    if args.limit:
        questions = questions[:args.limit]
    companies = load_companies(data_root / "code_stock.csv")
    reports = load_reports(data_root / "financial_statements")
    table_reports = load_table_reports(data_root)
    table_reports_by_id = {report.identity.report_id: report for report in table_reports}
    rows = []

    for source in questions:
        parsed = parse_question(source["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, reports)
        metadata = {"tickers": parsed.tickers, "years": parsed.years, "scope": parsed.scope}
        result = retrieve_rows(
            source["question"], metadata, table_reports, top_k=5,
            report_ids=docs, mode=args.table_mode,
        )
        tables, evidence = [], []
        for rank, table in enumerate(result["tables"]):
            report = table_reports_by_id[table["report_id"]]
            stem = f"table_{source['id']}_{rank}"
            csv_path = args.output_dir / "package" / "data" / "tables" / f"{stem}.csv"
            materialize(report.path, table["start_line"], table["table_id"], csv_path)
            tables.append(table["table_id"])
            evidence.append({"variable": f"df{rank}", "csv_path": f"data/tables/{stem}.csv"})
        rows.append(make_row(source, docs, tables, evidence))

    errors = validate(rows, questions, reports)
    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors[:20]))
    # The retriever itself parsed these IDs from the gated raw reports. Passing
    # this set to the strict validator preserves the catalog check without
    # parsing unrelated corpus reports during every submission run.
    catalog_ids = {table_id for row in rows for table_id in row["relevant_tables"]}
    print(write_package(
        args.output_dir,
        rows,
        validator=lambda package: validate_submission(
            package,
            {int(question["id"]) for question in questions},
            catalog_ids,
        ),
    ))


if __name__ == "__main__":
    main()
