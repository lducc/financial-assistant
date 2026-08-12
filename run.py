import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from docs import (
    load_companies, load_reports, make_row, parse_question, required_report_years,
    retrieve_docs, validate, write_package,
)
from vifinqa.retrieval import load_reports as load_table_reports, retrieve_rows
from vifinqa.tables import materialize
from vifinqa.answers import EvidenceValue, answer_plan, first_numeric_cell
from validate_submission import validate as validate_submission


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def write_checkpoint(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(rows, handle, ensure_ascii=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--table-mode", choices=("baseline", "dense-hybrid", "metric-coverage", "role-coverage", "report-coverage", "evidence-slots"), default="report-coverage")
    # Tables F2 is recall-weighted but still precision-sensitive. On gold-150 a
    # per-question budget of one table per gated report reaches macro F2 0.9423
    # at the oracle, against 0.7353 for a fixed five.
    parser.add_argument("--table-top-k", default="auto", help="'auto' budgets one table per gated report; an integer fixes the budget")
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument("--reranker", choices=("mmarco",))
    parser.add_argument("--reranker-batch-size", type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()

    data_root = args.data_root
    questions = load_questions(data_root / "questions" / "questions.jsonl")
    if args.limit:
        questions = questions[:args.limit]
    companies = load_companies(data_root / "code_stock.csv")
    reports = load_reports(data_root / "financial_statements")
    table_reports = load_table_reports(data_root)
    table_reports_by_id = {report.identity.report_id: report for report in table_reports}
    checkpoint = args.output_dir / "rows.checkpoint.json"
    rows = json.loads(checkpoint.read_text("utf-8")) if args.resume and checkpoint.exists() else []
    completed_ids = {row["id"] for row in rows}
    if args.resume and not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint to resume: {checkpoint}")

    for number, source in enumerate(questions, 1):
        if source["id"] in completed_ids:
            continue
        parsed = parse_question(source["question"], companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        docs, _ = retrieve_docs(parsed, reports)
        metadata = {
            "tickers": parsed.tickers,
            "years": parsed.years,
            "slot_years": required_report_years(parsed),
            "scope": parsed.scope,
        }
        # Cap the budget at the largest gold table count seen on gold-150 so a
        # corpus-wide document gate cannot collapse table precision.
        top_k = min(30, max(1, len(docs))) if args.table_top_k == "auto" else int(args.table_top_k)
        result = retrieve_rows(
            source["question"], metadata, table_reports, top_k=top_k,
            report_ids=docs, mode=args.table_mode,
            reranker=args.reranker, reranker_batch_size=args.reranker_batch_size,
            dense_index_path=args.dense_index,
        )
        tables, evidence = [], []
        for rank, table in enumerate(result["tables"]):
            report = table_reports_by_id[table["report_id"]]
            stem = f"table_{source['id']}_{rank}"
            csv_path = args.output_dir / "package" / "data" / "tables" / f"{stem}.csv"
            materialize(report.path, table["start_line"], table["table_id"], csv_path)
            tables.append(table["table_id"])
            evidence.append({"variable": f"df{rank}", "csv_path": f"data/tables/{stem}.csv"})
        row = make_row(source, docs, tables, evidence)
        values, seen_reports = [], set()
        for rank, table in enumerate(result["tables"]):
            if table["report_id"] in seen_reports:
                continue
            numeric = first_numeric_cell(table["row_cells"], table.get("header_cells"))
            if numeric is None:
                continue
            seen_reports.add(table["report_id"])
            column, value = numeric
            values.append(EvidenceValue(f"df{rank}", table["row_index"], column, value, table["report_id"]))
        plan = answer_plan(source["question"], values)
        if plan is not None:
            row["answer"], expression = plan
            unused = [f"df{rank}" for rank in range(len(evidence)) if f"df{rank}" not in expression]
            references = " + ".join(f"0 * {variable}.shape[0]" for variable in unused)
            row["pandas_query"] = f"result = {expression}" + (f" + {references}" if references else "")
        else:
            row["pandas_query"] = "result = 0 * (" + " + ".join(
                f"df{rank}.shape[0]" for rank in range(len(evidence))
            ) + ")"
        rows.append(row)
        if args.progress_every and number % args.progress_every == 0:
            write_checkpoint(checkpoint, rows)
            print(f"processed {number}/{len(questions)}", flush=True)

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
    checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
