import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from docs import (
    load_companies, load_reports, make_row, parse_question, required_report_years,
    retrieve_docs, validate, write_package,
)
from vifinqa.answers import (
    EvidenceValue, answer_plan, bound_cells, first_numeric_cell, fold,
    header_years, source_scale, strict_number,
)
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import (
    header_cells, load_reports as load_table_reports, metric_query_tokens, report_tables,
    retrieve_rows, table_budget,
)
from vifinqa.tables import materialize
from validate_submission import validate as validate_submission


def resolve_table(table_id: str, report_id: str, reports_by_id: dict) -> dict | None:
    report = reports_by_id.get(report_id)
    if report is None:
        return None
    table = next((item for item in report_tables(str(report.path), report.identity)
                  if item.table_id == table_id), None)
    if table is None:
        return None
    headers = header_cells(table)
    row_index = next(
        (index for index in range(1, len(table.rows))
         if first_numeric_cell(list(table.rows[index]), headers) is not None),
        0,
    )
    return {
        "table_id": table.table_id,
        "report_id": table.report_id,
        "page": table.page,
        "start_line": table.start_line,
        "score": 0.0,
        "row_index": row_index,
        "row_cells": list(table.rows[row_index]),
        "rows": [list(row) for row in table.rows],
        "header_cells": headers,
        "title": table.title,
        "periods": list(table.periods),
        "unit": table.unit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--table-top-k", default="auto",
                        help="'auto', 'auto:n', an integer, or 'ranking' (use the ranking's own length)")
    parser.add_argument("--ranking", type=Path)
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--extra-tables", type=Path,
                        help="table_id to report_id map for tables the document gate never selected")
    args = parser.parse_args()

    data_root = args.data_root
    questions = load_jsonl(data_root / "questions" / "questions.jsonl")
    if args.limit:
        questions = questions[:args.limit]
    companies = load_companies(data_root / "code_stock.csv")
    reports = load_reports(data_root / "financial_statements")
    table_reports = load_table_reports(data_root)
    table_reports_by_id = {report.identity.report_id: report for report in table_reports}
    ranking = json.loads(args.ranking.read_text("utf-8")) if args.ranking else None
    extra_tables = json.loads(args.extra_tables.read_text("utf-8")) if args.extra_tables else {}
    rows = []

    for number, source in enumerate(questions, 1):
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
        top_k = table_budget(len(docs), args.table_top_k)
        if args.table_top_k == "ranking" and ranking:
            top_k = len(ranking.get(str(source["id"]), [])) or table_budget(len(docs), "auto")
        result = retrieve_rows(
            source["question"], metadata, table_reports,
            top_k=args.rerank_depth if ranking else top_k,
            report_ids=docs,
        )
        if ranking:
            ordered = ranking.get(str(source["id"]), [])
            order = {table_id: rank for rank, table_id in enumerate(ordered)}
            held = {table["table_id"] for table in result["tables"]}
            for table_id in ordered[:top_k]:
                if table_id not in held and table_id in extra_tables:
                    entry = resolve_table(table_id, extra_tables[table_id], table_reports_by_id)
                    if entry is not None:
                        result["tables"].append(entry)
            result["tables"] = sorted(
                result["tables"],
                key=lambda table: order.get(table["table_id"], len(order)),
            )[:top_k]
        for table in result["tables"]:
            if table["report_id"] not in docs:
                docs.append(table["report_id"])
        tables, evidence = [], []
        for rank, table in enumerate(result["tables"]):
            report = table_reports_by_id[table["report_id"]]
            stem = f"table_{source['id']}_{rank}"
            csv_path = args.output_dir / "package" / "data" / "tables" / f"{stem}.csv"
            materialize(report.path, table["start_line"], csv_path)
            tables.append(table["table_id"])
            evidence.append({"variable": f"df{rank}", "csv_path": f"data/tables/{stem}.csv"})
        row = make_row(source, docs, tables, evidence)
        years = required_report_years(parsed)
        tokens = {token for token in metric_query_tokens(source["question"], metadata) if len(token) > 2}
        values, lookup_scores = [], []
        for rank, table in enumerate(result["tables"]):
            for index, column in bound_cells(
                table["rows"], table.get("header_cells"), tokens, years,
            ):
                raw_value = strict_number(table["rows"][index][column])
                if raw_value is None:
                    continue
                header = table.get("header_cells", [])
                found = header_years(header[column]) & set(years) if column < len(header) else set()
                if not found:
                    found = header_years(table["report_id"]) & set(years)
                scale = source_scale(" ".join((table.get("unit", ""), *header)), table["rows"][index][column])
                values.append(EvidenceValue(
                    f"df{rank}", index, column, raw_value * scale, table["report_id"],
                    min(found) if found else None, scale,
                    " ".join(fold(cell) for cell in table["rows"][index][:2]),
                ))
                label = " ".join(fold(cell) for cell in table["rows"][index][:2])
                lookup_scores.append(
                    sum(1 for token in tokens if token in label)
                    + (0.5 if found else 0.0)
                )
        plan = answer_plan(source["question"], values)
        if plan is None and values:
            best = max(range(len(values)), key=lambda position: (lookup_scores[position], -values[position].column))
            plan = answer_plan(source["question"], [values[best]])
        if plan is not None:
            row["answer"], expression = plan
            row["pandas_query"] = f"result = {expression}"
        else:
            row["pandas_query"] = "result = 0 * df0.shape[0]"
        rows.append(row)
        if args.progress_every and number % args.progress_every == 0:
            print(f"processed {number}/{len(questions)}", flush=True)

    errors = validate(rows, questions, reports)
    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors[:20]))
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
