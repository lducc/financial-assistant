#!/usr/bin/env python3
"""Evaluate fixed-K ViFinQA table retrieval against internal annotations."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docs import (
    load_companies, load_reports as load_doc_reports, parse_question, required_report_years,
    retrieve_docs,
)
from vifinqa.retrieval import load_reports, retrieve_rows, table_budget


RANKS = (1, 3, 5, 10)
CANDIDATE_RANKS = (5, 10, 20, 50)


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def prefix_score(gold_tables: list[str], ranked_tables: list[str], k: int) -> dict:
    gold = set(gold_tables)
    ranked = unique(ranked_tables)[:k]
    predicted = set(ranked)
    hits = len(gold & predicted)
    precision = hits / k if k else 0.0
    recall = hits / len(gold) if gold else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f2": f2}


def reciprocal_rank(gold_tables: list[str], ranked_tables: list[str], k: int) -> float:
    gold = set(gold_tables)
    for rank, table_id in enumerate(unique(ranked_tables)[:k], 1):
        if table_id in gold:
            return 1 / rank
    return 0.0


def binary_ndcg(gold_tables: list[str], ranked_tables: list[str], k: int) -> float:
    gold = set(gold_tables)
    if not gold:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, table_id in enumerate(unique(ranked_tables)[:k], 1)
        if table_id in gold
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def coverage(gold_reports: list[str], selected_docs: list[str], ranked_tables: list[str], k: int) -> dict:
    gold = set(gold_reports)
    docs = set(selected_docs)
    table_reports = {table_id.partition("|")[0] for table_id in unique(ranked_tables)[:k]}
    return {
        "document_report_recall": len(gold & docs) / len(gold) if gold else 0.0,
        "all_gold_reports_in_docs": bool(gold) and gold <= docs,
        "ranked_report_recall": len(gold & table_reports) / len(gold) if gold else 0.0,
        "all_gold_reports_ranked": bool(gold) and gold <= table_reports,
    }


def gold_table_ranks(gold_tables: list[str], ranked_tables: list[str], depth: int) -> dict[str, int | None]:
    positions = {table_id: rank for rank, table_id in enumerate(unique(ranked_tables)[:depth], 1)}
    return {table_id: positions.get(table_id) for table_id in unique(gold_tables)}


def score_record(
    record: dict,
    ranked_tables: list[str],
    selected_docs: list[str],
    retrieval: dict | None = None,
    ranked_depth: int = max(CANDIDATE_RANKS),
    budget: int | None = None,
) -> dict:
    annotation = record["annotation"]
    gold_tables = annotation["gold_tables"]
    trace = {
        "id": record["id"],
        "question": record["question"],
        "gold_tables": gold_tables,
        "gold_reports": annotation["gold_reports"],
        "selected_docs": unique(selected_docs),
        "ranked_tables": unique(ranked_tables)[:ranked_depth],
        "operation": record.get("taxonomy", {}).get("operation", "unknown"),
        "table_count": record.get("taxonomy", {}).get("table_count", len(gold_tables)),
    }
    trace["prefix"] = {str(k): prefix_score(gold_tables, ranked_tables, k) for k in RANKS}
    # Fixed prefixes describe the ranking; the submitted budget describes the score.
    if budget is not None:
        trace["submitted"] = {"k": budget, **prefix_score(gold_tables, ranked_tables, budget)}
    trace["mrr"] = {str(k): reciprocal_rank(gold_tables, ranked_tables, k) for k in (5, 10)}
    trace["ndcg@5"] = binary_ndcg(gold_tables, ranked_tables, 5)
    trace["all_gold_covered@5"] = set(gold_tables) <= set(unique(ranked_tables)[:5])
    trace["gold_table_ranks"] = gold_table_ranks(gold_tables, ranked_tables, ranked_depth)
    trace["candidate_recall"] = {
        str(k): prefix_score(gold_tables, ranked_tables, k)["recall"] for k in CANDIDATE_RANKS
    }
    trace["coverage"] = coverage(annotation["gold_reports"], selected_docs, ranked_tables, 5)
    if retrieval is not None:
        trace["query_tokens"] = retrieval["query_tokens"]
        trace["context_query_tokens"] = retrieval["context_query_tokens"]
        trace["filter_stage"] = retrieval["filter_stage"]
        trace["candidate_report_count"] = retrieval["candidate_report_count"]
        trace["candidate_table_count"] = retrieval["candidate_table_count"]
        trace["candidate_row_count"] = retrieval["candidate_row_count"]
        trace["latency_ms"] = retrieval["latency_ms"]
    return trace


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize(traces: list[dict]) -> dict:
    prefix = {
        str(k): {
            name: mean([trace["prefix"][str(k)][name] for trace in traces])
            for name in ("precision", "recall", "f2")
        }
        for k in RANKS
    }
    submitted = [trace for trace in traces if "submitted" in trace]
    return {
        "records": len(traces),
        "prefix": prefix,
        **({"submitted": {
            "mean_k": mean([trace["submitted"]["k"] for trace in submitted]),
            **{name: mean([trace["submitted"][name] for trace in submitted]) for name in ("precision", "recall", "f2")},
        }} if submitted else {}),
        "mrr@5": mean([trace["mrr"]["5"] for trace in traces]),
        "mrr@10": mean([trace["mrr"]["10"] for trace in traces]),
        "ndcg@5": mean([trace["ndcg@5"] for trace in traces]),
        "all_gold_covered@5": mean([float(trace["all_gold_covered@5"]) for trace in traces]),
        "document_report_recall": mean([trace["coverage"]["document_report_recall"] for trace in traces]),
        "all_gold_reports_in_docs": mean([float(trace["coverage"]["all_gold_reports_in_docs"]) for trace in traces]),
        "ranked_report_recall@5": mean([trace["coverage"]["ranked_report_recall"] for trace in traces]),
        "all_gold_reports_ranked@5": mean([float(trace["coverage"]["all_gold_reports_ranked"]) for trace in traces]),
        "candidate_recall": {
            str(k): mean([trace["candidate_recall"][str(k)] for trace in traces])
            for k in CANDIDATE_RANKS
        },
    }


def grouped_summary(traces: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        groups[str(trace[field])].append(trace)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def connected_report_groups(records: list[dict]) -> list[tuple[str, ...]]:
    parent: dict[str, str] = {}

    def find(report: str) -> str:
        parent.setdefault(report, report)
        if parent[report] != report:
            parent[report] = find(parent[report])
        return parent[report]

    def join(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for record in records:
        reports = sorted(set(record["annotation"]["gold_reports"]))
        for report in reports:
            find(report)
        for report in reports[1:]:
            join(reports[0], report)
    groups: dict[str, list[str]] = defaultdict(list)
    for report in parent:
        groups[find(report)].append(report)
    return sorted(tuple(sorted(group)) for group in groups.values())


def split_records(records: list[dict], split: str) -> list[dict]:
    if split == "all":
        return records
    report_split = {}
    for group in connected_report_groups(records):
        key = "|".join(group)
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        report_split.update({report: "test" if int.from_bytes(digest, "big") % 5 == 0 else "dev" for report in group})
    return [
        record for record in records
        if report_split[record["annotation"]["gold_reports"][0]] == split
    ]


def production_trace(
    record: dict,
    table_reports: list,
    companies: dict,
    doc_reports: dict,
    ranked_depth: int,
    table_mode: str,
    reranker: str | None = None,
    table_top_k: str | int = "auto",
) -> dict:
    parsed = parse_question(record["question"], companies)
    if not parsed.tickers and parsed.candidate_tickers:
        parsed.tickers = parsed.candidate_tickers[:1]
    docs, _ = retrieve_docs(parsed, doc_reports)
    metadata = {
        "tickers": parsed.tickers,
        "years": parsed.years,
        "slot_years": required_report_years(parsed),
        "scope": parsed.scope,
    }
    # Retrieve to full depth for candidate-recall diagnostics. Coverage modes reserve
    # per report in gate order, so truncating to the budget matches what run.py emits.
    retrieval = retrieve_rows(
        record["question"], metadata, table_reports, top_k=ranked_depth,
        report_ids=docs, mode=table_mode, reranker=reranker,
    )
    return score_record(
        record, [table["table_id"] for table in retrieval["tables"]], docs, retrieval, ranked_depth,
        budget=table_budget(len(docs), table_top_k),
    )


def load_submission(path: Path) -> dict[int, dict]:
    if path.is_dir():
        path = path / "submission.json"
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            rows = json.loads(archive.read("submission.json").decode("utf-8"))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("submission must be a JSON list")
    return {int(row["id"]): row for row in rows}


def submission_traces(records: list[dict], submission: dict[int, dict], ranked_depth: int) -> list[dict]:
    traces = []
    for record in records:
        row = submission.get(int(record["id"]))
        if row is None:
            raise ValueError(f"submission missing id={record['id']}")
        traces.append(score_record(record, row.get("relevant_tables", []), row.get("relevant_docs", []), ranked_depth=ranked_depth))
    return traces


def write_results(output_dir: Path, name: str, traces: list[dict], split: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_traces.jsonl").write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces), encoding="utf-8"
    )
    summary = {
        "source": name,
        "split": split,
        **summarize(traces),
        "by_operation": grouped_summary(traces, "operation"),
        "by_table_count": grouped_summary(traces, "table_count"),
    }
    (output_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument(
        "--labels", "--pilot", dest="labels", type=Path,
        default=ROOT / "annotations" / "gold_150.jsonl",
        help="retrieval labels; not organizer answer ground truth",
    )
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--ranked-depth", type=int, default=max(CANDIDATE_RANKS))
    parser.add_argument("--table-mode", default="baseline")
    parser.add_argument("--table-top-k", default="auto", help="'auto' budgets one table per gated report; an integer fixes the budget")
    parser.add_argument("--reranker", choices=("mmarco",))
    parser.add_argument(
        "--experimental-mode",
        action="store_true",
        help="allow archived research-only table modes in this evaluator",
    )
    parser.add_argument("--submission", type=Path, help="submission.json or its containing package directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "table_retrieval_eval")
    args = parser.parse_args()
    production_modes = {"baseline", "role-coverage"}
    experimental_modes = {"metric-focused", "metric-coverage", "field-aware", "field-coverage", "rank-fusion", "report-coverage", "evidence-slots"}
    if args.table_mode not in production_modes | experimental_modes:
        raise ValueError(f"unknown table mode: {args.table_mode}")
    if args.table_mode in experimental_modes and not args.experimental_mode:
        raise ValueError(
            f"{args.table_mode} is research-only; pass --experimental-mode to evaluate it"
        )
    if args.ranked_depth < max(CANDIDATE_RANKS):
        raise ValueError(f"--ranked-depth must be at least {max(CANDIDATE_RANKS)}")

    records = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = split_records(records, args.split)
    if not records:
        raise ValueError(f"no records in split={args.split}")
    table_reports = load_reports(args.dataset_root)
    companies = load_companies(args.dataset_root / "code_stock.csv")
    doc_reports = load_doc_reports(args.dataset_root / "financial_statements")
    summaries = [write_results(
        args.output_dir,
        "production",
        [
            production_trace(
                record, table_reports, companies, doc_reports, args.ranked_depth, args.table_mode,
                args.reranker, args.table_top_k,
            )
            for record in records
        ],
        args.split,
    )]
    if args.submission:
        summaries.append(write_results(
            args.output_dir, "submission", submission_traces(records, load_submission(args.submission), args.ranked_depth), args.split
        ))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
