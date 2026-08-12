#!/usr/bin/env python3
"""Generate deterministic, exact-ID v2 retrieval predictions and traces."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docs import load_companies, load_reports as load_document_reports, parse_question, required_report_years, retrieve_docs
from vifinqa.evaluation_v2 import index_records, load_jsonl
from vifinqa.retrieval import load_reports, retrieve_rows


PROMOTION_MODES = ("baseline", "metric-focused", "metric-coverage", "role-coverage", "evidence-slots")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mode", choices=PROMOTION_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="development")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prediction run: {args.output_dir}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample = index_records(manifest.get("splits", {}).get(args.split, []), "manifest split")
    queue = index_records(load_jsonl(args.queue), "queue")
    if set(queue) != set(sample):
        raise ValueError("queue and manifest split IDs must match exactly")
    companies = load_companies(args.raw_root / "code_stock.csv")
    document_reports = load_document_reports(args.raw_root / "financial_statements")
    table_reports = load_reports(args.raw_root)
    predictions, traces = [], []
    for identifier in sorted(queue):
        question = queue[identifier]["question"]
        parsed = parse_question(question, companies)
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
        documents, _ = retrieve_docs(parsed, document_reports)
        result = retrieve_rows(question, {
            "tickers": parsed.tickers, "years": parsed.years,
            "slot_years": required_report_years(parsed), "scope": parsed.scope,
        }, table_reports, top_k=5, report_ids=documents, mode=args.mode)
        ranked_tables = [table["table_id"] for table in result["tables"]]
        source_valid = all(table.partition("|")[0] in documents for table in ranked_tables)
        trace = {
            "id": identifier, "ranked_tables": ranked_tables,
            "latency_ms": result["latency_ms"], "candidate_count": result["candidate_table_count"],
            "fallback": bool(result.get("experimental_fallback") or result.get("reranker_fallback")),
            "source_binding_valid": source_valid, "top_five_valid": len(ranked_tables) == 5 and len(ranked_tables) == len(set(ranked_tables)),
            "document_ids": documents, "mode": args.mode,
        }
        predictions.append({key: trace[key] for key in ("id", "ranked_tables", "latency_ms", "candidate_count")})
        traces.append(trace)
    args.output_dir.mkdir(parents=True)
    for name, records in (("predictions.jsonl", predictions), ("traces.jsonl", traces)):
        (args.output_dir / name).write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8",
        )
    (args.output_dir / "run.json").write_text(json.dumps({
        "mode": args.mode, "split": args.split, "records": len(predictions),
        "queue": str(args.queue), "manifest": str(args.manifest), "raw_root": str(args.raw_root),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
