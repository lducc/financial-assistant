"""E008b row retrieval with metadata-aware lexical query cleaning."""

from __future__ import annotations

import time

from .retrieval import RetrievedTable, Report, bm25, filter_reports, tokenize
from .row_retrieval import iter_row_documents


STOPWORDS = {
    "bao", "nhieu", "la", "cua", "cho", "trong", "nam", "vao", "den", "ngay",
    "cong", "ty", "vnd", "dong", "tyle", "phan", "tram", "trieu", "nghin",
}


def cleaned_query_tokens(question: str, metadata: dict) -> list[str]:
    remove = {str(year) for year in metadata.get("years", [])}
    remove.update(ticker.lower() for ticker in metadata.get("tickers", []))
    return [token for token in tokenize(question) if token not in STOPWORDS and token not in remove and not token.isdigit()]


def retrieve_rows_clean(question: str, metadata: dict, reports: list[Report], top_k: int = 10) -> dict:
    started = time.perf_counter()
    candidate_reports, stage = filter_reports(reports, metadata)
    rows = [row for report in candidate_reports for row in iter_row_documents(report)]
    query_tokens = cleaned_query_tokens(question, metadata)
    scores = bm25(query_tokens, rows)
    best_by_table: dict[str, tuple[float, dict]] = {}
    for score, row in zip(scores, rows):
        current = best_by_table.get(row["table_id"])
        if current is None or score > current[0] or (score == current[0] and row["row_index"] < current[1]["row_index"]):
            best_by_table[row["table_id"]] = (score, row)
    ranked = sorted(best_by_table.values(), key=lambda item: (-item[0], item[1]["table_id"]))[:top_k]
    tables = [
        {
            **RetrievedTable(
                table_id=row["table_id"], report_id=row["report_id"], page=row["page"],
                start_line=row["start_line"], score=round(score, 6), preview=" | ".join(row["cells"])[:500],
            ).__dict__,
            "row_index": row["row_index"], "row_cells": row["cells"],
        }
        for score, row in ranked
    ]
    return {
        "filter_stage": stage,
        "query_tokens": query_tokens,
        "candidate_report_count": len(candidate_reports),
        "candidate_row_count": len(rows),
        "candidate_table_count": len(best_by_table),
        "zero_score": bool(not ranked or ranked[0][0] == 0),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "tables": tables,
    }

