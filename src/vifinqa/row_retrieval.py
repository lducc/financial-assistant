"""Row-centric BM25 retrieval with table-level aggregation."""

from __future__ import annotations

from collections import defaultdict
import time

from .catalog import PAGE_RE, TABLE_RE, parse_table_rows
from .retrieval import RetrievedTable, Report, bm25, filter_reports, tokenize


def iter_row_documents(report: Report):
    text = report.path.read_text(encoding="utf-8")
    pages = list(PAGE_RE.finditer(text))
    page_index, page = 0, None
    current_line, cursor = 1, 0
    for match in TABLE_RE.finditer(text):
        current_line += text.count("\n", cursor, match.start())
        cursor = match.start()
        while page_index < len(pages) and pages[page_index].start() < match.start():
            page = int(pages[page_index].group(1))
            page_index += 1
        table_id = f"{report.identity.report_id}|{current_line}"
        for row_index, cells in enumerate(parse_table_rows(match.group(0))):
            text_row = " ".join(cells)
            yield {
                "table_id": table_id,
                "report_id": report.identity.report_id,
                "page": page,
                "start_line": current_line,
                "row_index": row_index,
                "cells": cells,
                "tokens": tokenize(text_row),
            }


def retrieve_rows(question: str, metadata: dict, reports: list[Report], top_k: int = 10) -> dict:
    started = time.perf_counter()
    candidate_reports, stage = filter_reports(reports, metadata)
    rows = [row for report in candidate_reports for row in iter_row_documents(report)]
    scores = bm25(tokenize(question), rows)
    best_by_table: dict[str, tuple[float, dict]] = {}
    for score, row in zip(scores, rows):
        existing = best_by_table.get(row["table_id"])
        if existing is None or score > existing[0] or (score == existing[0] and row["row_index"] < existing[1]["row_index"]):
            best_by_table[row["table_id"]] = (score, row)
    ranked = sorted(best_by_table.values(), key=lambda item: (-item[0], item[1]["table_id"]))[:top_k]
    tables = [
        {
            **RetrievedTable(
                table_id=row["table_id"], report_id=row["report_id"], page=row["page"],
                start_line=row["start_line"], score=round(score, 6), preview=" | ".join(row["cells"])[:500],
            ).__dict__,
            "row_index": row["row_index"],
            "row_cells": row["cells"],
        }
        for score, row in ranked
    ]
    return {
        "filter_stage": stage,
        "candidate_report_count": len(candidate_reports),
        "candidate_row_count": len(rows),
        "candidate_table_count": len(best_by_table),
        "zero_score": bool(not ranked or ranked[0][0] == 0),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "tables": tables,
    }

