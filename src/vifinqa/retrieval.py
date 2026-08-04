"""Deterministic metadata-filtered BM25 retrieval over literal OCR tables."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
import statistics
import time
import unicodedata

from .catalog import PAGE_RE, TABLE_RE, ReportIdentity, iter_report_paths, parse_report_identity, parse_table_rows


TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


@dataclass(frozen=True)
class Report:
    identity: ReportIdentity
    path: Path


@dataclass(frozen=True)
class RetrievedTable:
    table_id: str
    report_id: str
    page: int | None
    start_line: int
    score: float
    preview: str


def tokenize(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(char for char in decomposed if unicodedata.category(char) != "Mn").replace("đ", "d")
    return TOKEN_RE.findall(normalized)


def load_reports(dataset_root: Path) -> list[Report]:
    return [Report(parse_report_identity(path, dataset_root), path) for path in iter_report_paths(dataset_root)]


def filter_reports(reports: list[Report], metadata: dict) -> tuple[list[Report], str]:
    tickers = set(metadata.get("tickers", []))
    years = set(metadata.get("years", []))
    scope = metadata.get("scope")
    stages = [
        ("ticker_year_scope", lambda report: (not tickers or report.identity.ticker in tickers) and (not years or report.identity.year in years) and (scope is None or report.identity.scope == scope)),
        ("ticker_year", lambda report: (not tickers or report.identity.ticker in tickers) and (not years or report.identity.year in years)),
        ("ticker", lambda report: not tickers or report.identity.ticker in tickers),
        ("year", lambda report: not years or report.identity.year in years),
        ("global", lambda report: True),
    ]
    for stage, predicate in stages:
        candidates = [report for report in reports if predicate(report)]
        if candidates:
            return candidates, stage
    return [], "empty"


def iter_table_documents(report: Report):
    text = report.path.read_text(encoding="utf-8")
    pages = list(PAGE_RE.finditer(text))
    page_index, page = 0, None
    current_line, cursor = 1, 0
    for ordinal, match in enumerate(TABLE_RE.finditer(text), start=1):
        current_line += text.count("\n", cursor, match.start())
        cursor = match.start()
        while page_index < len(pages) and pages[page_index].start() < match.start():
            page = int(pages[page_index].group(1))
            page_index += 1
        rows = parse_table_rows(match.group(0))
        body = " ".join(cell for row in rows for cell in row)
        preview = " | ".join(" | ".join(row) for row in rows[:3])[:500]
        yield {
            "table_id": f"{report.identity.report_id}|{current_line}",
            "report_id": report.identity.report_id,
            "page": page,
            "start_line": current_line,
            "tokens": tokenize(body),
            "preview": preview,
            "ordinal": ordinal,
        }


def bm25(query_tokens: list[str], documents: list[dict], k1: float = 1.2, b: float = 0.75) -> list[float]:
    if not documents:
        return []
    query_terms = set(query_tokens)
    doc_freq = Counter(term for doc in documents for term in set(doc["tokens"]) if term in query_terms)
    avg_length = statistics.fmean(max(1, len(doc["tokens"])) for doc in documents)
    scores = []
    for doc in documents:
        frequencies = Counter(doc["tokens"])
        length = max(1, len(doc["tokens"]))
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            idf = math.log(1 + (len(documents) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / avg_length)
            score += idf * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def retrieve(question: str, metadata: dict, reports: list[Report], top_k: int = 10) -> dict:
    started = time.perf_counter()
    candidate_reports, stage = filter_reports(reports, metadata)
    documents = [document for report in candidate_reports for document in iter_table_documents(report)]
    scores = bm25(tokenize(question), documents)
    ranked = sorted(zip(scores, documents), key=lambda item: (-item[0], item[1]["table_id"]))[:top_k]
    tables = [
        RetrievedTable(
            table_id=document["table_id"], report_id=document["report_id"], page=document["page"],
            start_line=document["start_line"], score=round(score, 6), preview=document["preview"],
        ).__dict__
        for score, document in ranked
    ]
    return {
        "filter_stage": stage,
        "candidate_report_count": len(candidate_reports),
        "candidate_table_count": len(documents),
        "zero_score": bool(not ranked or ranked[0][0] == 0),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "tables": tables,
    }

