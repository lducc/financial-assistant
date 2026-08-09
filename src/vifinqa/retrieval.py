"""Metadata-gated BM25 retrieval over OCR table rows."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
import statistics
import time
import unicodedata

from .tables import PAGE_RE, TABLE_RE, ReportIdentity, iter_report_paths, parse_report_identity, parse_table_rows


TOKEN_RE = re.compile(r"[a-z0-9%]{2,}")
STOPWORDS = {
    "bao", "nhieu", "la", "cua", "cho", "trong", "nam", "vao", "den", "ngay",
    "cong", "ty", "vnd", "dong", "tyle", "phan", "tram", "trieu", "nghin",
}
CONTEXT_STOPWORDS = STOPWORDS - {"vnd", "dong", "tyle", "phan", "tram", "trieu", "nghin"}
ROLE_STOPWORDS = {
    "cac", "co", "ghi", "nhan", "so", "gia", "tri", "nao", "cao", "thap", "lon", "nho",
    "hon", "nhat", "me", "tap", "doan", "ctcp", "ma", "tinh", "va",
}
PERIOD_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")
TITLE_RE = re.compile(r"\b(?:bao cao|bang|thuyet minh)\b")
UNIT_PHRASES = ("vnd", "don vi", "trieu dong", "nghin dong", "ty dong", "million", "billion")
HEADER_TOKENS = {"ma", "so", "thuyet", "minh", "chi", "tieu", "don", "vi"}
RRF_OFFSET = 60


@dataclass(frozen=True)
class Report:
    identity: ReportIdentity
    path: Path


@dataclass(frozen=True)
class Table:
    table_id: str
    report_id: str
    page: int | None
    start_line: int
    rows: tuple[tuple[str, ...], ...]
    title: str
    context: tuple[str, ...]
    headers: tuple[tuple[str, ...], ...]
    periods: tuple[str, ...]
    unit: str


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")
    return TOKEN_RE.findall(text)


def load_reports(dataset_root: Path) -> list[Report]:
    return [Report(parse_report_identity(path, dataset_root), path) for path in iter_report_paths(dataset_root)]


def filter_reports(reports: list[Report], metadata: dict) -> tuple[list[Report], str]:
    tickers, years, scope = set(metadata.get("tickers", [])), set(metadata.get("years", [])), metadata.get("scope")
    stages = (
        ("ticker_year_scope", lambda report: (not tickers or report.identity.ticker in tickers) and (not years or report.identity.year in years) and (scope is None or report.identity.scope == scope)),
        ("ticker_year", lambda report: (not tickers or report.identity.ticker in tickers) and (not years or report.identity.year in years)),
        ("ticker", lambda report: not tickers or report.identity.ticker in tickers),
        ("year", lambda report: not years or report.identity.year in years),
        ("global", lambda report: True),
    )
    for stage, matches in stages:
        candidates = [report for report in reports if matches(report)]
        if candidates:
            return candidates, stage
    return [], "empty"


def table_context(lines: list[str]) -> tuple[str, tuple[str, ...]]:
    lines = [line for line in lines if line and not PAGE_RE.fullmatch(line)]
    title_lines = [line for line in lines if TITLE_RE.search(" ".join(tokenize(line)))]
    return (title_lines[-1] if title_lines else lines[-1] if lines else ""), tuple(lines[-8:])


def table_metadata(title: str, context: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...], str]:
    headers = tuple(
        row for row in rows[:3]
        if PERIOD_RE.search(" ".join(row)) or HEADER_TOKENS & set(tokenize(" ".join(row)))
    ) or rows[:1]
    source = " ".join((title, *context, *(" ".join(row) for row in headers)))
    periods = tuple(dict.fromkeys(PERIOD_RE.findall(source)))
    unit = " ".join(
        line for line in (title, *context, *(" ".join(row) for row in headers))
        if "%" in line or any(phrase in " ".join(tokenize(line)) for phrase in UNIT_PHRASES)
    )
    return headers, periods, unit


# Corpus contains more than 256 reports; retain parsed tables for one full run.
@lru_cache(maxsize=4096)
def report_tables(path_text: str, identity: ReportIdentity) -> tuple[Table, ...]:
    text = Path(path_text).read_text(encoding="utf-8")
    pages = list(PAGE_RE.finditer(text))
    page_index, page, cursor, line, context_cursor = 0, None, 0, 1, 0
    context_lines: list[str] = []
    tables = []
    for match in TABLE_RE.finditer(text):
        context_lines.extend(" ".join(value.split()) for value in text[context_cursor:match.start()].splitlines())
        context_lines = context_lines[-16:]
        line += text.count("\n", cursor, match.start())
        cursor = match.start()
        while page_index < len(pages) and pages[page_index].start() < match.start():
            page = int(pages[page_index].group(1))
            page_index += 1
        rows = tuple(tuple(row) for row in parse_table_rows(match.group(0)))
        title, context = table_context(context_lines)
        headers, periods, unit = table_metadata(title, context, rows)
        tables.append(Table(
            f"{identity.report_id}|{line}", identity.report_id, page, line, rows,
            title, context, headers, periods, unit,
        ))
        context_cursor = match.end()
    return tuple(tables)


def query_tokens(question: str, metadata: dict) -> list[str]:
    blocked = {str(year) for year in metadata.get("years", [])}
    blocked.update(ticker.lower() for ticker in metadata.get("tickers", []))
    return [token for token in tokenize(question) if token not in STOPWORDS and token not in blocked and not token.isdigit()]


def context_query_tokens(question: str, metadata: dict) -> list[str]:
    blocked = {ticker.lower() for ticker in metadata.get("tickers", [])}
    return [token for token in tokenize(question.replace("%", " percent ")) if token not in CONTEXT_STOPWORDS and token not in blocked]


def contextual_prefix(table: Table) -> str:
    return " ".join((
        table.title,
        *table.context,
        *(" ".join(header) for header in table.headers),
        " ".join(table.periods),
        table.unit,
    )).replace("%", " percent ")


def contextual_row(table: Table, row: str) -> str:
    return f"{contextual_prefix(table)} {row}"


def bm25(query: list[str], rows: list[str]) -> list[float]:
    if not query or not rows:
        return [0.0] * len(rows)
    terms = set(query)
    tokens = [tokenize(row) for row in rows]
    frequencies = [Counter(row) for row in tokens]
    document_frequency = Counter(term for row in tokens for term in set(row) if term in terms)
    average_length = statistics.fmean(max(1, len(row)) for row in tokens)
    scores = []
    for row, counts in zip(tokens, frequencies):
        score, length = 0.0, max(1, len(row))
        for term in terms:
            frequency = counts[term]
            if frequency:
                idf = math.log(1 + (len(rows) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        scores.append(score)
    return scores


def phrase_bonus(query: list[str], row: str) -> float:
    tokens = tokenize(row)
    bonus = 0.0
    for size, weight in ((3, 1.4), (2, 0.35)):
        query_phrases = {tuple(query[index:index + size]) for index in range(len(query) - size + 1)}
        row_phrases = {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}
        bonus += len(query_phrases & row_phrases) * weight
    return bonus


def ranked_tables(best: dict[str, tuple[float, Table, int]]) -> list[tuple[float, Table, int]]:
    return sorted(best.values(), key=lambda item: (-item[0], item[1].table_id))


def role_rankings(
    question: str,
    metadata: dict,
    candidates: list[Report],
    rows: list[tuple[Table, int, str]],
) -> dict[int, list[tuple[float, Table, int]]]:
    years = sorted(set(metadata.get("years", [])))
    if len(years) < 2:
        return {}
    year_table_ids = {
        year: {
            table.table_id
            for report in candidates
            for table in report_tables(str(report.path), report.identity)
            if report.identity.year == year or str(year) in table.periods
        }
        for year in years
    }
    all_years = {str(year) for year in years}
    result = {}
    for year, table_ids in year_table_ids.items():
        role_rows = [(table, row_index, row) for table, row_index, row in rows if table.table_id in table_ids]
        query = [
            token for token in context_query_tokens(question, metadata)
            if (token not in all_years or token == str(year))
            and token not in ROLE_STOPWORDS
            and (not token.isdigit() or token == str(year))
        ]
        scores = bm25(query, [contextual_row(table, row) for table, _, row in role_rows])
        best = {}
        for (table, row_index, _), score in zip(role_rows, scores):
            previous = best.get(table.table_id)
            if score > 0 and (previous is None or score > previous[0] or score == previous[0] and row_index < previous[2]):
                best[table.table_id] = score, table, row_index
        if best:
            result[year] = ranked_tables(best)
    return result


def retrieve_rows(
    question: str,
    metadata: dict,
    reports: list[Report],
    top_k: int = 5,
    report_ids: list[str] | None = None,
    mode: str = "baseline",
) -> dict:
    if mode not in {"baseline", "role-coverage"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    started = time.perf_counter()
    if report_ids is None:
        candidates, stage = filter_reports(reports, metadata)
    else:
        reports_by_id = {report.identity.report_id: report for report in reports}
        candidates = [reports_by_id[report_id] for report_id in report_ids if report_id in reports_by_id]
        stage = "report_ids"
    tables = [table for report in candidates for table in report_tables(str(report.path), report.identity)]
    query = query_tokens(question, metadata)
    context_query = context_query_tokens(question, metadata)
    rows = [(table, index, " ".join(row)) for table in tables for index, row in enumerate(table.rows)]
    scores = bm25(query, [row for _, _, row in rows])
    context_scores = bm25(
        context_query,
        [contextual_row(table, row) for table, _, row in rows],
    )
    raw_best: dict[str, tuple[float, Table, int]] = {}
    context_best: dict[str, tuple[float, Table, int]] = {}
    for (table, row_index, row), raw_score, context_score in zip(rows, scores, context_scores):
        raw_score += phrase_bonus(query, row)
        for score, best in ((raw_score, raw_best), (context_score, context_best)):
            previous = best.get(table.table_id)
            if previous is None or score > previous[0] or score == previous[0] and row_index < previous[2]:
                best[table.table_id] = score, table, row_index
    raw_ranked = ranked_tables(raw_best)
    context_ranked = ranked_tables(context_best)
    raw_positive = [item for item in raw_ranked if item[0] > 0]
    context_positive = [item for item in context_ranked if item[0] > 0]
    if context_positive:
        raw_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(raw_positive, 1)}
        context_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(context_positive, 1)}
        selected = {
            table_id: raw_best[table_id] if table_id in raw_ranks else context_best[table_id]
            for table_id in raw_ranks.keys() | context_ranks.keys()
        }
        ranked = [
            (
                (1 / (RRF_OFFSET + raw_ranks[table_id]) if table_id in raw_ranks else 0)
                + (1 / (RRF_OFFSET + context_ranks[table_id]) if table_id in context_ranks else 0),
                selected[table_id][1],
                selected[table_id][2],
            )
            for table_id in raw_ranks.keys() | context_ranks.keys()
        ]
        ranked.sort(key=lambda item: (-item[0], item[1].table_id))
        selected_ids = {table.table_id for _, table, _ in ranked}
        ranked.extend(item for item in raw_ranked if item[1].table_id not in selected_ids)
    else:
        ranked = raw_ranked
    baseline_rank = {table.table_id: rank for rank, (_, table, _) in enumerate(ranked, 1)}
    role_ranks = {}
    if mode == "role-coverage":
        for year, role_ranked in role_rankings(question, metadata, candidates, rows).items():
            for rank, (_, table, _) in enumerate(role_ranked, 1):
                role_ranks.setdefault(table.table_id, {})[year] = rank
        if role_ranks:
            fused = []
            for score, table, row_index in ranked:
                role_score = sum(1 / (RRF_OFFSET + rank) for rank in role_ranks.get(table.table_id, {}).values())
                fused.append((score + role_score, table, row_index))
            ranked = sorted(fused, key=lambda item: (-item[0], item[1].table_id))
    ranked = ranked[:top_k]
    return {
        "filter_stage": stage,
        "query_tokens": query,
        "context_query_tokens": context_query,
        "candidate_report_count": len(candidates),
        "candidate_table_count": len(tables),
        "candidate_row_count": len(rows),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "tables": [
            {
                "table_id": table.table_id,
                "report_id": table.report_id,
                "page": table.page,
                "start_line": table.start_line,
                "score": round(score, 6),
                "row_index": row_index,
                "row_cells": list(table.rows[row_index]),
                "title": table.title,
                "periods": list(table.periods),
                "unit": table.unit,
                "pre_role_rank": baseline_rank.get(table.table_id),
                "role_ranks": role_ranks.get(table.table_id, {}),
            }
            for score, table, row_index in ranked
        ],
    }
