from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
import statistics
import unicodedata

from .tables import PAGE_RE, TABLE_RE, ReportIdentity, iter_report_paths, parse_report_identity, parse_table_rows


TOKEN_RE = re.compile(r"[a-z0-9%]{2,}")
UNICODE_TOKEN_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)
STOPWORDS = {
    "bao", "nhieu", "la", "cua", "cho", "trong", "nam", "vao", "den", "ngay",
    "cong", "ty", "vnd", "dong", "tyle", "phan", "tram", "trieu", "nghin",
}
CONTEXT_STOPWORDS = STOPWORDS - {"vnd", "dong", "tyle", "phan", "tram", "trieu", "nghin"}
METRIC_STOPWORDS = CONTEXT_STOPWORDS | {
    "tinh", "hay", "biet", "gia", "tri", "muc", "do", "so", "voi", "giua",
    "tu", "den", "vao", "tai", "cuoi", "dau", "ky", "theo", "tren",
    "duoi", "tang", "giam", "truong", "ty", "le", "phan", "tram",
    "chenh", "lech", "binh", "quan", "lon", "nho", "cao", "thap",
}
PERIOD_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")
# at least two digits, so account codes and list numbering are not figures
NUMERIC_CELL_RE = re.compile(r"\d[\d.,]*\d")
TITLE_RE = re.compile(r"\b(?:bao cao|bang|thuyet minh)\b")
UNIT_PHRASES = ("vnd", "don vi", "trieu dong", "nghin dong", "ty dong", "million", "billion")
HEADER_TOKENS = {"ma", "so", "thuyet", "minh", "chi", "tieu", "don", "vi"}
RRF_OFFSET = 60
# questions cite few line items; counting more rows would reward long tables
SUPPORTING_ROWS = 3


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


def unicode_tokenize(text: str) -> list[str]:
    return UNICODE_TOKEN_RE.findall(unicodedata.normalize("NFC", text.lower()))


def load_reports(dataset_root: Path) -> list[Report]:
    return [Report(parse_report_identity(path, dataset_root), path) for path in iter_report_paths(dataset_root)]


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


# the corpus has more than 256 reports; keep parsed tables for a full run
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


def metric_query_tokens(question: str, metadata: dict) -> list[str]:
    blocked = {ticker.lower() for ticker in metadata.get("tickers", [])}
    blocked.update(str(year) for year in metadata.get("years", []))
    return [
        token for token in tokenize(question.replace("%", " percent "))
        if token not in METRIC_STOPWORDS and token not in blocked and not token.isdigit()
    ]


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


def score_bm25(terms: set[str], tokens: list[list[str]]) -> list[float]:
    ordered_terms = sorted(terms)
    document_frequency = Counter(
        term for row in tokens for term in set(row) if term in terms
    )
    average_length = statistics.fmean(max(1, len(row)) for row in tokens)
    scores = []
    for row in tokens:
        counts = Counter(row)
        score, length = 0.0, max(1, len(row))
        # set order depends on PYTHONHASHSEED; sort so tied rows stay stable
        for term in ordered_terms:
            frequency = counts[term]
            if frequency:
                seen = document_frequency[term]
                idf = math.log(1 + (len(tokens) - seen + 0.5) / (seen + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        scores.append(score)
    return scores


def bm25(query: list[str], rows: list[str]) -> list[float]:
    if not query or not rows:
        return [0.0] * len(rows)
    return score_bm25(set(query), [tokenize(row) for row in rows])


def phrase_bonus(query: list[str], row: str) -> float:
    tokens = tokenize(row)
    bonus = 0.0
    for size, weight in ((3, 1.4), (2, 0.35)):
        query_phrases = {tuple(query[index:index + size]) for index in range(len(query) - size + 1)}
        row_phrases = {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}
        bonus += len(query_phrases & row_phrases) * weight
    return bonus


def header_cells(table: Table) -> list[str]:
    width = max((len(row) for row in table.headers), default=0)
    return [
        " ".join(row[column] for row in table.headers if column < len(row) and row[column])
        for column in range(width)
    ]


FULL_NUMBER_RE = re.compile(r"^\s*[([]?[+-]?\d[\d.,\s]*[%)]?\s*$")
ROW_METADATA_HEADERS = {"ma so", "thuyet minh", "stt", "so thu tu"}


def distinct_path(parts) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in parts:
        value = " ".join(value.split())
        key = " ".join(tokenize(value))
        if not value or not key or key in seen:
            continue
        result.append(value)
        seen.add(key)
    return tuple(result)


def column_paths(table: Table) -> tuple[tuple[str, ...], ...]:
    width = max((len(row) for row in (*table.headers, *table.rows)), default=0)
    return tuple(
        distinct_path(
            row[column] for row in table.headers
            if column < len(row)
        )
        for column in range(width)
    )


def row_path(table: Table, row_index: int) -> tuple[str, ...]:
    paths = column_paths(table)
    labels = []
    for column, cell in enumerate(table.rows[row_index]):
        value = " ".join(cell.split())
        if not value or FULL_NUMBER_RE.fullmatch(value):
            continue
        header = " ".join(tokenize(" ".join(paths[column]))) if column < len(paths) else ""
        if header in ROW_METADATA_HEADERS:
            continue
        labels.append(value)
    return distinct_path(labels)


def row_paths(table: Table) -> tuple[tuple[str, ...], ...]:
    header_rows = set(table.headers)
    columns = column_paths(table)
    active_section: tuple[str, ...] = ()
    paths = []
    for index, row in enumerate(table.rows):
        direct = row_path(table, index)
        if row in header_rows:
            paths.append(direct)
            active_section = ()
            continue
        has_value = any(
            FULL_NUMBER_RE.fullmatch(cell or "")
            and " ".join(tokenize(" ".join(columns[column]))) not in ROW_METADATA_HEADERS
            for column, cell in enumerate(row)
            if column < len(columns)
        )
        if direct and not has_value:
            active_section = direct
            paths.append(direct)
        elif active_section and direct:
            paths.append(distinct_path((*active_section, *direct)))
        else:
            paths.append(direct or active_section)
    return tuple(paths)


def positive_ranks(ranked: list[tuple[float, Table, int]]) -> dict[str, int]:
    positive = [item for item in ranked if item[0] > 0]
    return {table.table_id: rank for rank, (_, table, _) in enumerate(positive, 1)}


def ranked_tables(best: dict[str, tuple[float, Table, int]]) -> list[tuple[float, Table, int]]:
    return sorted(best.values(), key=lambda item: (-item[0], item[1].table_id))


def carries_figures(table: Table) -> bool:
    if max((len(row) for row in table.rows), default=0) < 2:
        return False
    if is_contents_page(table):
        return False
    return any(NUMERIC_CELL_RE.search(cell) for row in table.rows for cell in row[1:])


PAGE_RANGE_RE = re.compile(r"^\s*\d{1,3}\s*[-–]\s*\d{1,3}\s*$")


def is_contents_page(table: Table) -> bool:
    values = [cell for row in table.rows for cell in row[1:] if cell.strip()]
    if not values:
        return False
    if any(tokenize(cell) == ["trang"] for row in table.rows for cell in row[1:]):
        return True
    # A page column is ranges and small integers; a statement column is not.
    return sum(1 for cell in values if PAGE_RANGE_RE.match(cell)) / len(values) >= 0.5


def materialize_candidate_rows(candidates: list[Report], *, hierarchy: bool = False) -> list[tuple[Table, int, str]]:
    tables = [
        table for report in candidates
        for table in report_tables(str(report.path), report.identity)
        if carries_figures(table)
    ]
    rows = []
    for table in tables:
        paths = row_paths(table) if hierarchy else ()
        for index, row in enumerate(table.rows):
            raw = " ".join(row)
            path = " > ".join(paths[index]) if hierarchy and paths[index] else ""
            text = f"{path} | {raw}" if path else raw
            rows.append((table, index, text))
    return rows


def baseline_ranked_rows(
    query: list[str], context_query: list[str], metric_query: list[str],
    rows: list[tuple[Table, int, str]],
) -> list[tuple[float, Table, int]]:
    scores = bm25(query, [row for _, _, row in rows])
    context_scores = bm25(context_query, [contextual_row(table, row) for table, _, row in rows])
    metric_scores = bm25(metric_query, [row for _, _, row in rows])
    raw_best: dict[str, tuple[float, Table, int]] = {}
    context_best: dict[str, tuple[float, Table, int]] = {}
    metric_best: dict[str, tuple[float, Table, int]] = {}
    supporting: dict[str, list[float]] = {}
    for (table, row_index, row), raw_score, context_score, metric_score in zip(
        rows, scores, context_scores, metric_scores,
    ):
        raw_score += phrase_bonus(query, row)
        metric_score += phrase_bonus(metric_query, row)
        supporting.setdefault(table.table_id, []).append(max(raw_score, metric_score))
        for score, best in ((raw_score, raw_best), (context_score, context_best), (metric_score, metric_best)):
            previous = best.get(table.table_id)
            if previous is None or score > previous[0] or score == previous[0] and row_index < previous[2]:
                best[table.table_id] = score, table, row_index
    # sum the top supporting rows too, so a balance sheet beats a note repeating one item
    supporting_best = {
        table_id: (sum(sorted(supporting[table_id], reverse=True)[:SUPPORTING_ROWS]), table, row_index)
        for table_id, (_, table, row_index) in raw_best.items()
    }
    raw_ranked = ranked_tables(raw_best)
    context_ranks = positive_ranks(ranked_tables(context_best))
    metric_ranks = positive_ranks(ranked_tables(metric_best))
    if not context_ranks and not metric_ranks:
        return raw_ranked
    raw_ranks = positive_ranks(raw_ranked)
    supporting_ranks = positive_ranks(ranked_tables(supporting_best))
    ranked = []
    for table_id in raw_ranks.keys() | context_ranks.keys() | metric_ranks.keys():
        score = sum(
            1 / (RRF_OFFSET + ranks[table_id])
            for ranks in (raw_ranks, context_ranks, metric_ranks, supporting_ranks)
            if table_id in ranks
        )
        _, table, row_index = (
            raw_best[table_id] if table_id in raw_ranks
            else metric_best[table_id] if table_id in metric_ranks
            else context_best[table_id]
        )
        ranked.append((score, table, row_index))
    ranked.sort(key=lambda item: (-item[0], item[1].table_id))
    selected_ids = {table.table_id for _, table, _ in ranked}
    ranked.extend(item for item in raw_ranked if item[1].table_id not in selected_ids)
    return ranked


def table_budget(report_count: int, setting: str = "auto") -> int:
    # "auto" or "auto:n" for n tables per gated report
    multiplier = setting.partition(":")[2]
    return min(30, max(1, int(multiplier or 2) * report_count))


def select_report_coverage(
    ranked: list[tuple[float, Table, int]], candidates: list[Report], top_k: int,
) -> list[tuple[float, Table, int]]:
    selected: list[tuple[float, Table, int]] = []
    selected_ids: set[str] = set()
    for report in candidates[:top_k]:
        item = next(
            (item for item in ranked if item[1].report_id == report.identity.report_id and item[1].table_id not in selected_ids),
            None,
        )
        if item is not None:
            selected.append(item)
            selected_ids.add(item[1].table_id)
    selected.extend(item for item in ranked if item[1].table_id not in selected_ids)
    return selected


def retrieve_rows(
    question: str,
    metadata: dict,
    reports: list[Report],
    report_ids: list[str],
    top_k: int = 5,
    hierarchy: bool = False,
) -> dict:
    reports_by_id = {report.identity.report_id: report for report in reports}
    candidates = [reports_by_id[report_id] for report_id in report_ids if report_id in reports_by_id]
    rows = materialize_candidate_rows(candidates, hierarchy=hierarchy)
    ranked = baseline_ranked_rows(
        query_tokens(question, metadata),
        context_query_tokens(question, metadata),
        metric_query_tokens(question, metadata),
        rows,
    )
    ranked = select_report_coverage(ranked, candidates, top_k)[:top_k]
    return {
        "tables": [
            {
                "table_id": table.table_id,
                "report_id": table.report_id,
                "page": table.page,
                "start_line": table.start_line,
                "score": round(score, 6),
                "row_index": row_index,
                "row_cells": list(table.rows[row_index]),
                "rows": [list(row) for row in table.rows],
                "header_cells": header_cells(table),
                "title": table.title,
                "periods": list(table.periods),
                "unit": table.unit,
            }
            for score, table, row_index in ranked
        ],
    }
