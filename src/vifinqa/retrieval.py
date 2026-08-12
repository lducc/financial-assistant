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
ROLE_STOPWORDS = {
    "cac", "co", "ghi", "nhan", "so", "gia", "tri", "nao", "cao", "thap", "lon", "nho",
    "hon", "nhat", "me", "tap", "doan", "ctcp", "ma", "tinh", "va",
}
PERIOD_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")
TITLE_RE = re.compile(r"\b(?:bao cao|bang|thuyet minh)\b")
UNIT_PHRASES = ("vnd", "don vi", "trieu dong", "nghin dong", "ty dong", "million", "billion")
HEADER_TOKENS = {"ma", "so", "thuyet", "minh", "chi", "tieu", "don", "vi"}
RRF_OFFSET = 60
# Equal-start weights for field-aware fusion; tune on dev only.
FIELD_WEIGHTS = {
    "row": 4.0,
    "title": 4.0,
    "header": 4.0,
    "unit": 1.0,
    "phrase": 4.0,
    "rrf": 4.0,
}
RANK_FUSION_FAMILIES = {
    "row": ("folded_row", "unicode_row"),
    "context": ("folded_context", "unicode_context"),
    "metadata": ("title", "header", "unit"),
}


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
    """Tokenize NFC text without removing Vietnamese diacritics."""
    return UNICODE_TOKEN_RE.findall(unicodedata.normalize("NFC", text.lower()))


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


def metric_query_tokens(question: str, metadata: dict, *, keep_years: bool = False) -> list[str]:
    """Keep line-item terms while removing arithmetic wording from a question."""
    blocked = {ticker.lower() for ticker in metadata.get("tickers", [])}
    if not keep_years:
        blocked.update(str(year) for year in metadata.get("years", []))
    return [
        token for token in tokenize(question.replace("%", " percent "))
        if token not in METRIC_STOPWORDS and token not in blocked and (keep_years or not token.isdigit())
    ]


def unicode_query_tokens(question: str, metadata: dict, *, keep_context: bool = False) -> list[str]:
    """Mirror folded query filtering while retaining NFC Vietnamese tokens."""
    blocked = set() if keep_context else {str(year) for year in metadata.get("years", [])}
    blocked.update(ticker.lower() for ticker in metadata.get("tickers", []))
    stopwords = CONTEXT_STOPWORDS if keep_context else STOPWORDS
    result = []
    for token in unicode_tokenize(question.replace("%", " percent ")):
        folded = tokenize(token)
        folded_token = folded[0] if folded else token
        if folded_token in stopwords or folded_token in blocked:
            continue
        if not keep_context and token.isdigit():
            continue
        result.append(token)
    return result


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


def unicode_bm25(query: list[str], rows: list[str]) -> list[float]:
    """BM25 variant preserving NFC Vietnamese distinctions."""
    if not query or not rows:
        return [0.0] * len(rows)
    terms = set(query)
    tokens = [unicode_tokenize(row) for row in rows]
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


def field_texts(table: Table) -> dict[str, str]:
    return {
        "title": " ".join((table.title, *table.context)).replace("%", " percent "),
        "header": " ".join((
            *(" ".join(header) for header in table.headers),
            " ".join(table.periods),
        )).replace("%", " percent "),
        "unit": (table.unit or "").replace("%", " percent "),
    }


def field_aware_table_scores(
    query: list[str],
    context_query: list[str],
    rows: list[tuple[Table, int, str]],
    weights: dict[str, float] | None = None,
) -> dict[str, tuple[float, Table, int, dict[str, float]]]:
    """Best row per table under field-decomposed BM25 + RRF with baseline."""
    weights = {**FIELD_WEIGHTS, **(weights or {})}
    if not rows:
        return {}
    row_texts = [row for _, _, row in rows]
    raw_scores = bm25(query, row_texts)
    context_scores = bm25(context_query, [contextual_row(table, row) for table, _, row in rows])

    unique_tables: dict[str, Table] = {}
    for table, _, _ in rows:
        unique_tables.setdefault(table.table_id, table)
    table_list = list(unique_tables.values())
    fields = [field_texts(table) for table in table_list]
    title_scores = bm25(context_query, [item["title"] for item in fields])
    header_scores = bm25(context_query, [item["header"] for item in fields])
    unit_scores = bm25(context_query, [item["unit"] for item in fields])
    field_by_id = {
        table.table_id: {
            "title": title_scores[index],
            "header": header_scores[index],
            "unit": unit_scores[index],
        }
        for index, table in enumerate(table_list)
    }

    raw_best: dict[str, tuple[float, Table, int]] = {}
    context_best: dict[str, tuple[float, Table, int]] = {}
    row_best: dict[str, tuple[float, Table, int]] = {}
    for (table, row_index, row), raw_score, context_score in zip(rows, raw_scores, context_scores):
        phrase = phrase_bonus(query, row)
        raw_with_phrase = raw_score + phrase
        for score, best in (
            (raw_with_phrase, raw_best),
            (context_score, context_best),
            (raw_score, row_best),
        ):
            previous = best.get(table.table_id)
            if previous is None or score > previous[0] or score == previous[0] and row_index < previous[2]:
                best[table.table_id] = score, table, row_index

    raw_ranked = ranked_tables(raw_best)
    context_ranked = ranked_tables(context_best)
    raw_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(
        [item for item in raw_ranked if item[0] > 0], 1
    )}
    context_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(
        [item for item in context_ranked if item[0] > 0], 1
    )}

    fused: dict[str, tuple[float, Table, int, dict[str, float]]] = {}
    for table_id, table in unique_tables.items():
        row_score, _, row_index = row_best.get(table_id, (0.0, table, 0))
        # Prefer row index from strongest raw+phrase hit when present.
        if table_id in raw_best:
            _, table, row_index = raw_best[table_id]
        parts = field_by_id.get(table_id, {"title": 0.0, "header": 0.0, "unit": 0.0})
        phrase = 0.0
        if table_id in raw_best and table_id in row_best:
            phrase = max(0.0, raw_best[table_id][0] - row_best[table_id][0])
        rrf = 0.0
        if table_id in raw_ranks:
            rrf += 1 / (RRF_OFFSET + raw_ranks[table_id])
        if table_id in context_ranks:
            rrf += 1 / (RRF_OFFSET + context_ranks[table_id])
        breakdown = {
            "row": row_score,
            "title": parts["title"],
            "header": parts["header"],
            "unit": parts["unit"],
            "phrase": phrase,
            "rrf": rrf,
        }
        total = (
            weights["row"] * breakdown["row"]
            + weights["title"] * breakdown["title"]
            + weights["header"] * breakdown["header"]
            + weights["unit"] * breakdown["unit"]
            + weights["phrase"] * breakdown["phrase"]
            + weights["rrf"] * breakdown["rrf"]
        )
        if total > 0 or row_score > 0 or rrf > 0:
            fused[table_id] = total, table, row_index, breakdown
    return fused


def rank_fuse_signal_scores(
    signal_scores: dict[str, dict[str, float]],
    families: dict[str, tuple[str, ...]] = RANK_FUSION_FAMILIES,
) -> tuple[
    dict[str, float],
    dict[str, dict[str, int]],
    dict[str, dict[str, float]],
]:
    """Fuse positive signal ranks with one equal vote per signal family."""
    ranks_by_signal: dict[str, dict[str, int]] = {}
    for signal in (signal for members in families.values() for signal in members):
        positive = [
            (score, table_id)
            for table_id, score in signal_scores.get(signal, {}).items()
            if math.isfinite(score) and score > 0
        ]
        if positive:
            positive.sort(key=lambda item: (-item[0], item[1]))
            ranks_by_signal[signal] = {
                table_id: rank for rank, (_, table_id) in enumerate(positive, 1)
            }

    table_ids = sorted({table_id for ranks in ranks_by_signal.values() for table_id in ranks})
    totals: dict[str, float] = {}
    table_ranks: dict[str, dict[str, int]] = {}
    family_contributions: dict[str, dict[str, float]] = {}
    for table_id in table_ids:
        table_ranks[table_id] = {
            signal: ranks[table_id]
            for signal, ranks in ranks_by_signal.items()
            if table_id in ranks
        }
        contributions = {}
        for family, members in families.items():
            active = [signal for signal in members if signal in ranks_by_signal]
            if active:
                contributions[family] = sum(
                    1 / (RRF_OFFSET + ranks_by_signal[signal][table_id])
                    if table_id in ranks_by_signal[signal] else 0.0
                    for signal in active
                ) / len(active)
        family_contributions[table_id] = contributions
        totals[table_id] = sum(contributions.values())
    return totals, table_ranks, family_contributions


def _best_table_rows(
    rows: list[tuple[Table, int, str]], scores: list[float],
) -> dict[str, tuple[float, Table, int]]:
    best = {}
    for (table, row_index, _), score in zip(rows, scores):
        previous = best.get(table.table_id)
        if previous is None or score > previous[0] or score == previous[0] and row_index < previous[2]:
            best[table.table_id] = score, table, row_index
    return best


def rank_fusion_table_scores(
    question: str,
    metadata: dict,
    query: list[str],
    context_query: list[str],
    rows: list[tuple[Table, int, str]],
) -> dict[str, tuple[float, Table, int, dict]]:
    """Rank-fuse folded, NFC, contextual, and structured table signals."""
    if not rows:
        return {}
    row_texts = [row for _, _, row in rows]
    context_texts = [contextual_row(table, row) for table, _, row in rows]
    row_best = {
        "folded_row": _best_table_rows(rows, bm25(query, row_texts)),
        "unicode_row": _best_table_rows(
            rows, unicode_bm25(unicode_query_tokens(question, metadata), row_texts),
        ),
        "folded_context": _best_table_rows(rows, bm25(context_query, context_texts)),
        "unicode_context": _best_table_rows(
            rows,
            unicode_bm25(
                unicode_query_tokens(question, metadata, keep_context=True), context_texts,
            ),
        ),
    }

    unique_tables: dict[str, Table] = {}
    for table, _, _ in rows:
        unique_tables.setdefault(table.table_id, table)
    table_list = list(unique_tables.values())
    fields = [field_texts(table) for table in table_list]
    signal_scores = {
        signal: {table_id: item[0] for table_id, item in best.items()}
        for signal, best in row_best.items()
    }
    for field in ("title", "header", "unit"):
        signal_scores[field] = {
            table.table_id: score
            for table, score in zip(
                table_list, bm25(context_query, [item[field] for item in fields]),
            )
        }

    totals, signal_ranks, family_contributions = rank_fuse_signal_scores(signal_scores)
    fused = {}
    row_signal_order = (
        "folded_row", "unicode_row", "folded_context", "unicode_context",
    )
    for table_id, total in totals.items():
        table = unique_tables[table_id]
        row_index = 0
        for signal in row_signal_order:
            candidate = row_best[signal].get(table_id)
            if candidate is not None and candidate[0] > 0:
                _, table, row_index = candidate
                break
        fused[table_id] = total, table, row_index, {
            "signal_ranks": signal_ranks[table_id],
            "family_contributions": family_contributions[table_id],
        }
    return fused


def header_cells(table: Table) -> list[str]:
    """Merge every header row into one label per column for cell-level gating."""
    width = max((len(row) for row in table.headers), default=0)
    return [
        " ".join(row[column] for row in table.headers if column < len(row) and row[column])
        for column in range(width)
    ]


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


def select_candidate_reports(
    reports: list[Report], metadata: dict, report_ids: list[str] | None,
) -> tuple[list[Report], str]:
    """Stage 1: preserve report-gate behavior for implicit and explicit IDs."""
    if report_ids is None:
        return filter_reports(reports, metadata)
    reports_by_id = {report.identity.report_id: report for report in reports}
    return [reports_by_id[report_id] for report_id in report_ids if report_id in reports_by_id], "report_ids"


def materialize_candidate_rows(candidates: list[Report]) -> tuple[list[Table], list[tuple[Table, int, str]]]:
    """Stage 2: materialize immutable table and row candidates in report order."""
    tables = [table for report in candidates for table in report_tables(str(report.path), report.identity)]
    return tables, [(table, index, " ".join(row)) for table in tables for index, row in enumerate(table.rows)]


def baseline_ranked_rows(
    query: list[str], context_query: list[str], rows: list[tuple[Table, int, str]],
) -> list[tuple[float, Table, int]]:
    """Stage 2 baseline ranker. Keep production ranking byte-for-byte stable."""
    scores = bm25(query, [row for _, _, row in rows])
    context_scores = bm25(context_query, [contextual_row(table, row) for table, _, row in rows])
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
    if not context_positive:
        return raw_ranked
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
            selected[table_id][1], selected[table_id][2],
        )
        for table_id in raw_ranks.keys() | context_ranks.keys()
    ]
    ranked.sort(key=lambda item: (-item[0], item[1].table_id))
    selected_ids = {table.table_id for _, table, _ in ranked}
    ranked.extend(item for item in raw_ranked if item[1].table_id not in selected_ids)
    return ranked


def select_evidence_slots(
    ranked: list[tuple[float, Table, int]], candidates: list[Report], metadata: dict, top_k: int,
) -> tuple[list[tuple[float, Table, int]], list[int]]:
    """Reserve one unique baseline table per required report-year before fill."""
    years = list(dict.fromkeys(metadata.get("slot_years", metadata.get("years", []))))
    report_year = {report.identity.report_id: report.identity.year for report in candidates}
    selected: list[tuple[float, Table, int]] = []
    selected_ids: set[str] = set()
    uncovered: list[int] = []
    for year in years[:top_k]:
        item = next((item for item in ranked if item[1].table_id not in selected_ids and report_year.get(item[1].report_id) == year), None)
        if item is None:
            uncovered.append(year)
        else:
            selected.append(item)
            selected_ids.add(item[1].table_id)
    selected.extend(item for item in ranked if item[1].table_id not in selected_ids)
    return selected, uncovered


def select_report_coverage(
    ranked: list[tuple[float, Table, int]], candidates: list[Report], top_k: int,
) -> list[tuple[float, Table, int]]:
    """Reserve one relevant table per gated report before relevance-only fill."""
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


@lru_cache(maxsize=2)
def load_dense_index(path_text: str):
    """Load a reusable E5 index only for explicit dense-hybrid retrieval."""
    from .dense import DenseIndex
    return DenseIndex(Path(path_text))


def retrieve_rows(
    question: str,
    metadata: dict,
    reports: list[Report],
    top_k: int = 5,
    report_ids: list[str] | None = None,
    mode: str = "baseline",
    field_weights: dict[str, float] | None = None,
    reranker: str | None = None,
    reranker_batch_size: int = 8,
    dense_index_path: str | Path | None = None,
) -> dict:
    if mode not in {"baseline", "dense-hybrid", "metric-focused", "metric-coverage", "role-coverage", "report-coverage", "field-aware", "field-coverage", "rank-fusion", "evidence-slots"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    if reranker not in {None, "mmarco"}:
        raise ValueError(f"Unknown reranker: {reranker}")
    started = time.perf_counter()
    candidate_metadata = metadata
    if mode == "evidence-slots" and metadata.get("slot_years"):
        candidate_metadata = {**metadata, "years": metadata["slot_years"]}
    candidates, stage = select_candidate_reports(reports, candidate_metadata, report_ids)
    tables, rows = materialize_candidate_rows(candidates)
    query = query_tokens(question, metadata)
    context_query = context_query_tokens(question, metadata)
    if mode in {"metric-focused", "metric-coverage"}:
        query = metric_query_tokens(question, metadata)
        context_query = metric_query_tokens(question, metadata, keep_years=True)
    baseline_ranked = baseline_ranked_rows(query, context_query, rows)
    if mode == "dense-hybrid":
        if dense_index_path is None:
            raise ValueError("dense-hybrid retrieval requires dense_index_path")
        from .dense import encode_query, fused_rankings
        tables_by_id = {table.table_id: table for table in tables}
        dense_ranked = load_dense_index(str(dense_index_path)).rank(
            encode_query(question), [report.identity.report_id for report in candidates], tables_by_id, top_k=50,
        )
        ranked = fused_rankings(baseline_ranked[:50], dense_ranked)
    else:
        ranked = baseline_ranked
    baseline_top_50 = {table.table_id for _, table, _ in baseline_ranked[:50]}
    experimental_rows = [row for row in rows if row[0].table_id in baseline_top_50]
    field_breakdowns: dict[str, dict[str, float]] = {}
    experimental_fallback = False
    reranker_fallback = False
    uncovered_slots: list[int] = []
    if mode in {"field-aware", "field-coverage", "rank-fusion"}:
        try:
            fused = (
                field_aware_table_scores(query, context_query, experimental_rows, field_weights)
                if mode in {"field-aware", "field-coverage"}
                else rank_fusion_table_scores(question, metadata, query, context_query, experimental_rows)
            )
            ranked = sorted(
                ((total, table, row_index) for total, table, row_index, _ in fused.values()),
                key=lambda item: (-item[0], item[1].table_id),
            )
            field_breakdowns = {table_id: breakdown for table_id, (_, _, _, breakdown) in fused.items()}
            if not ranked:
                ranked = baseline_ranked
                experimental_fallback = True
        except Exception:
            ranked = baseline_ranked
            experimental_fallback = True
    elif mode != "dense-hybrid":
        ranked = baseline_ranked
    baseline_rank = {
        table.table_id: rank for rank, (_, table, _) in enumerate(
        baseline_ranked if mode in {"dense-hybrid", "field-aware", "field-coverage", "rank-fusion"} else ranked, 1,
        )
    }
    role_ranks = {}
    if mode == "role-coverage":
        for year, role_ranked in role_rankings(question, metadata, candidates, rows).items():
            for rank, (_, table, _) in enumerate(role_ranked, 1):
                role_ranks.setdefault(table.table_id, {})[year] = rank
        if role_ranks:
            fused_role = []
            for score, table, row_index in ranked:
                role_score = sum(1 / (RRF_OFFSET + rank) for rank in role_ranks.get(table.table_id, {}).values())
                fused_role.append((score + role_score, table, row_index))
            ranked = sorted(fused_role, key=lambda item: (-item[0], item[1].table_id))
    if reranker:
        from .rerank import rerank
        try:
            reranked = rerank(question, baseline_ranked[:50], batch_size=reranker_batch_size)
            if not reranked:
                raise RuntimeError("reranker returned no candidates")
            ranked = reranked
        except Exception:
            ranked = baseline_ranked
            reranker_fallback = True
    if mode == "evidence-slots" and not reranker_fallback:
        ranked, uncovered_slots = select_evidence_slots(ranked, candidates, metadata, top_k)
    if mode in {"report-coverage", "metric-coverage", "field-coverage", "dense-hybrid"} and not reranker_fallback:
        ranked = select_report_coverage(ranked, candidates, top_k)
    ranked = ranked[:top_k]
    return {
        "filter_stage": stage,
        "query_tokens": query,
        "context_query_tokens": context_query,
        "mode": mode,
        "reranker": reranker,
        **({"reranker_fallback": reranker_fallback} if reranker else {}),
        "uncovered_slots": uncovered_slots,
        "candidate_report_count": len(candidates),
        "candidate_table_count": len(tables),
        "candidate_row_count": len(rows),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        **({"experimental_fallback": experimental_fallback} if mode in {"field-aware", "rank-fusion"} else {}),
        "tables": [
            {
                "table_id": table.table_id,
                "report_id": table.report_id,
                "page": table.page,
                "start_line": table.start_line,
                "score": round(score, 6),
                "row_index": row_index,
                "row_cells": list(table.rows[row_index]),
                "header_cells": header_cells(table),
                "title": table.title,
                "periods": list(table.periods),
                "unit": table.unit,
                "pre_role_rank": baseline_rank.get(table.table_id),
                "role_ranks": role_ranks.get(table.table_id, {}),
                "field_scores": field_breakdowns.get(table.table_id, {}),
            }
            for score, table, row_index in ranked
        ],
    }
