from __future__ import annotations

from collections import defaultdict

from .catalog import Report
from .parser import ParsedQuestion


def minimal_report_years(years: list[int]) -> list[int]:
    """Annual report Y normally contains current Y and comparative Y-1 values."""
    uncovered = set(years)
    selected = []
    while uncovered:
        year = max(uncovered)
        selected.append(year)
        uncovered.discard(year)
        uncovered.discard(year - 1)
    return sorted(selected)


def choose_report(candidates: list[Report], scope: str) -> Report | None:
    if not candidates:
        return None
    exact = [report for report in candidates if report.scope == scope]
    unspecified = [report for report in candidates if report.scope == "unknown"]
    pool = exact or unspecified or candidates
    # Prefer the official consolidated naming over the legacy aggregated variant.
    return sorted(pool, key=lambda report: ("_aggregated" in report.doc_id, report.doc_id))[0]


def retrieve_docs(parsed: ParsedQuestion, reports: dict[str, Report], use_comparative_cover: bool = True) -> tuple[list[str], list[dict]]:
    by_ticker_year: dict[tuple[str, int], list[Report]] = defaultdict(list)
    for report in reports.values():
        by_ticker_year[(report.ticker, report.year)].append(report)
    selected: list[Report] = []
    decisions = []
    requested_years = parsed.years or sorted({report.year for report in reports.values()})
    report_years = minimal_report_years(requested_years) if use_comparative_cover else requested_years
    for ticker in parsed.tickers:
        for year in report_years:
            scope = parsed.scope_by_year.get(year, parsed.scope if parsed.scope != "mixed" else "consolidated")
            report = choose_report(by_ticker_year.get((ticker, year), []), scope)
            decisions.append({
                "ticker": ticker,
                "requested_year": year,
                "scope": scope,
                "selected_doc": report.doc_id if report else None,
            })
            if report is not None:
                selected.append(report)
    docs = list(dict.fromkeys(report.doc_id for report in selected))
    return docs, decisions

