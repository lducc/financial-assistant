"""Targeted E005b repair: exact bare ticker detection."""

from __future__ import annotations

import re

from .metadata import Company, parse_question


UPPER_TOKEN_RE = re.compile(r"(?<![A-Z])([A-Z]{2,5})(?![A-Z])")


def parse_question_with_bare_tickers(question: str, companies: list[Company]) -> dict:
    parsed = parse_question(question, companies)
    known = {company.ticker for company in companies}
    candidates = {item["ticker"]: item for item in parsed["entity_candidates"]}
    for token in UPPER_TOKEN_RE.findall(question):
        if token in known and token not in candidates:
            candidates[token] = {"ticker": token, "source": "explicit_bare_ticker"}
    tickers = sorted(candidates)
    parsed["tickers"] = tickers
    parsed["entity_candidates"] = [candidates[ticker] for ticker in tickers]
    parsed["hard_filter"]["tickers"] = tickers
    parsed["unresolved_entity"] = not tickers
    parsed["entity_count"] = len(tickers)
    return parsed

