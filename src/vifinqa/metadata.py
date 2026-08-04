"""Conservative question metadata extraction for ViFinQA retrieval."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import re
import unicodedata
from pathlib import Path


YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
PAREN_TOKEN_RE = re.compile(r"\(([A-Z]{2,5})\)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Accent-insensitive, punctuation-insensitive form used only for matching."""

    decomposed = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return NON_ALNUM_RE.sub(" ", without_marks).strip()


def aliases(company_name: str) -> set[str]:
    value = normalize(company_name)
    values = {value}
    for prefix in (
        "ctcp ",
        "cong ty co phan ",
        "cong ty tnhh ",
        "tong cong ty co phan ",
        "tong cong ty ",
    ):
        if value.startswith(prefix):
            values.add(value.removeprefix(prefix).strip())
    return {item for item in values if len(item) >= 8}


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    aliases: tuple[str, ...]


def load_companies(path: Path) -> list[Company]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        companies = []
        for row in reader:
            if len(row) < 2:
                continue
            ticker, name = row[0].strip().upper(), row[1].strip()
            if ticker and name:
                companies.append(Company(ticker=ticker, name=name, aliases=tuple(sorted(aliases(name), key=len, reverse=True))))
    return companies


def parse_question(question: str, companies: list[Company]) -> dict:
    normalized_question = normalize(question)
    candidates: dict[str, dict] = {}
    ticker_set = {company.ticker for company in companies}
    for ticker in PAREN_TOKEN_RE.findall(question):
        if ticker in ticker_set:
            candidates[ticker] = {"ticker": ticker, "source": "explicit_parenthetical_ticker"}
    for company in companies:
        if company.ticker in candidates:
            continue
        matched_alias = next((alias for alias in company.aliases if alias in normalized_question), None)
        if matched_alias:
            candidates[company.ticker] = {"ticker": company.ticker, "source": "company_alias", "matched_alias": matched_alias}
    lower = question.lower()
    if any(phrase in lower for phrase in ("công ty mẹ", "báo cáo tài chính riêng", "báo cáo riêng")):
        scope, scope_confidence = "separate", "high"
    elif any(phrase in lower for phrase in ("báo cáo tài chính hợp nhất", "báo cáo hợp nhất")):
        scope, scope_confidence = "consolidated", "high"
    else:
        scope, scope_confidence = None, "unspecified"
    years = sorted({int(value) for value in YEAR_RE.findall(question)})
    return {
        "question": question,
        "tickers": sorted(candidates),
        "entity_candidates": [candidates[ticker] for ticker in sorted(candidates)],
        "years": years,
        "scope": scope,
        "scope_confidence": scope_confidence,
        "hard_filter": {
            "tickers": sorted(candidates),
            "years": years,
            "scope": scope,
        },
        "unresolved_entity": not candidates,
        "entity_count": len(candidates),
    }

