from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .catalog import Company
from .aliases import match_aliases
from .normalize import content_tokens, fold


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
RANGE_RE = re.compile(
    r"(?:giai doan|trong cac nam(?: tai chinh)? tu|qua cac nam tu|tu nam)\s+"
    r"((?:19|20)\d{2})\s*(?:(?:den|toi)\s*)?(?:nam\s+)?((?:19|20)\d{2})"
)


@dataclass
class ParsedQuestion:
    tickers: list[str]
    years: list[int]
    scope: str
    scope_by_year: dict[int, str]
    entity_confidence: float
    entity_method: str
    candidate_tickers: list[str]


def _explicit_tickers(question: str, known: set[str]) -> list[str]:
    parenthesized = re.findall(r"\(([A-Za-z0-9]{2,6})\)", question)
    upper_tokens = re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,5}(?![A-Za-z0-9])", question)
    explicit = [token.upper() for token in parenthesized + upper_tokens if token.upper() in known]
    normalized = fold(question)
    if any(cue in normalized for cue in ("ma co phieu", "gom cac cong ty", "nhom")):
        raw_tokens = re.findall(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]{1,5}(?![A-Za-z0-9])", question)
        explicit.extend(token.upper() for token in raw_tokens if token.upper() in known)
    return list(dict.fromkeys(explicit))


def _company_score(question_folded: str, question_content: set[str], company: Company) -> float:
    if company.folded_name and company.folded_name in question_folded:
        return 1.0
    if company.content_name and company.content_name in question_folded:
        return 0.98
    company_tokens = set(content_tokens(company.name))
    if not company_tokens:
        return 0.0
    overlap = len(question_content & company_tokens) / len(company_tokens)
    sequence = SequenceMatcher(None, company.content_name, question_folded).ratio()
    return 0.82 * overlap + 0.18 * sequence


def entity_candidates(question: str, companies: dict[str, Company]) -> list[tuple[str, float]]:
    qfold = fold(question)
    qtokens = set(content_tokens(question))
    scored = [(ticker, _company_score(qfold, qtokens, company)) for ticker, company in companies.items()]
    return sorted(scored, key=lambda item: (item[1], item[0]), reverse=True)


def infer_entities(question: str, companies: dict[str, Company]) -> tuple[list[str], float, str, list[str]]:
    explicit = _explicit_tickers(question, set(companies))
    scored = entity_candidates(question, companies)
    fuzzy_shortlist = [ticker for ticker, score in scored[:8] if score >= 0.32]

    # Prefer complete official names. Only fall back to shortened names when no
    # complete name matches, then add historical aliases verified in BTC reports.
    qfold = fold(question)
    qcontent = " ".join(content_tokens(question))
    full_names = [
        ticker for ticker, company in companies.items()
        if company.folded_name
        and re.search(rf"(?<![a-z0-9]){re.escape(company.folded_name)}(?![a-z0-9])", qfold)
    ]
    content_names = [
        ticker for ticker, company in companies.items()
        if len(company.content_name.split()) >= 3
        and re.search(rf"(?<![a-z0-9]){re.escape(company.content_name)}(?![a-z0-9])", qcontent)
    ]
    alias_names = match_aliases(question)

    def is_counterparty_only(ticker: str) -> bool:
        phrase = companies[ticker].content_name
        found = False
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", qcontent):
            found = True
            prefix = qcontent[max(0, match.start() - 80):match.start()]
            relationship = re.search(
                r"(?:phai thu tu|phai tra cho|dau tu vao|ban (?:hang )?(?:cho|voi)|giao dich voi)"
                r"\s+((?:[a-z0-9]+\s+){0,7})$",
                prefix,
            )
            # "dau tu vao ... cua cong ty me X" describes a metric of X,
            # while "dau tu vao X cua Y" makes X the investee.
            if relationship and "cua me" not in relationship.group(1):
                continue
            else:
                return False
        return found

    full_names = [ticker for ticker in full_names if not is_counterparty_only(ticker)]
    content_names = [ticker for ticker in content_names if not is_counterparty_only(ticker)]
    alias_names = [ticker for ticker in alias_names if not is_counterparty_only(ticker)]
    specific_content_names = [
        ticker for ticker in content_names
        if not any(
            companies[ticker].content_name != companies[other].content_name
            and companies[ticker].content_name in companies[other].content_name
            for other in content_names
        )
    ]
    exact_names = list(dict.fromkeys(full_names + specific_content_names + alias_names))
    # A brand token can also be a listed ticker. Prefer the specific company
    # mention, e.g. "Chung khoan FPT" -> FTS rather than FPT.
    explicit = [
        ticker for ticker in explicit
        if not any(
            ticker != other and fold(ticker) in companies[other].content_name.split()
            for other in exact_names
        )
    ]
    entities = list(dict.fromkeys(explicit + exact_names))
    shortlist = list(dict.fromkeys(entities + fuzzy_shortlist))
    if entities:
        if explicit and len(entities) == len(explicit):
            method = "explicit_ticker"
        elif len(entities) == 1:
            method = "single_official_name"
        else:
            method = "union_all_entity_detectors"
        return entities, 1.0 if explicit else 0.98, method, shortlist
    best_ticker, best_score = scored[0]
    second_score = scored[1][1]
    if best_score >= 0.73 and best_score - second_score >= 0.08:
        return [best_ticker], best_score, "fuzzy_high_confidence", shortlist
    return [], best_score, "ambiguous", shortlist


def infer_years(question: str) -> list[int]:
    normalized = fold(question)
    years = {int(value) for value in YEAR_RE.findall(question)}
    for start, end in RANGE_RE.findall(normalized):
        lo, hi = sorted((int(start), int(end)))
        if hi - lo <= 15:
            years.update(range(lo, hi + 1))
    return sorted(years)


def infer_scope(question: str, years: list[int]) -> tuple[str, dict[int, str]]:
    normalized = fold(question)
    separate_cues = ("cong ty me", "bao cao rieng", "co so rieng")
    consolidated_cues = ("hop nhat", "bao cao hop nhat")
    has_separate = any(cue in normalized for cue in separate_cues)
    has_consolidated = any(cue in normalized for cue in consolidated_cues)
    if has_separate and not has_consolidated:
        return "separate", {year: "separate" for year in years}
    if has_consolidated and not has_separate:
        return "consolidated", {year: "consolidated" for year in years}
    if not (has_separate and has_consolidated):
        return "consolidated", {year: "consolidated" for year in years}

    # Mixed-scope questions are rare. Associate each year with its nearest preceding cue.
    cue_matches = []
    for cue in separate_cues:
        cue_matches.extend((m.start(), "separate") for m in re.finditer(cue, normalized))
    for cue in consolidated_cues:
        cue_matches.extend((m.start(), "consolidated") for m in re.finditer(cue, normalized))
    cue_matches.sort()
    scope_by_year = {}
    for year in years:
        positions = [m.start() for m in re.finditer(str(year), normalized)]
        position = positions[0] if positions else len(normalized)
        preceding = [item for item in cue_matches if item[0] <= position]
        scope_by_year[year] = preceding[-1][1] if preceding else cue_matches[0][1]
    return "mixed", scope_by_year


def parse_question(question: str, companies: dict[str, Company]) -> ParsedQuestion:
    tickers, confidence, method, candidates = infer_entities(question, companies)
    years = infer_years(question)
    scope, scope_by_year = infer_scope(question, years)
    return ParsedQuestion(tickers, years, scope, scope_by_year, confidence, method, candidates)

