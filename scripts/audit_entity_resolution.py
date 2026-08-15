#!/usr/bin/env python3
"""Check every resolved ticker against the question that produced it.

The document gate is deterministic and has never been tuned on a leaderboard —
Docs F2 has been identical at 0.9711 across every submission — so it cannot be
overfitted in the usual sense. But 200 of the 1,012 questions are resolved only
through the 57-entry alias table in `src/docs.py`, and 11 of those aliases never
fire, which is the signature of a table grown against the cases that happened to
break rather than designed. The failure it invites is silent: an unlisted
subsidiary whose name contains a parent's alias resolves to the parent, gates the
wrong reports, and loses both the document and the table score with no error.

This needs no labels, so it covers the questions we will be scored on privately
as well as the ones we can see scored now. For each resolved ticker it asks the
only question that can be asked without gold: does the company we resolved to
actually appear in the question? A ticker written in the text, an official name
present in full, or an alias that fires are all evidence. A resolution supported
by none of those is reported.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

from docs import ALIAS_SPECS, load_companies, match_aliases, parse_question
from vifinqa.answers import fold
from vifinqa.jsonl import load_jsonl


# Legal-form words carry no identity: every company has them.
LEGAL_TOKENS = {
    "cong", "ty", "co", "phan", "ctcp", "tong", "tnhh", "tap", "doan",
    "ngan", "hang", "tmcp", "mtv", "nhtmcp",
}


def normalize(text: str) -> str:
    """Fold accents, drop punctuation, collapse spaces.

    Questions write the same entity many ways — "hpx" lowercase in a list,
    "C.E.O" with stops, "Sài Gòn - Hà Nội" with a hyphen — and an audit that
    misses those reports resolver bugs that are only audit bugs. Comparing on
    letters and digits alone removes that whole class of false alarm.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", fold(text))).strip()


def evidence_for(question: str, ticker: str, companies: dict) -> str | None:
    """How the question names this company, or None when it does not."""
    text = normalize(question)
    padded = f" {text} "
    if f" {ticker.lower()} " in padded:
        return "ticker_in_text"
    # "C.E.O" and "V N M" survive normalization as separated letters.
    if f" {' '.join(ticker.lower())} " in padded:
        return "ticker_spelled_out"
    company = companies.get(ticker)
    if company is None:
        return None
    # Token coverage rather than substring: a registered name carries legal
    # boilerplate the question drops ("- CTCP"), and a short name like "CTCP
    # Tasco" is two tokens. Asking how much of the distinctive name the question
    # reproduces handles both without inventing failures.
    words = set(text.split())
    for attribute, label in (("name", "official_name"), ("content_name", "content_name")):
        tokens = [token for token in normalize(getattr(company, attribute, "") or "").split()
                  if token not in LEGAL_TOKENS]
        if tokens and sum(token in words for token in tokens) / len(tokens) >= 0.8:
            return label
    if ticker in match_aliases(question):
        return "alias"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "raw" / "vifinqa" / "questions" / "questions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "diagnostics" / "entity_audit.json")
    args = parser.parse_args()

    companies = load_companies(args.dataset_root / "code_stock.csv")
    questions = load_jsonl(args.questions)

    support: Counter = Counter()
    unsupported, unresolved = [], []
    alias_use: Counter = Counter()
    for question in questions:
        parsed = parse_question(question["question"], companies)
        if not parsed.tickers:
            unresolved.append({"id": question["id"], "question": question["question"][:110]})
            continue
        for ticker in parsed.tickers:
            reason = evidence_for(question["question"], ticker, companies)
            support[reason or "NONE"] += 1
            if reason == "alias":
                alias_use[ticker] += 1
            if reason is None:
                unsupported.append({
                    "id": question["id"],
                    "ticker": ticker,
                    "resolved_name": getattr(companies.get(ticker), "name", "?"),
                    "method": parsed.entity_method,
                    "question": question["question"][:140],
                })

    fired = {ticker for _, ticker in ALIAS_SPECS if alias_use[ticker]}
    report = {
        "questions": len(questions),
        "support": dict(support),
        "unsupported_resolutions": len(unsupported),
        "unresolved_questions": len(unresolved),
        "alias_entries": len(ALIAS_SPECS),
        "alias_tickers_that_fire": len(fired),
        "examples": unsupported[:25],
        "unresolved": unresolved[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "questions", "support", "unsupported_resolutions", "unresolved_questions",
        "alias_entries", "alias_tickers_that_fire",
    )}, ensure_ascii=False, indent=2))
    for row in unsupported[:12]:
        print(f"  id={row['id']:>4} -> {row['ticker']} ({row['method']}) :: {row['question'][:90]}")


if __name__ == "__main__":
    main()
