#!/usr/bin/env python3
"""Propose complete labels for multi-operand questions, for accept/reject review.

Batch 0 produced 21 labels from 50 questions because every multi-hop question was
hunted by hand, and 141 of the 161 hard and intermediate questions still queued
name two or more line items. Reviewing a proposal is far cheaper than building
one, so this does the mechanical part: find each named line item in the gated
reports, read its value for each report, and lay the evidence out so a reviewer
can accept the label or reject it.

Discovery must stay independent of our own retriever, or the benchmark starts
measuring the retriever against itself. Rows are found by folded exact matching
of corpus row labels over every table in the gated reports — never by a ranked
list — the same method behind `scripts/search_evidence.py`.

The proposer never guesses. Where a line item resolves to several rows in one
report, or to none, the question is emitted as `needs_review` with the ambiguity
shown rather than a label invented to fill the gap.
"""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

from docs import load_companies
from vifinqa.answers import fold, line_item_phrases, parse_ocr_number
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import load_reports, report_tables

PERIOD_TOKENS = {"nam", "quy", "thang", "ngay", "ky", "tai", "cuoi", "dau"}
# Company names appear as row labels too — subsidiary lists, related-party tables —
# so the corpus lexicon contains them and a question naming its own issuer matches.
# "Tổng cộng nguồn vốn của Công ty Cổ phần Đầu tư Dịch vụ Hoàng Huy" then drags in
# every subsidiary-investment table as if it were evidence. "tổng công ty" is not
# in this list on purpose: it prefixes real line items.
ENTITY_PHRASES = ("cong ty", "ngan hang", "tap doan", "ctcp", "chi nhanh", "quy dau tu")


@lru_cache(maxsize=1)
def company_names(dataset_root: str) -> tuple[str, ...]:
    """Folded official company names, used to keep issuer text out of line items."""
    companies = load_companies(Path(dataset_root) / "code_stock.csv")
    return tuple(fold(company.name) for company in companies.values())


def named_line_items(question: str, limit: int = 3, dataset_root: str | None = None) -> list[str]:
    """Corpus row labels the question names, longest first, without period phrases.

    Subsidiary lists and related-party tables put company names in row labels, so
    the corpus lexicon contains them and a question naming its own issuer matches
    them. "Tổng cộng nguồn vốn của Công ty Cổ phần Đầu tư Dịch vụ Hoàng Huy" then
    drags every subsidiary-investment table in as evidence. Anything contained in
    a real company name is dropped — precise, unlike guessing at prefixes, which
    left "cổ phần" and "đầu tư" behind.
    """
    text = f" {fold(question)} "
    issuers = company_names(dataset_root or str(ROOT / "data" / "raw" / "vifinqa"))
    found: list[str] = []
    for label in line_item_phrases():
        if all(token in PERIOD_TOKENS or token.isdigit() for token in label.split()):
            continue
        if any(phrase in label for phrase in ENTITY_PHRASES):
            continue
        if any(label in issuer for issuer in issuers):
            continue
        # "cổ phiếu" is a fragment of "lãi cơ bản trên cổ phiếu" and matches 22 rows
        # a report on its own; keep only phrases no longer phrase contains.
        if label in text and not any(label in kept for kept in found):
            found.append(label)
    return found[:limit]


def specific(evidence: dict[str, list[dict]], ceiling: int) -> bool:
    """Whether a line item names one figure rather than a whole family of rows.

    A figure legitimately appears two or three times in a report — the statement,
    a note restating it, sometimes the cash-flow — and all of those are gold, so
    they are not ambiguity. Dozens of matches mean the phrase is generic.
    """
    return all(len(hits) <= ceiling for hits in evidence.values())


def row_matches(table, label: str) -> list[int]:
    """Rows whose leading cells carry this line item."""
    matches = []
    for index, row in enumerate(table.rows):
        if not row:
            continue
        if label in fold(" ".join(row[:2])):
            matches.append(index)
    return matches


def value_columns(table, row_index: int) -> list[tuple[int, float]]:
    """Numeric cells of a row, in column order, as candidate periods."""
    row = table.rows[row_index]
    found = []
    for column, cell in enumerate(row):
        if column == 0:
            continue
        value = parse_ocr_number(cell)
        # Account codes and note references are small integers; figures are not.
        if value is not None and abs(value) >= 1000:
            found.append((column, value))
    return found


def collect(report, label: str, keep: int = 3) -> list[dict]:
    """Rows carrying this line item, closest label first.

    "Doanh thu thuần" appears in the income statement, in its note, and in every
    segment breakdown, so counting matches to judge specificity rejects the most
    common — and most asked-about — figures in the corpus. Ranking by how much
    text surrounds the item picks the rows that are about it rather than the rows
    that merely mention it.
    """
    hits = []
    for table in report_tables(str(report.path), report.identity):
        for row_index in row_matches(table, label):
            columns = value_columns(table, row_index)
            if columns:
                row_label = " ".join(table.rows[row_index][:2]).strip()[:70]
                hits.append({
                    "table": table.table_id,
                    "row": row_index,
                    "row_label": row_label,
                    "title": table.title[:60],
                    "surplus": len(fold(row_label)) - len(label),
                    "columns": [{"column": column, "value": value, "raw": table.rows[row_index][column]}
                                for column, value in columns],
                })
    hits.sort(key=lambda hit: (hit["surplus"], hit["table"], hit["row"]))
    return hits[:keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--queue", type=Path, default=ROOT / "annotations" / "v3" / "queue.jsonl")
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument("--labelled", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "v3" / "proposals.jsonl")
    parser.add_argument("--tier", action="append", default=[], help="restrict to these tiers; repeatable")
    parser.add_argument("--max-rows", type=int, default=4, help="rows per report above which a phrase is generic")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    queue = load_jsonl(args.queue)
    tiers = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.tiers.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    done = set()
    for path in args.labelled:
        done.update(
            json.loads(line)["id"]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    wanted = set(args.tier) or {"hard", "intermediate", "medium", "easy"}
    selected = [item for item in queue if item["tier"] in wanted and item["id"] not in done][: args.limit]

    reports = {report.identity.report_id: report for report in load_reports(args.dataset_root)}
    status: Counter = Counter()
    proposals = []
    for number, item in enumerate(selected, 1):
        meta = tiers[item["id"]]
        labels = named_line_items(item["question"])
        evidence: dict[str, dict[str, list[dict]]] = defaultdict(dict)
        for label in labels:
            for report_id in meta["gated_report_ids"]:
                report = reports.get(report_id)
                if report is None:
                    continue
                hits = collect(report, label)
                if hits:
                    evidence[label][report_id] = hits
            report_tables.cache_clear()
        # Generic phrases match a whole family of rows and carry no information;
        # drop them rather than asking a reviewer to wade through them.
        evidence = {
            label: per_report for label, per_report in evidence.items()
            if specific(per_report, args.max_rows)
        }
        labels = [label for label in labels if label in evidence]
        covered = bool(labels) and all(
            len(evidence[label]) == len(meta["gated_report_ids"]) for label in labels
        )
        state = "proposed" if covered else "needs_review" if evidence else "no_match"
        status[state] += 1
        proposals.append({
            "id": item["id"],
            "question": item["question"],
            "tier": item["tier"],
            "status": state,
            "line_items": labels,
            "gated_reports": meta["gated_report_ids"],
            "evidence": {label: per_report for label, per_report in evidence.items()},
        })
        if args.progress_every and number % args.progress_every == 0:
            print(f"proposed {number}/{len(selected)}", flush=True, file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in proposals), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "questions": len(proposals),
        "status": dict(status),
        "tiers": dict(Counter(record["tier"] for record in proposals)),
        "mean_line_items": round(
            sum(len(record["line_items"]) for record in proposals) / max(1, len(proposals)), 2
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
