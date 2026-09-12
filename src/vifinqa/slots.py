import json
import math
from pathlib import Path

from .retrieval import tokenize


PERIOD_WORDS = {"nam", "quy", "thang", "ngay", "ky", "tai", "cuoi", "dau"}
ENTITY_PHRASES = ("cong ty", "ngan hang", "tap doan", "ctcp", "chi nhanh", "quy dau tu")


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def load_line_items(path: Path) -> tuple[str, ...]:
    labels = json.loads(path.read_text(encoding="utf-8"))
    normalized = {normalize(label) for label in labels} - {""}
    return tuple(sorted(normalized, key=lambda label: (-len(label), label)))


def named_line_items(question: str, labels: tuple[str, ...]) -> list[str]:
    """The longest corpus line items the question names, at most three."""
    text = f" {normalize(question)} "
    found = []
    for label in labels:
        words = label.split()
        if not words or all(word in PERIOD_WORDS or word.isdigit() for word in words):
            continue
        if any(phrase in label for phrase in ENTITY_PHRASES):
            continue
        if f" {label} " not in text:
            continue
        if any(label in longer for longer in found):
            continue
        found.append(label)
        if len(found) == 3:
            break
    return found


def _logit(probability: float) -> float:
    probability = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(probability / (1.0 - probability))


def append_confidence_item_tables(
    record: dict,
    starting_order: list[str],
    candidate_order: list[str],
    line_items: list[str],
    scores: dict[str, float],
    logit_gap: float,
) -> list[str]:
    """Append scored tables that carry a named item, at most three per report."""
    candidates = {candidate["table_id"]: candidate for candidate in record["candidates"]}
    selected = list(dict.fromkeys(starting_order))
    seen = set(selected)
    reports = set(record.get("selected_docs", []))
    counts = {report_id: sum(
        table_id.rsplit("|", 1)[0] == report_id for table_id in selected
    ) for report_id in reports}
    scored = {table_id: float(value) for table_id, value in scores.items()
              if table_id in candidates}
    top_logit = max((_logit(value) for value in scored.values()), default=0.0)
    for table_id in candidate_order:
        if table_id in seen or table_id not in scored:
            continue
        report_id = table_id.rsplit("|", 1)[0]
        if report_id not in reports or counts[report_id] >= 3:
            continue
        if top_logit - _logit(scored[table_id]) > logit_gap:
            continue
        text = f" {normalize(candidates[table_id].get('text', ''))} "
        if line_items and not any(f" {item} " in text for item in line_items):
            continue
        selected.append(table_id)
        seen.add(table_id)
        counts[report_id] += 1
    return selected
