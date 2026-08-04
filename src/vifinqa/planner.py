"""E009: conservative, inspectable operation templates for Vietnamese finance QA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata


OPERATIONS = frozenset({
    "lookup", "difference", "growth_or_change", "ratio_or_percent",
    "average", "selector", "aggregate",
})


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9%]+", " ", plain.replace("đ", "d")).strip()


@dataclass(frozen=True)
class Template:
    operation: str
    cues: tuple[str, ...]
    operand_roles: tuple[str, ...]


def classify_operation(question: str) -> Template:
    """Return a computation shape, deliberately not a semantic row mapping."""

    text = normalize(question)
    def matched(cues: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(cue for cue in cues if cue in text)

    checks = (
        ("growth_or_change", ("tang truong", "tang bao nhieu", "giam bao nhieu", "bien dong", "so voi"), ("current", "comparison")),
        ("ratio_or_percent", ("ty le", "phan tram", "roe", "roa", "bien loi nhuan", "he so"), ("numerator", "denominator")),
        ("average", ("trung binh", "binh quan"), ("items",)),
        ("selector", ("cao nhat", "lon nhat", "thap nhat", "nho nhat"), ("candidate_values", "selection_rule")),
        ("difference", ("chenh lech", "khac biet", "be hon", "lon hon"), ("left", "right")),
        # Do not use bare 'tong': it is commonly part of a metric name, e.g. tong tai san.
        ("aggregate", ("tong cong", "cong don", "cong lai", "tong cua cac"), ("items",)),
    )
    for operation, cues, roles in checks:
        hits = matched(cues)
        if hits:
            return Template(operation, hits, roles)
    return Template("lookup", (), ("value",))


def output_unit_hint(question: str) -> str:
    text = normalize(question)
    if "%" in question or "phan tram" in text or "ty le" in text:
        return "percent_points"
    if "ty dong" in text or "ti dong" in text:
        return "billion_vnd"
    if "trieu dong" in text:
        return "million_vnd"
    if "nghin dong" in text:
        return "thousand_vnd"
    if "co phieu" in text:
        return "shares_or_count"
    return "unspecified"


def build_plan(question: str, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    template = classify_operation(question)
    plan = {
        "planner": "rules_v1",
        "question": question,
        "operation": template.operation,
        "operation_cues": list(template.cues),
        "operand_roles": list(template.operand_roles),
        "constraints": {
            "tickers": list(metadata.get("tickers", [])),
            "years": list(metadata.get("years", [])),
            "scope": metadata.get("scope"),
        },
        "output_unit_hint": output_unit_hint(question),
        "needs_row_bindings": True,
        "warning": "Template is structural only; it does not identify an evidence row or prove semantic correctness.",
    }
    if plan["operation"] not in OPERATIONS or not plan["operand_roles"]:
        raise ValueError(f"Invalid template: {asdict(template)}")
    return plan
