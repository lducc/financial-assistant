"""E012 deterministic subset selection from a frozen E003 queue."""

from __future__ import annotations

from collections import defaultdict


def select_double_annotation(records: list[dict], size: int = 30) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["features"]["primary_stratum"]].append(record)
    for group in groups.values():
        group.sort(key=lambda record: record["id"])
    selected: list[dict] = []
    offset = 0
    strata = sorted(groups)
    while len(selected) < size:
        added = 0
        for stratum in strata:
            group = groups[stratum]
            if offset < len(group) and len(selected) < size:
                selected.append(group[offset])
                added += 1
        if not added:
            break
        offset += 1
    if len(selected) != size or len({record["id"] for record in selected}) != size:
        raise ValueError(f"Cannot select {size} distinct records from {len(records)}")
    return selected
