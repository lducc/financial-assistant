from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .normalize import content_tokens, fold


@dataclass(frozen=True)
class Report:
    doc_id: str
    ticker: str
    year: int
    scope: str
    source_path: str


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    folded_name: str
    content_name: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_companies(csv_path: Path) -> dict[str, Company]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ValueError(f"Empty company registry: {csv_path}")
    companies = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        ticker, name = row[0].strip().upper(), row[1].strip()
        companies[ticker] = Company(
            ticker=ticker,
            name=name,
            folded_name=fold(name),
            content_name=" ".join(content_tokens(name)),
        )
    return companies


def scope_from_id(doc_id: str) -> str:
    lowered = doc_id.casefold()
    if "_separate" in lowered:
        return "separate"
    if "_consolidated" in lowered or "_aggregated" in lowered:
        return "consolidated"
    return "unknown"


def load_reports(statements_root: Path) -> dict[str, Report]:
    reports = {}
    for path in sorted(statements_root.glob("**/*_extracted.txt")):
        doc_id = path.parent.name
        relative = path.relative_to(statements_root).as_posix()
        parts = relative.split("/")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        report = Report(
            doc_id=doc_id,
            ticker=parts[0].upper(),
            year=int(parts[1]),
            scope=scope_from_id(doc_id),
            source_path=relative,
        )
        reports[doc_id] = report
    return reports


def catalog_rows(reports: dict[str, Report]) -> list[dict]:
    return [asdict(reports[key]) for key in sorted(reports)]

