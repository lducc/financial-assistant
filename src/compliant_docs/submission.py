from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .catalog import Report


REQUIRED_KEYS = {
    "id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"
}


def make_row(source: dict, docs: list[str]) -> dict:
    return {
        "id": int(source["id"]),
        "question": source["question"],
        "answer": 0.0,
        "relevant_docs": docs,
        "relevant_tables": [],
        "evidence": [],
        "pandas_query": "",
    }


def validate(rows: list[dict], questions: list[dict], reports: dict[str, Report]) -> list[str]:
    errors = []
    if len(rows) != len(questions):
        errors.append(f"row_count={len(rows)} expected={len(questions)}")
    if [row.get("id") for row in rows] != [int(q["id"]) for q in questions]:
        errors.append("IDs/order differ from official questions")
    valid_docs = set(reports)
    for row in rows:
        missing = REQUIRED_KEYS - set(row)
        if missing:
            errors.append(f"id={row.get('id')} missing={sorted(missing)}")
        if row.get("question") != questions[int(row.get("id", 0)) - 1].get("question"):
            errors.append(f"id={row.get('id')} question mismatch")
        invalid = [doc for doc in row.get("relevant_docs", []) if doc not in valid_docs]
        if invalid:
            errors.append(f"id={row.get('id')} invalid_docs={invalid}")
        if len(row.get("relevant_docs", [])) != len(set(row.get("relevant_docs", []))):
            errors.append(f"id={row.get('id')} duplicate docs")
    return errors


def write_package(output_dir: Path, rows: list[dict]) -> Path:
    package = output_dir / "package"
    data_dir = package / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    submission_path = package / "submission.json"
    submission_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    zip_path = output_dir / "submission.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        archive.writestr("data/", "")
    return zip_path

