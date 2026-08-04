#!/usr/bin/env python3
"""Fail-fast structural validator for a ViFinQA submission directory or ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile


VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REQUIRED = {"id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"}


def validate(root: Path, expected_ids: set[int] | None) -> list[str]:
    errors: list[str] = []
    json_files = list(root.glob("*.json"))
    if len(json_files) != 1:
        return [f"Expected exactly one top-level JSON file, found {len(json_files)}"]
    try:
        rows = json.loads(json_files[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON: {exc}"]
    if not isinstance(rows, list):
        return ["Submission JSON must contain a list"]

    seen_ids: set[int] = set()
    for index, row in enumerate(rows):
        prefix = f"record {index}"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"{prefix}: missing keys {sorted(missing)}")
            continue
        identifier = row["id"]
        if not isinstance(identifier, int) or identifier in seen_ids:
            errors.append(f"{prefix}: ID must be a unique integer")
        seen_ids.add(identifier)
        if not isinstance(row["answer"], (int, float)) or isinstance(row["answer"], bool):
            errors.append(f"{prefix}: answer must be numeric")
        if not isinstance(row["relevant_docs"], list) or not all(isinstance(v, str) for v in row["relevant_docs"]):
            errors.append(f"{prefix}: relevant_docs must be a string list")
        if not isinstance(row["relevant_tables"], list) or not all(isinstance(v, str) and "|" in v for v in row["relevant_tables"]):
            errors.append(f"{prefix}: relevant_tables must be a list of report|position strings")
        variables: set[str] = set()
        if not isinstance(row["evidence"], list):
            errors.append(f"{prefix}: evidence must be a list")
        else:
            for evidence in row["evidence"]:
                if not isinstance(evidence, dict):
                    errors.append(f"{prefix}: evidence entry must be an object")
                    continue
                variable, csv_path = evidence.get("variable"), evidence.get("csv_path")
                if not isinstance(variable, str) or not VARIABLE_RE.fullmatch(variable) or variable in variables:
                    errors.append(f"{prefix}: evidence variables must be unique Python identifiers")
                variables.add(variable)
                if not isinstance(csv_path, str) or not csv_path.startswith("data/"):
                    errors.append(f"{prefix}: csv_path must be a relative data/ path")
                elif not (root / csv_path).is_file():
                    errors.append(f"{prefix}: missing evidence file {csv_path}")
        if not isinstance(row["pandas_query"], str) or not row["pandas_query"].strip():
            errors.append(f"{prefix}: pandas_query must be non-empty text")

    if expected_ids is not None and seen_ids != expected_ids:
        errors.append("Submission IDs do not exactly match the question IDs")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--questions", type=Path)
    args = parser.parse_args()
    expected_ids = None
    if args.questions:
        expected_ids = {json.loads(line)["id"] for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()}
    with tempfile.TemporaryDirectory() as temporary:
        root = args.submission
        if args.submission.suffix.lower() == ".zip":
            with zipfile.ZipFile(args.submission) as archive:
                archive.extractall(temporary)
            root = Path(temporary)
        errors = validate(root, expected_ids)
    if errors:
        print("INVALID")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("VALID")


if __name__ == "__main__":
    main()

