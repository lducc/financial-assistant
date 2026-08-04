#!/usr/bin/env python3
"""Self-contained E004 fixtures for the static submission validator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_submission_v2 import validate


def package(root: Path, *, query: str = "df1.iloc[0, 0]", csv_exists: bool = True, nested: bool = False) -> Path:
    destination = root / "nested" if nested else root
    destination.mkdir(parents=True, exist_ok=True)
    data = destination / "data"
    data.mkdir(exist_ok=True)
    if csv_exists:
        (data / "R_10.csv").write_text("value\n1\n", encoding="utf-8")
    row = [{
        "id": 1, "question": "q", "answer": 1.0,
        "relevant_docs": ["R"], "relevant_tables": ["R|10"],
        "evidence": [{"variable": "df1", "csv_path": "data/R_10.csv"}],
        "pandas_query": query,
    }]
    (destination / "submission.json").write_text(json.dumps(row), encoding="utf-8")
    return destination


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        expected, catalog = {1}, {"R|10"}
        valid = package(root / "valid")
        assert not validate(valid, expected, catalog)
        constant = package(root / "constant", query="result = 1")
        assert any("constant" in error for error in validate(constant, expected, catalog))
        missing = package(root / "missing", csv_exists=False)
        assert any("missing evidence" in error for error in validate(missing, expected, catalog))
        wrong_table = package(root / "wrong")
        row_path = wrong_table / "submission.json"
        row = json.loads(row_path.read_text(encoding="utf-8"))
        row[0]["relevant_tables"] = ["R|11"]
        row_path.write_text(json.dumps(row), encoding="utf-8")
        assert any("absent from the supplied catalog" in error for error in validate(wrong_table, expected, catalog))
        nested = package(root / "zip_source", nested=True)
        archive_path = root / "nested.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in nested.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(root / "zip_source"))
        with tempfile.TemporaryDirectory() as unpacked:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(unpacked)
            assert any("top-level JSON" in error for error in validate(Path(unpacked), expected, catalog))
    print("E004 fixtures passed")


if __name__ == "__main__":
    main()

