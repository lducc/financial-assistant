"""Read and write the line-delimited JSON this project stores everything in.

Twenty-five copies of the same three-line comprehension were spread across
`scripts/`, six of them as an identically-worded `load_jsonl`. They agreed, but
only by accident: a blank trailing line, a BOM, or a partly written last line
from an interrupted run are each handled by whichever copy happened to be edited.
"""

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path | str) -> list[dict]:
    """Every record in the file, skipping blank lines."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path | str, records: Iterable[dict]) -> None:
    """Write records as one JSON object per line, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
