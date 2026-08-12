"""Merge reviewed v2 JSONL shards without accepting duplicate or missing IDs."""

import argparse
import json
from pathlib import Path
import tempfile


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite review file: {args.output}")
    expected = {record["id"] for record in records(args.queue)}
    merged: dict[int, dict] = {}
    for input_path in args.inputs:
        for record in records(input_path):
            identifier = record.get("id")
            if not isinstance(identifier, int) or identifier in merged:
                raise ValueError(f"duplicate or invalid review ID: {identifier}")
            merged[identifier] = record
    if set(merged) != expected:
        missing, extra = sorted(expected - set(merged)), sorted(set(merged) - expected)
        raise ValueError(f"review IDs do not match queue; missing={missing[:5]} extra={extra[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as handle:
        for identifier in sorted(merged):
            handle.write(json.dumps(merged[identifier], ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
