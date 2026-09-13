# Merge table_id -> report_id maps; conflicting entries are an error.
import argparse
import json
from pathlib import Path


def load_map(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    for table_id, report_id in value.items():
        if not isinstance(report_id, str) or not report_id:
            raise SystemExit(f"{path}: {table_id} has no report id")
    return value


def merge(sources: list[tuple[Path, dict[str, str]]]) -> tuple[dict[str, str], list[str]]:
    merged: dict[str, str] = {}
    origin: dict[str, Path] = {}
    conflicts = []
    for path, mapping in sources:
        for table_id, report_id in mapping.items():
            held = merged.get(table_id)
            if held is None:
                merged[table_id] = report_id
                origin[table_id] = path
            elif held != report_id:
                conflicts.append(
                    f"{table_id}: {origin[table_id]} says {held}, {path} says {report_id}"
                )
    return merged, conflicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=[], required=True,
                        help="a table_id -> report_id map; repeat per file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = []
    for path in args.input:
        sources.append((path, load_map(path)))

    merged, conflicts = merge(sources)
    if conflicts:
        print(f"conflicting report ids for {len(conflicts)} tables:")
        for line in conflicts[:20]:
            print(f"  {line}")
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    print(json.dumps({
        "inputs": {str(path): len(mapping) for path, mapping in sources},
        "tables": len(merged),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
