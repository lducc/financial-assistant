"""Materialize isolated question-only v2 annotation queues from frozen samples."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite annotation queues: {args.output_dir}")
    questions = {
        record["id"]: record["question"]
        for record in (json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, sample in manifest["splits"].items():
        rows = []
        for item in sample:
            identifier = item["id"]
            question = questions.get(identifier)
            if question is None:
                raise ValueError(f"manifest ID not found in question file: {identifier}")
            rows.append({"id": identifier, "question": question, "question_hash": item["question_hash"]})
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )


if __name__ == "__main__":
    main()
