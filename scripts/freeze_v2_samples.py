"""Freeze v2 development and sealed sampling queues from question-only inputs."""

import argparse
import inspect
import json
from pathlib import Path

from vifinqa.evaluation_v2 import (
    SAMPLE_SEED, build_frame, corpus_tree_hash, hash_file, manifest,
    read_legacy_ids, read_question_file, sample_splits, sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]
from docs import load_companies, parse_question


def question_metadata(records: list[dict], companies: dict) -> dict[int, tuple[str | None, int | None, str]]:
    """Use permitted question and company metadata; never use labels or predictions."""
    result = {}
    for record in records:
        parsed = parse_question(record["question"], companies)
        tickers = parsed.tickers or parsed.candidate_tickers
        result[record["id"]] = (
            tickers[0] if tickers else None,
            parsed.years[0] if parsed.years else None,
            parsed.scope or "unknown",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--legacy-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--expected-development-eligible", type=int, default=455)
    parser.add_argument(
        "--expected-held-out-tickers",
        default="HUT,HAG,VIC,PLX,BVH,HPX,ASM,ACV",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    questions = read_question_file(args.questions)
    frame = build_frame(questions, question_metadata(questions, load_companies(args.companies)))
    legacy_ids = read_legacy_ids(args.legacy_ids)
    splits, diagnostics = sample_splits(frame, legacy_ids, seed=args.seed, return_metadata=True)
    expected_tickers = [ticker for ticker in args.expected_held_out_tickers.split(",") if ticker]
    if diagnostics["development_eligible"] != args.expected_development_eligible:
        raise ValueError(
            f"development eligibility mismatch: expected {args.expected_development_eligible}, got {diagnostics['development_eligible']}"
        )
    if diagnostics["held_out_tickers"] != expected_tickers:
        raise ValueError(
            f"held-out tickers mismatch: expected {expected_tickers}, got {diagnostics['held_out_tickers']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest(
        splits,
        seed=args.seed,
        population_size=len(frame) - len(legacy_ids),
        input_hashes={
            "questions": hash_file(args.questions),
            "companies": hash_file(args.companies),
            "legacy_ids": hash_file(args.legacy_ids),
            "corpus_tree": corpus_tree_hash(args.questions.parents[1]),
            "sampler_code": hash_file(Path(__file__)),
        },
        held_out_tickers=diagnostics["held_out_tickers"],
        parser_hash=sha256_text(inspect.getsource(question_metadata)),
    ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
