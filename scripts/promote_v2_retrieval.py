#!/usr/bin/env python3
"""Evaluate frozen v2 candidates and publish an immutable promotion decision."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from vifinqa.evaluation_v2 import (
    atomic_write_run, corpus_tree_hash, evaluate_v2_predictions, load_jsonl, paired_cluster_bootstrap,
    promotion_gate, summarize_v2_traces, validate_annotation_batch,
)

ROOT = Path(__file__).resolve().parents[1]


def read_run_traces(prediction_dir: Path) -> list[dict]:
    return load_jsonl(prediction_dir / "traces.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    args = parser.parse_args()
    annotations = load_jsonl(args.annotations)
    errors = validate_annotation_batch(annotations, load_jsonl(args.reviewer_a), load_jsonl(args.reviewer_b), args.raw_root)
    if errors:
        raise ValueError("annotation validation failed:\n" + "\n".join(errors[:20]))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline = evaluate_v2_predictions(
        annotations, load_jsonl(args.baseline_dir / "predictions.jsonl"), manifest, split="development",
    )
    baseline_summary = summarize_v2_traces(baseline, weighted=False)
    candidates, eligible = {}, []
    for candidate_dir in args.candidate_dir:
        run = json.loads((candidate_dir / "run.json").read_text(encoding="utf-8"))
        mode = run["mode"]
        candidate = evaluate_v2_predictions(
            annotations, load_jsonl(candidate_dir / "predictions.jsonl"), manifest, split="development",
        )
        summary = summarize_v2_traces(candidate, weighted=False)
        bootstrap = paired_cluster_bootstrap(baseline, candidate, iterations=args.bootstrap_iterations)
        gate = promotion_gate(baseline_summary, summary, bootstrap, read_run_traces(candidate_dir))
        candidates[mode] = {"summary": summary, "paired_bootstrap": bootstrap, "gate": gate, "traces": candidate}
        if gate["passed"]:
            eligible.append(mode)
    selected = sorted(eligible, key=lambda mode: (-candidates[mode]["paired_bootstrap"]["slot_recall"]["delta"], mode))[0] if eligible else "baseline"
    decision = {
        "selected_mode": selected, "control": "baseline", "eligible_modes": sorted(eligible),
        "rule": "slot-recall@5 paired-bootstrap ci95 lower bound > 0; non-negative MRR/nDCG; no fallback; valid source bindings/top-5; p95 latency <= 125% baseline",
    }
    atomic_write_run(args.output_dir, {
        "baseline_traces.json": baseline, "baseline_summary.json": baseline_summary,
        "candidates.json": candidates, "decision.json": decision,
        "provenance.json": {
            "command": [sys.executable, *sys.argv],
            "code_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, check=True, capture_output=True).stdout.strip(),
            "working_diff": subprocess.run(["git", "diff", "--no-ext-diff"], cwd=ROOT, text=True, check=True, capture_output=True).stdout,
            "corpus_tree_hash": corpus_tree_hash(args.raw_root),
            "candidate_modes": ["baseline", *[json.loads((directory / "run.json").read_text(encoding="utf-8"))["mode"] for directory in args.candidate_dir]],
        },
    }, {
        "annotations": args.annotations, "reviewer_a": args.reviewer_a, "reviewer_b": args.reviewer_b,
        "manifest": args.manifest, "baseline_predictions": args.baseline_dir / "predictions.jsonl",
        **{f"candidate_{index}": directory / "predictions.jsonl" for index, directory in enumerate(args.candidate_dir)},
    })
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
