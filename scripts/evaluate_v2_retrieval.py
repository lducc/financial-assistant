"""Run immutable slot-aware evaluation on frozen v2 annotations."""

import argparse
import json
from pathlib import Path

from vifinqa.evaluation_v2 import (
    atomic_write_run,
    evaluate_v2_predictions,
    load_jsonl,
    paired_cluster_bootstrap,
    summarize_v2_traces,
    validate_annotation_batch,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("sealed_core", "sealed_ood", "development"), default="sealed_core")
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    reviewer_a = load_jsonl(args.reviewer_a)
    reviewer_b = load_jsonl(args.reviewer_b)
    errors = validate_annotation_batch(annotations, reviewer_a, reviewer_b, args.raw_root)
    if errors:
        raise ValueError("annotation validation failed:\n" + "\n".join(errors[:20]))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline = evaluate_v2_predictions(annotations, load_jsonl(args.baseline), manifest, split=args.split)
    candidate = evaluate_v2_predictions(annotations, load_jsonl(args.candidate), manifest, split=args.split)
    weighted = args.split == "sealed_core"
    artifacts = {
        "baseline_traces.json": baseline,
        "candidate_traces.json": candidate,
        "summary.json": {
            "split": args.split,
            "baseline": summarize_v2_traces(baseline, weighted=weighted),
            "candidate": summarize_v2_traces(candidate, weighted=weighted),
        },
        "paired_bootstrap.json": paired_cluster_bootstrap(
            baseline, candidate, iterations=args.bootstrap_iterations,
        ),
    }
    atomic_write_run(args.output_dir, artifacts, {
        "annotations": args.annotations,
        "reviewer_a": args.reviewer_a,
        "reviewer_b": args.reviewer_b,
        "manifest": args.manifest,
        "baseline": args.baseline,
        "candidate": args.candidate,
    })
    print(args.output_dir)


if __name__ == "__main__":
    main()
