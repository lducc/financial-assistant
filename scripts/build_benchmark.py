#!/usr/bin/env python3
"""Merge the labelled sets into one benchmark file with a reproducible manifest.

Two label sets exist and they were built differently, which matters when reading
any number computed from them:

* `gold-150` was seeded from public submission 2333's candidate tables, then every
  operand was relocated in raw source. Verification was independent; discovery was
  not, so tables no retriever surfaced cannot appear in it.
* `v3` was discovered by folded substring search over raw OCR, with no retriever
  in the loop, and carries the branch computation for multi-hop questions.

Both are kept, both are labelled with their origin, and the manifest records the
corpus hash so a result can be tied to the data that produced it.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifinqa.evaluation_v2 import corpus_tree_hash, sha256_text
from evaluate_table_retrieval import connected_report_groups


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tier_weights(records: list[dict], corpus: Counter) -> dict[str, float]:
    """Per-tier weights that make the sample estimate the corpus.

    The sample deliberately over-represents hard questions, because that is where
    multi-hop retrieval fails and where 27 records could resolve nothing. Weighting
    the aggregate keeps the headline number an estimate of corpus performance while
    the per-tier numbers stay readable on their own.
    """
    sampled = Counter(record["tier"] for record in records)
    total_corpus, total_sample = sum(corpus.values()), len(records)
    return {
        tier: round((corpus[tier] / total_corpus) / (count / total_sample), 4)
        for tier, count in sampled.items() if count and corpus.get(tier)
    }


def normalize(record: dict, source: str, tier: str | None) -> dict:
    annotation = record["annotation"]
    return {
        "id": record["id"],
        "question": record["question"],
        "tier": record.get("tier") or tier or "unclassified",
        "source": source,
        "taxonomy": {
            "operation": record.get("taxonomy", {}).get("operation", "unlabelled"),
            "table_count": len(annotation["gold_tables"]),
            "report_count": len(set(annotation["gold_reports"])),
        },
        "provenance": record.get("provenance", {
            "discovery": "seeded from public submission 2333 candidates, then relocated in raw source",
            "independent_of_retriever": False,
        }),
        "annotation": {
            "status": "complete",
            "gold_reports": sorted(set(annotation["gold_reports"])),
            "gold_tables": sorted(set(annotation["gold_tables"])),
            "row_column_bindings": annotation.get("row_column_bindings", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--gold", type=Path, default=ROOT / "annotations" / "gold_150.jsonl")
    parser.add_argument("--v3", type=Path, action="append", default=[])
    parser.add_argument("--tiers", type=Path, default=ROOT / "data" / "derived" / "question_tiers.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations" / "benchmark.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "annotations" / "benchmark_manifest.json")
    args = parser.parse_args()

    v3_paths = args.v3 or sorted((ROOT / "annotations" / "v3").glob("labels_*.jsonl"))
    tiers = {record["id"]: record["tier"] for record in load_jsonl(args.tiers)} if args.tiers.exists() else {}

    records: dict[int, dict] = {}
    for record in load_jsonl(args.gold):
        records[record["id"]] = normalize(record, "gold_150", tiers.get(record["id"]))
    for path in v3_paths:
        for record in load_jsonl(path):
            if record["annotation"].get("status") == "complete":
                records[record["id"]] = normalize(record, "v3", tiers.get(record["id"]))

    ordered = [records[identifier] for identifier in sorted(records)]
    corpus_tiers = Counter(record["tier"] for record in load_jsonl(args.tiers)) if args.tiers.exists() else Counter()
    weights = tier_weights(ordered, corpus_tiers) if corpus_tiers else {}
    for record in ordered:
        record["weight"] = weights.get(record["tier"], 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered), encoding="utf-8"
    )

    # Questions sharing a report share a bootstrap cluster; the count of clusters,
    # not of questions, sets how much independent evidence the benchmark holds.
    groups = connected_report_groups([
        {"annotation": {"gold_reports": record["annotation"]["gold_reports"]}} for record in ordered
    ])
    cluster_by_report = {report: group[0] for group in groups for report in group}
    clusters = {record["annotation"]["gold_reports"][0] for record in ordered}
    seeded = [record for record in ordered if record["source"] == "gold_150"]

    manifest = {
        "records": len(ordered),
        "clusters": len({cluster_by_report[record["annotation"]["gold_reports"][0]] for record in ordered}),
        "distinct_first_reports": len(clusters),
        "gold_tables": sum(len(record["annotation"]["gold_tables"]) for record in ordered),
        "by_source": dict(Counter(record["source"] for record in ordered)),
        "by_tier": dict(Counter(record["tier"] for record in ordered)),
        "by_tier_and_source": {
            tier: dict(Counter(record["source"] for record in ordered if record["tier"] == tier))
            for tier in sorted({record["tier"] for record in ordered})
        },
        "table_count_distribution": dict(sorted(Counter(
            record["taxonomy"]["table_count"] for record in ordered
        ).items())),
        "seeded_share": round(len(seeded) / len(ordered), 4),
        "tier_weights": weights,
        "corpus_tree_hash": corpus_tree_hash(args.dataset_root),
        "benchmark_sha256": sha256_text(args.output.read_text(encoding="utf-8")),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
