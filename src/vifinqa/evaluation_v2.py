"""Deterministic, label-free sample construction for the v2 retrieval study."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
import unicodedata
from typing import Iterable

from .tables import extract_rows_at_line
from .retrieval import load_reports

from .jsonl import load_jsonl


SAMPLE_SEED = 20260812
ENTITY_RE = re.compile(r"\(([A-Za-z0-9]{1,12})\)")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\d+(?:[.,/]\d+)*")
PAREN_CODE_RE = re.compile(r"\([^)]{1,12}\)")


@dataclass(frozen=True)
class QuestionFrame:
    """The sampler's permitted, question-only view of an item."""

    id: int
    question: str
    question_hash: str
    template_hash: str
    ticker: str | None
    year: int | None
    scope: str

    @property
    def report_key(self) -> tuple[str | None, int | None, str]:
        return self.ticker, self.year, self.scope


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_tree_hash(dataset_root: Path) -> str:
    """Hash every raw report path and byte stream in deterministic order."""
    digest = hashlib.sha256()
    for report in load_reports(dataset_root):
        digest.update(report.identity.source_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(report.path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_template(question: str) -> str:
    """Normalize entity, year, and numeric variation before template hashing."""
    text = unicodedata.normalize("NFKD", question.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = PAREN_CODE_RE.sub(" <entity> ", text)
    text = YEAR_RE.sub(" <year> ", text)
    text = NUMBER_RE.sub(" <num> ", text)
    text = re.sub(r"[^a-z0-9<>%]+", " ", text)
    return " ".join(text.split())


def question_metadata(question: str) -> tuple[str | None, int | None, str]:
    """Extract only question-visible grouping keys; unknown values remain unknown."""
    entity = ENTITY_RE.search(question)
    year = YEAR_RE.search(question)
    lowered = unicodedata.normalize("NFKD", question.lower())
    lowered = "".join(char for char in lowered if not unicodedata.combining(char))
    scope = "separate" if "cong ty me" in lowered else "unknown"
    return (
        entity.group(1).upper() if entity else None,
        int(year.group(0)) if year else None,
        scope,
    )


def build_frame(
    records: Iterable[dict], metadata_by_id: dict[int, tuple[str | None, int | None, str]] | None = None,
) -> list[QuestionFrame]:
    """Build and validate the sampler frame without accepting labels or predictions."""
    frame: list[QuestionFrame] = []
    ids: set[int] = set()
    for record in records:
        if set(record) - {"id", "question"}:
            raise ValueError("sampler accepts only question id and text")
        item_id, question = record.get("id"), record.get("question")
        if isinstance(item_id, bool) or not isinstance(item_id, int) or not isinstance(question, str):
            raise ValueError("each question must contain integer id and text")
        if item_id in ids:
            raise ValueError(f"duplicate question id: {item_id}")
        ids.add(item_id)
        ticker, year, scope = (metadata_by_id or {}).get(item_id, question_metadata(question))
        normalized = normalize_template(question)
        frame.append(QuestionFrame(
            id=item_id,
            question=question,
            question_hash=sha256_text(question),
            template_hash=sha256_text(normalized),
            ticker=ticker,
            year=year,
            scope=scope,
        ))
    return sorted(frame, key=lambda item: item.id)


def read_question_file(path: Path) -> list[dict]:
    records = load_jsonl(path)
    if any(not isinstance(record, dict) or set(record) != {"id", "question"} for record in records):
        raise ValueError("question input must contain only id and question fields")
    return records


def read_legacy_ids(path: Path) -> set[int]:
    """Read an ID-only JSONL/text file and reject annotation-bearing input."""
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line) if line.lstrip().startswith("{") else int(line)
        if isinstance(value, dict):
            if set(value) != {"id"}:
                raise ValueError("legacy exclusion input must contain IDs only")
            value = value["id"]
        if isinstance(value, bool) or not isinstance(value, int) or value in ids:
            raise ValueError("legacy exclusion IDs must be unique integers")
        ids.add(value)
    return ids


def _sample(rng: random.Random, population: list[QuestionFrame], count: int) -> list[QuestionFrame]:
    if len(population) < count:
        raise ValueError(f"sample requires {count} questions; only {len(population)} available")
    return sorted(rng.sample(population, count), key=lambda item: item.id)


def sample_splits(
    frame: list[QuestionFrame],
    legacy_ids: set[int],
    *,
    development_count: int = 150,
    core_count: int = 240,
    ood_count: int = 60,
    seed: int = SAMPLE_SEED,
    return_metadata: bool = False,
) -> dict[str, list[QuestionFrame]] | tuple[dict[str, list[QuestionFrame]], dict]:
    """Create disjoint core/OOD/development queues from a frozen question frame."""
    by_id = {item.id: item for item in frame}
    if len(by_id) != len(frame) or not legacy_ids <= by_id.keys():
        raise ValueError("legacy IDs must be unique and present in the question frame")
    unseen = [item for item in frame if item.id not in legacy_ids]
    rng = random.Random(seed)
    core = _sample(rng, unseen, core_count)
    core_ids = {item.id for item in core}

    ticker_counts = Counter(item.ticker for item in unseen if item.ticker)
    eligible_tickers = sorted(ticker for ticker, count in ticker_counts.items() if count >= 5)
    rng.shuffle(eligible_tickers)
    held_out: list[str] = []
    ood_pool: list[QuestionFrame] = []
    for ticker in eligible_tickers:
        held_out.append(ticker)
        ood_pool.extend(item for item in unseen if item.id not in core_ids and item.ticker == ticker)
        if len(ood_pool) >= ood_count:
            break
    if len(ood_pool) < ood_count:
        raise ValueError("insufficient primary-entity OOD pool")
    ood = _sample(rng, ood_pool, ood_count)
    sealed = core_ids | {item.id for item in ood}
    sealed_templates = {item.template_hash for item in core + ood}
    sealed_keys = {item.report_key for item in core + ood}
    development_pool = [
        item for item in frame
        if item.id not in legacy_ids | sealed
        and item.ticker not in held_out
        and item.template_hash not in sealed_templates
        and item.report_key not in sealed_keys
    ]
    development = _sample(rng, development_pool, development_count)
    splits = {"development": development, "sealed_core": core, "sealed_ood": ood}
    metadata = {"held_out_tickers": held_out, "development_eligible": len(development_pool)}
    return (splits, metadata) if return_metadata else splits


def manifest(
    splits: dict[str, list[QuestionFrame]],
    *,
    seed: int = SAMPLE_SEED,
    population_size: int = 862,
    input_hashes: dict[str, str] | None = None,
    held_out_tickers: list[str] | None = None,
    parser_hash: str | None = None,
) -> dict:
    core_size = len(splits["sealed_core"])
    result = {
        "schema_version": "2.0",
        "seed": seed,
        "parser_hash": parser_hash or sha256_text(inspect.getsource(question_metadata)),
        "input_hashes": input_hashes or {},
        "held_out_tickers": held_out_tickers or [],
        "population_size": population_size,
        "splits": {},
    }
    for split, items in splits.items():
        result["splits"][split] = [
            {
                "id": item.id,
                "question_hash": item.question_hash,
                "template_hash": item.template_hash,
                "ticker": item.ticker,
                "year": item.year,
                "scope": item.scope,
                "report_key": list(item.report_key),
                "inclusion_probability": core_size / population_size if split == "sealed_core" else None,
                "stratum": "representative_core" if split == "sealed_core" else "primary_entity_ood" if split == "sealed_ood" else "development",
                "selection_reason": split,
            }
            for item in items
        ]
    return result


def validate_v2_source_bindings(record: dict, raw_root: Path, reports: dict[str, object] | None = None) -> list[str]:
    """Validate every v2 cell coordinate and OCR value against raw source tables."""
    reports = reports or {report.identity.report_id: report for report in load_reports(raw_root)}
    errors: list[str] = []
    for slot_index, slot in enumerate(record.get("slots", [])):
        if not isinstance(slot, dict):
            continue
        for alternative_index, alternative in enumerate(slot.get("alternatives", [])):
            if not isinstance(alternative, dict):
                continue
            for cell_index, cell in enumerate(alternative.get("cells", [])):
                prefix = f"slots[{slot_index}].alternatives[{alternative_index}].cells[{cell_index}]"
                if not isinstance(cell, dict):
                    continue
                table_id = cell.get("table")
                match = re.fullmatch(r"([^|\s]+)\|([1-9]\d*)", str(table_id))
                if not match:
                    errors.append(f"{prefix} invalid table ID")
                    continue
                report = reports.get(match.group(1))
                if report is None:
                    errors.append(f"{prefix} missing source report")
                    continue
                try:
                    rows = extract_rows_at_line(report.path.read_text(encoding="utf-8"), int(match.group(2)))
                except (OSError, KeyError, ValueError) as error:
                    errors.append(f"{prefix} cannot extract source table: {error}")
                    continue
                row, column = cell.get("row"), cell.get("column")
                if not isinstance(row, int) or isinstance(row, bool) or not isinstance(column, int) or isinstance(column, bool):
                    errors.append(f"{prefix} row and column must be integers")
                elif row < 0 or column < 0 or row >= len(rows) or column >= len(rows[row]):
                    errors.append(f"{prefix} cell out of range")
                elif rows[row][column] != cell.get("raw"):
                    errors.append(f"{prefix} raw cell mismatch")
    return errors




def index_records(records: list[dict], name: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for record in records:
        identifier = record.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier in indexed:
            raise ValueError(f"{name} has duplicate or invalid IDs")
        indexed[identifier] = record
    return indexed


def validate_annotation_batch(
    adjudications: list[dict], reviewer_a: list[dict], reviewer_b: list[dict], raw_root: Path,
) -> list[str]:
    """Validate exact reviewer/adjudication ID coverage before any evaluation."""
    adjudicated = index_records(adjudications, "adjudications")
    reviewers_a = index_records(reviewer_a, "reviewer A")
    reviewers_b = index_records(reviewer_b, "reviewer B")
    if set(adjudicated) != set(reviewers_a) or set(adjudicated) != set(reviewers_b):
        return ["reviewer and adjudication IDs must match exactly"]
    reports = {report.identity.report_id: report for report in load_reports(raw_root)}
    errors: list[str] = []
    for identifier, record in adjudicated.items():
        errors.extend(f"id={identifier}: {error}" for error in validate_adjudication(record, reviewers_a[identifier], reviewers_b[identifier]))
        errors.extend(f"id={identifier}: {error}" for error in validate_v2_source_bindings(record, raw_root, reports))
        errors.extend(f"id={identifier}: reviewer_a: {error}" for error in validate_v2_source_bindings(reviewers_a[identifier], raw_root, reports))
        errors.extend(f"id={identifier}: reviewer_b: {error}" for error in validate_v2_source_bindings(reviewers_b[identifier], raw_root, reports))
    return errors


def _required_tables(alternative: dict) -> set[str]:
    tables = set(alternative.get("tables", []))
    tables.update(cell.get("table") for cell in alternative.get("cells", []) if isinstance(cell, dict))
    return {table for table in tables if isinstance(table, str)}


def validate_v2_annotation(record: dict) -> list[str]:
    """Validate the v2 evidence contract without applying retrieval predictions."""
    errors: list[str] = []
    if record.get("schema_version") != "2.0":
        errors.append("schema_version must be 2.0")
    if isinstance(record.get("id"), bool) or not isinstance(record.get("id"), int):
        errors.append("id must be an integer")
    if not isinstance(record.get("question_hash"), str) or not record["question_hash"]:
        errors.append("question_hash is required")
    if record.get("complete") is not True:
        errors.append("annotation must be complete")
    if not isinstance(record.get("operation"), str) or not record["operation"]:
        errors.append("operation is required")
    confidence = record.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append("confidence must be a number from 0 through 1")
    if not isinstance(record.get("reviewer_protocol_hash"), str) or not record["reviewer_protocol_hash"]:
        errors.append("reviewer_protocol_hash is required")
    slots = record.get("slots")
    if not isinstance(slots, list) or not slots:
        return errors + ["slots must be a nonempty list"]
    seen_slots: set[str] = set()
    for slot_index, slot in enumerate(slots):
        prefix = f"slots[{slot_index}]"
        if not isinstance(slot, dict):
            errors.append(f"{prefix} must be an object")
            continue
        slot_id = slot.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id or slot_id in seen_slots:
            errors.append(f"{prefix}.slot_id must be unique")
        seen_slots.add(slot_id)
        for field in ("entity", "report_year", "scope", "metric", "operand_role"):
            if field not in slot:
                errors.append(f"{prefix}.{field} is required")
        if slot.get("report_year") != "unknown" and (
            isinstance(slot.get("report_year"), bool) or not isinstance(slot.get("report_year"), int)
        ):
            errors.append(f"{prefix}.report_year must be an integer or unknown")
        for field in ("entity", "scope", "metric", "operand_role"):
            if not isinstance(slot.get(field), str) or not slot[field]:
                errors.append(f"{prefix}.{field} must be a nonempty string or unknown")
        alternatives = slot.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            errors.append(f"{prefix}.alternatives must be nonempty")
            continue
        for alternative_index, alternative in enumerate(alternatives):
            alt_prefix = f"{prefix}.alternatives[{alternative_index}]"
            if not isinstance(alternative, dict):
                errors.append(f"{alt_prefix} must be an object")
                continue
            if not isinstance(alternative.get("reports"), list) or not isinstance(alternative.get("tables"), list):
                errors.append(f"{alt_prefix} reports and tables must be lists")
            tables = _required_tables(alternative)
            if not tables:
                errors.append(f"{alt_prefix} must bind at least one table")
            for cell_index, cell in enumerate(alternative.get("cells", [])):
                cell_prefix = f"{alt_prefix}.cells[{cell_index}]"
                if not isinstance(cell, dict):
                    errors.append(f"{cell_prefix} must be an object")
                    continue
                for field in ("table", "row", "column", "raw", "period", "unit"):
                    if field not in cell:
                        errors.append(f"{cell_prefix}.{field} is required")
                if isinstance(cell.get("row"), bool) or not isinstance(cell.get("row"), int) or cell["row"] < 0:
                    errors.append(f"{cell_prefix}.row must be a nonnegative integer")
                if isinstance(cell.get("column"), bool) or not isinstance(cell.get("column"), int) or cell["column"] < 0:
                    errors.append(f"{cell_prefix}.column must be a nonnegative integer")
    return errors


def validate_adjudication(record: dict, reviewer_a: dict | None, reviewer_b: dict | None) -> list[str]:
    """Block evaluation unless two independent reviews and a completed adjudication exist."""
    errors = validate_v2_annotation(record)
    if reviewer_a is None or reviewer_b is None:
        return errors + ["both independent reviewer files are required"]
    for name, review in (("reviewer_a", reviewer_a), ("reviewer_b", reviewer_b)):
        review_errors = validate_v2_annotation(review)
        errors.extend(f"{name}: {error}" for error in review_errors)
        if review.get("id") != record.get("id") or review.get("question_hash") != record.get("question_hash"):
            errors.append(f"{name} does not match adjudicated question")
    if reviewer_a.get("reviewer_protocol_hash") == reviewer_b.get("reviewer_protocol_hash"):
        errors.append("reviewer protocol hashes must identify independent reviews")
    if not isinstance(record.get("adjudication"), dict) or record["adjudication"].get("resolved") is not True:
        errors.append("adjudication must explicitly resolve disagreements")
    return errors


def _discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _minimum_required_tables(slots: list[dict]) -> int:
    """Find fewest unique tables from one complete alternative per slot."""
    unions = [set()]
    for slot in slots:
        alternatives = [_required_tables(alternative) for alternative in slot["alternatives"]]
        unions = [previous | required for previous in unions for required in alternatives]
        unions.sort(key=len)
        retained: list[set[str]] = []
        for candidate in unions:
            if not any(existing <= candidate for existing in retained):
                retained.append(candidate)
        unions = retained
    return min(map(len, unions))


def score_slots(annotation: dict, ranked_tables: list[str], *, k: int = 5) -> dict[str, float | bool]:
    """Score slot-aware alternatives; incomplete mixtures never receive complete credit."""
    errors = validate_v2_annotation(annotation)
    if errors:
        raise ValueError("malformed annotation: " + "; ".join(errors))
    ranked = ranked_tables[:k]
    ranked_set = set(ranked)
    recalls: list[float] = []
    completes: list[bool] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    feasible = _minimum_required_tables(annotation["slots"]) <= k
    for slot in annotation["slots"]:
        alternatives = [_required_tables(alternative) for alternative in slot["alternatives"]]
        recalls.append(max(len(required & ranked_set) / len(required) for required in alternatives))
        completes.append(any(required <= ranked_set for required in alternatives))
        first_rank = min((rank + 1 for rank, table in enumerate(ranked) if any(table in required for required in alternatives)), default=None)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        alternative_ndcgs = []
        for required in alternatives:
            dcg = sum(_discount(rank) for rank, table in enumerate(ranked, 1) if table in required)
            ideal = sum(_discount(rank) for rank in range(1, min(len(required), k) + 1))
            alternative_ndcgs.append(dcg / ideal if ideal else 0.0)
        ndcgs.append(max(alternative_ndcgs))
    return {
        "slot_recall": sum(recalls) / len(recalls),
        "slot_complete": sum(completes) / len(completes),
        "question_recall": sum(recalls) / len(recalls),
        "question_complete": all(completes),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg": sum(ndcgs) / len(ndcgs),
        "feasible_coverage": feasible,
    }


METRICS = ("slot_recall", "slot_complete", "question_recall", "question_complete", "mrr", "ndcg")


def summarize_v2_traces(traces: list[dict], *, weighted: bool) -> dict:
    """Summarize sealed-core traces with manifest inclusion weights when requested."""
    if not traces:
        raise ValueError("cannot summarize an empty trace set")
    weights = [float(trace.get("weight", 1.0)) if weighted else 1.0 for trace in traces]
    total_weight = sum(weights)
    summary = {"records": len(traces), "weighted": weighted}
    for metric in METRICS:
        summary[metric] = sum(weight * float(trace["metrics"][metric]) for weight, trace in zip(weights, traces)) / total_weight
    latencies = sorted(float(trace.get("latency_ms", 0.0)) for trace in traces)
    summary["latency_ms"] = {
        "p50": latencies[(len(latencies) - 1) // 2],
        "p95": latencies[math.ceil(len(latencies) * 0.95) - 1],
    }
    summary["candidate_count"] = sum(float(trace.get("candidate_count", 0.0)) for trace in traces) / len(traces)
    summary["feasible_coverage"] = sum(float(trace["metrics"]["feasible_coverage"]) for trace in traces) / len(traces)
    return summary


def paired_cluster_bootstrap(
    baseline: list[dict], candidate: list[dict], *, iterations: int = 20_000, seed: int = SAMPLE_SEED,
) -> dict[str, dict[str, float]]:
    """Paired group bootstrap for frozen report/template clusters."""
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    baseline_by_id, candidate_by_id = index_records(baseline, "baseline traces"), index_records(candidate, "candidate traces")
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("paired traces must contain identical IDs")
    clusters: dict[str, list[int]] = {}
    for identifier, trace in baseline_by_id.items():
        cluster = str(trace.get("cluster_id", identifier))
        clusters.setdefault(cluster, []).append(identifier)
    rng = random.Random(seed)
    deltas = {metric: [] for metric in METRICS}
    cluster_items = list(clusters.values())
    for _ in range(iterations):
        sampled = [identifier for _ in cluster_items for identifier in rng.choice(cluster_items)]
        for metric in METRICS:
            deltas[metric].append(sum(
                float(candidate_by_id[identifier]["metrics"][metric]) - float(baseline_by_id[identifier]["metrics"][metric])
                for identifier in sampled
            ) / len(sampled))
    result = {}
    for metric, samples in deltas.items():
        samples.sort()
        point = sum(
            float(candidate_by_id[identifier]["metrics"][metric]) - float(baseline_by_id[identifier]["metrics"][metric])
            for identifier in baseline_by_id
        ) / len(baseline_by_id)
        result[metric] = {
            "delta": point,
            "ci95_low": samples[int(0.025 * iterations)],
            "ci95_high": samples[min(iterations - 1, math.ceil(0.975 * iterations) - 1)],
        }
    return result


def promotion_gate(
    baseline_summary: dict, candidate_summary: dict, bootstrap: dict, candidate_run_traces: list[dict],
) -> dict:
    """Apply the fixed development-promotion rule without discretionary tuning."""
    failures = []
    if bootstrap["slot_recall"]["ci95_low"] <= 0:
        failures.append("slot_recall_ci95_low_not_positive")
    for metric in ("mrr", "ndcg"):
        if bootstrap[metric]["delta"] < 0:
            failures.append(f"{metric}_delta_negative")
    if candidate_summary["latency_ms"]["p95"] > baseline_summary["latency_ms"]["p95"] * 1.25:
        failures.append("p95_latency_over_125_percent_baseline")
    if len(candidate_run_traces) != candidate_summary["records"]:
        failures.append("trace_record_count_mismatch")
    for trace in candidate_run_traces:
        if trace.get("fallback"):
            failures.append("retrieval_fallback")
        if trace.get("source_binding_valid") is not True:
            failures.append("invalid_source_binding")
        if trace.get("top_five_valid") is not True:
            failures.append("invalid_top_five")
    return {"passed": not failures, "failures": sorted(set(failures))}


def evaluate_v2_predictions(
    annotations: list[dict], predictions: list[dict], manifest_data: dict, *, split: str, k: int = 5,
) -> list[dict]:
    """Score exact-ID prediction files against adjudicated v2 annotations."""
    annotation_by_id = index_records(annotations, "annotations")
    prediction_by_id = index_records(predictions, "predictions")
    sample = manifest_data.get("splits", {}).get(split)
    if not isinstance(sample, list):
        raise ValueError(f"manifest lacks split={split}")
    sample_by_id = index_records(sample, "manifest split")
    if set(annotation_by_id) != set(sample_by_id) or set(prediction_by_id) != set(sample_by_id):
        raise ValueError("annotations, predictions, and manifest IDs must match exactly")
    traces = []
    for identifier in sorted(sample_by_id):
        prediction = prediction_by_id[identifier]
        tables = prediction.get("ranked_tables")
        if not isinstance(tables, list) or not all(isinstance(table, str) for table in tables):
            raise ValueError(f"prediction id={identifier} lacks ranked_tables")
        if len(tables) != len(set(tables)):
            raise ValueError(f"prediction id={identifier} contains duplicate table IDs")
        sample_record = sample_by_id[identifier]
        traces.append({
            "id": identifier,
            "cluster_id": "|".join(map(str, sample_record.get("report_key", [identifier]))),
            "weight": 1 / float(sample_record["inclusion_probability"]) if sample_record.get("inclusion_probability") else 1.0,
            "metrics": score_slots(annotation_by_id[identifier], tables, k=k),
            "latency_ms": prediction.get("latency_ms", 0.0),
            "candidate_count": prediction.get("candidate_count", len(tables)),
        })
    return traces


def atomic_write_run(output_dir: Path, artifacts: dict[str, object], input_paths: dict[str, Path]) -> None:
    """Write immutable result artifacts, then atomically publish one new run directory."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        artifact_hashes = {}
        for name, artifact in artifacts.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            path.write_text(content, encoding="utf-8")
            artifact_hashes[name] = hash_file(path)
        run_manifest = {
            "inputs": {name: hash_file(path) for name, path in sorted(input_paths.items())},
            "artifacts": artifact_hashes,
        }
        (temporary / "manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
