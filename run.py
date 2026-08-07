from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from compliant_docs.catalog import catalog_rows, file_sha256, load_companies, load_reports
from compliant_docs.aliases import verify_alias_sources
from compliant_docs.ollama import OllamaResolver
from compliant_docs.normalize import fold
from compliant_docs.parser import parse_question
from compliant_docs.retriever import retrieve_docs
from compliant_docs.submission import make_row, validate, write_package


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--full-year-coverage", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    pseudo_rules_path = project / "sources" / "pseudo_gt_21_filtered_rules.txt"
    data_root = ROOT / "ViFinQA_data"
    questions_path = data_root / "questions" / "questions.jsonl"
    companies_path = data_root / "code_stock.csv"
    reports_root = data_root / "financial_statements"
    output_dir = project / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path)
    if args.limit:
        questions = questions[: args.limit]
    companies = load_companies(companies_path)
    reports = load_reports(reports_root)
    alias_evidence = verify_alias_sources(reports_root)
    resolver = None if args.no_llm else OllamaResolver(args.model, output_dir / "llm_cache.jsonl")
    rows, diagnostics = [], []

    for index, source in enumerate(questions, 1):
        parsed = parse_question(source["question"], companies)
        llm_record = None
        original_method = parsed.entity_method
        rule_tickers = list(parsed.tickers)
        normalized_question = fold(source["question"])
        generic_universe = original_method == "ambiguous" and normalized_question.startswith("trong cac cong ty co")
        if generic_universe:
            parsed.tickers = sorted(companies)
            parsed.entity_method = "generic_universe_recall_fallback"
        needs_llm = original_method == "ambiguous" and not generic_universe
        if needs_llm and resolver is not None:
            llm_record = resolver.resolve(int(source["id"]), source["question"], parsed, companies)
            if llm_record["tickers"]:
                group_cues = ("trong so", "trong nhom", "gom", "giua", "chenh lech", "so voi", "tru di")
                if original_method == "multiple_official_names" and any(cue in normalized_question for cue in group_cues):
                    parsed.tickers = list(dict.fromkeys(rule_tickers + llm_record["tickers"]))
                    parsed.entity_method = "rule_qwen_union_for_group"
                else:
                    parsed.tickers = llm_record["tickers"]
                    parsed.entity_method = "qwen2.5_7b_allowed_candidates"
                parsed.entity_confidence = llm_record["confidence"]
        if not parsed.tickers and parsed.candidate_tickers:
            parsed.tickers = parsed.candidate_tickers[:1]
            parsed.entity_method = "top_candidate_fallback"

        docs, decisions = retrieve_docs(
            parsed, reports, use_comparative_cover=not args.full_year_coverage
        )
        rows.append(make_row(source, docs))
        diagnostics.append({
            "id": source["id"],
            "question": source["question"],
            "tickers": parsed.tickers,
            "years": parsed.years,
            "scope": parsed.scope,
            "scope_by_year": parsed.scope_by_year,
            "entity_method": parsed.entity_method,
            "entity_confidence": parsed.entity_confidence,
            "candidate_tickers": parsed.candidate_tickers,
            "llm_cache_key": llm_record["key"] if llm_record else None,
            "decisions": decisions,
            "relevant_docs": docs,
        })
        if index % 50 == 0 or index == len(questions):
            print(f"progress={index}/{len(questions)}")

    errors = validate(rows, questions, reports)
    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors[:20]))
    (output_dir / "diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in diagnostics) + "\n", "utf-8"
    )
    (output_dir / "report_catalog.json").write_text(
        json.dumps(catalog_rows(reports), ensure_ascii=False, indent=2), "utf-8"
    )
    (output_dir / "alias_evidence.json").write_text(
        json.dumps(alias_evidence, ensure_ascii=False, indent=2), "utf-8"
    )
    manifest = {
        "pipeline": "023_compliant_docs_balanced",
        "model": None if args.no_llm else args.model,
        "model_role": "fallback entity resolution from an allow-list only",
        "model_calls": sum(item["llm_cache_key"] is not None for item in diagnostics),
        "model_upstream": "Qwen/Qwen2.5-7B-Instruct",
        "model_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        "ollama_model_id": "845dbda0ea48",
        "parameters": "7.61B",
        "license": "Apache-2.0",
        "release": "2024-09",
        "closed_api_used_for_predictions_or_labels": False,
        "old_submission_predictions_used": False,
        "pseudo_gt_predictions_used": False,
        "pseudo_gt_rules_used": True,
        "questions_sha256": file_sha256(questions_path),
        "pseudo_rules_sha256": file_sha256(pseudo_rules_path),
        "companies_sha256": file_sha256(companies_path),
        "reports": len(reports),
        "questions": len(questions),
        "comparative_year_cover": not args.full_year_coverage,
    }
    (output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    zip_path = write_package(output_dir, rows)
    distribution = Counter(len(row["relevant_docs"]) for row in rows)
    print("submission", zip_path)
    print("doc_count_distribution", dict(sorted(distribution.items())))
    print("empty_docs", sum(not row["relevant_docs"] for row in rows))


if __name__ == "__main__":
    main()

