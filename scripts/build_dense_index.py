#!/usr/bin/env python3
"""Build an E5 row index for hybrid table retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifinqa.dense import MAX_LENGTH, MODEL_NAME, row_text
from vifinqa.retrieval import load_reports, materialize_candidate_rows


def encode(texts: list[str], *, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)
    if device == "cuda":
        model = model.half()
    vectors = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(functional.normalize(pooled, p=2, dim=1).float().cpu().numpy())
        print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(vectors).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    reports = load_reports(args.data_root)
    tables, rows = materialize_candidate_rows(reports)
    table_ids = [table.table_id for table in tables]
    report_ids = list(dict.fromkeys(table.report_id for table in tables))
    table_index = {table_id: index for index, table_id in enumerate(table_ids)}
    report_index = {report_id: index for index, report_id in enumerate(report_ids)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "vectors.npy", encode([row_text(table, row_index) for table, row_index, _ in rows], batch_size=args.batch_size))
    np.save(args.output_dir / "row_table_ids.npy", np.asarray([table_index[table.table_id] for table, _, _ in rows], dtype=np.int32))
    np.save(args.output_dir / "row_report_ids.npy", np.asarray([report_index[table.report_id] for table, _, _ in rows], dtype=np.int16))
    np.save(args.output_dir / "row_indices.npy", np.asarray([row_index for _, row_index, _ in rows], dtype=np.int16))
    (args.output_dir / "table_ids.json").write_text(json.dumps(table_ids), encoding="utf-8")
    (args.output_dir / "report_ids.json").write_text(json.dumps(report_ids), encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model_name": MODEL_NAME, "max_length": MAX_LENGTH, "table_count": len(tables), "row_count": len(rows),
    }, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
