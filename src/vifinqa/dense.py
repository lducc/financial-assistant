"""Persistent multilingual dense row index for table candidate diversification."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .retrieval import Table


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
MAX_LENGTH = 512
QUERY_INSTRUCTION = "Given a Vietnamese financial question, retrieve supporting financial-statement table rows."


def row_text(table: "Table", row_index: int) -> str:
    """Keep row label together with its table headers and OCR context."""
    headers = " | ".join(" | ".join(row) for row in table.headers)
    context = " | ".join((table.title, *table.context, headers, table.unit))
    row = " | ".join(table.rows[row_index])
    return f"passage: {context}\nDòng dữ liệu: {row}"


@lru_cache(maxsize=1)
def _encoder():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("dense retrieval requires transformers and torch") from error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)
    if device == "cuda":
        model = model.half()
    return torch, tokenizer, model, device


def encode_query(question: str) -> np.ndarray:
    """Embed one E5 retrieval query with its required task instruction."""
    torch, tokenizer, model, device = _encoder()
    encoded = tokenizer(
        f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}", max_length=MAX_LENGTH,
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return (pooled / pooled.norm(dim=1, keepdim=True)).float().cpu().numpy()[0]


class DenseIndex:
    """Memory-mapped row vectors, reduced to best score for each table."""

    def __init__(self, directory: Path) -> None:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("model_name") != MODEL_NAME:
            raise ValueError(f"dense index model mismatch: {metadata.get('model_name')!r}")
        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        self.table_ids = json.loads((directory / "table_ids.json").read_text(encoding="utf-8"))
        self.report_ids = json.loads((directory / "report_ids.json").read_text(encoding="utf-8"))
        self.row_table_ids = np.load(directory / "row_table_ids.npy", mmap_mode="r")
        self.row_report_ids = np.load(directory / "row_report_ids.npy", mmap_mode="r")
        self.row_indices = np.load(directory / "row_indices.npy", mmap_mode="r")
        if len(self.vectors) != len(self.row_table_ids):
            raise ValueError("dense index vectors and row metadata have different lengths")
        self._rows_by_report: dict[str, np.ndarray] = {}
        for report_index, report_id in enumerate(self.report_ids):
            self._rows_by_report[report_id] = np.flatnonzero(self.row_report_ids == report_index)

    def rank(
        self,
        question_vector: np.ndarray,
        candidate_report_ids: list[str],
        tables_by_id: dict[str, "Table"],
        *,
        top_k: int,
    ) -> list[tuple[float, "Table", int]]:
        row_groups = [self._rows_by_report[report_id] for report_id in candidate_report_ids if report_id in self._rows_by_report]
        if not row_groups:
            return []
        rows = np.concatenate(row_groups)
        scores = np.asarray(self.vectors[rows], dtype=np.float32) @ question_vector
        best: dict[int, tuple[float, int]] = {}
        for local_index, row in enumerate(rows):
            table_index = int(self.row_table_ids[row])
            score = float(scores[local_index])
            previous = best.get(table_index)
            if previous is None or score > previous[0]:
                best[table_index] = score, int(self.row_indices[row])
        ranked = []
        for table_index, (score, row_index) in best.items():
            table_id = self.table_ids[table_index]
            table = tables_by_id.get(table_id)
            if table is not None:
                ranked.append((score, table, row_index))
        return sorted(ranked, key=lambda item: (-item[0], item[1].table_id))[:top_k]


def fused_rankings(
    sparse: list[tuple[float, "Table", int]],
    dense: list[tuple[float, "Table", int]],
    *,
    offset: int = 60,
) -> list[tuple[float, "Table", int]]:
    """RRF union: retain lexical exactness and add semantic-only candidates."""
    aggregate: dict[str, tuple[float, "Table", int]] = {}
    for ranking in (sparse, dense):
        for rank, (_, table, row_index) in enumerate(ranking, 1):
            score, _, chosen_row = aggregate.get(table.table_id, (0.0, table, row_index))
            aggregate[table.table_id] = score + 1 / (offset + rank), table, chosen_row
    return sorted(aggregate.values(), key=lambda item: (-item[0], item[1].table_id))
