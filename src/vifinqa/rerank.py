"""Optional zero-shot cross-encoder reranking for baseline table candidates."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import Table


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_LENGTH = 512


def table_representation(table: "Table", row_index: int) -> str:
    """Build one deterministic, non-repeated table representation."""
    headers = " | ".join(" | ".join(row) for row in table.headers)
    periods = " | ".join(table.periods)
    row = " | ".join(table.rows[row_index])
    parts = [table.title, headers, periods, table.unit, row]
    return "\n".join(part for part in dict.fromkeys(part.strip() for part in parts) if part)


@lru_cache(maxsize=1)
def _model():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("reranker requires uv sync") from error
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval().to(device)
    if device == "cuda":
        model = model.half()
    return torch, tokenizer, model, device


def rerank(question: str, ranked: list[tuple[float, "Table", int]], *, batch_size: int = 8) -> list[tuple[float, "Table", int]]:
    """Score immutable baseline candidates. Ties retain baseline order."""
    if not ranked:
        return []
    if batch_size not in {1, 2, 4, 8}:
        raise ValueError("batch_size must be one of 1, 2, 4, 8")
    torch, tokenizer, model, device = _model()
    scores: list[float] = []
    for start in range(0, len(ranked), batch_size):
        batch = ranked[start:start + batch_size]
        encoded = tokenizer(
            [question] * len(batch),
            [table_representation(table, row_index) for _, table, row_index in batch],
            padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            scores.extend(model(**encoded).logits.view(-1).float().cpu().tolist())
    return [item for _, _, item in sorted(
        ((-score, baseline_rank, item) for baseline_rank, (score, item) in enumerate(zip(scores, ranked))),
        key=lambda value: (value[0], value[1]),
    )]
