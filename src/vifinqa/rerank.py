"""Optional zero-shot cross-encoder reranking for baseline table candidates."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import Table


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# Sequence length dominates cost on CPU, and this machine has four cores. At 512
# the model does 3.3 pairs/sec, which is four hours for one submission; at 192 it
# does 6.5 and still covers a table's title, headers, periods, and matched row.
MAX_LENGTH = 192


def table_representation(
    table: "Table", row_index: int, *, inventory: int = 0, position: tuple[int, int] | None = None,
) -> str:
    """Build one deterministic, non-repeated table representation.

    With `inventory`, the table is described by its line items rather than by the
    single row the sparse ranker matched. That row is the one that made the table
    look relevant, so showing only it asks the reranker to judge a table through
    the weaker ranker's choice, and it cannot recover when that choice is wrong.
    Listing the row labels tells the model what the table actually contains.

    `position` is (ordinal, count) within the report. Primary statements come
    first and notes after, and gold sits in the first fifth 60% of the time
    against 34% for the non-gold tables we submit — but 40% of gold is a note,
    so it is a fact for the model to weigh per question, not a rule.
    """
    headers = " | ".join(" | ".join(row) for row in table.headers)
    periods = " | ".join(table.periods)
    row = " | ".join(table.rows[row_index])
    listed = ""
    if inventory:
        labels = dict.fromkeys(
            " ".join(cell for cell in item[:1] if cell).strip()
            for item in table.rows if item and item[0].strip()
        )
        listed = "; ".join(label for label in labels if label)[:inventory]
    # Ordered by how much each part decides the ranking, because the tail is what
    # truncation eats: the median representation is 621 characters and 16% run past
    # a 320-token window. Title and matched row identify the table, the inventory
    # says what else it holds, and the header and period boilerplate goes last.
    where = f"Vị trí: bảng {position[0] + 1}/{position[1]} trong báo cáo" if position else ""
    parts = [table.title, where, row, f"Các chỉ tiêu: {listed}" if listed else "", headers, periods, table.unit]
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
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval().to(device)
    if device == "cuda":
        model = model.half()
    return torch, tokenizer, model, device


def rerank(question: str, ranked: list[tuple[float, "Table", int]], *, batch_size: int = 8) -> list[tuple[float, "Table", int]]:
    """Score immutable baseline candidates. Ties retain baseline order."""
    if not ranked:
        return []
    if batch_size not in {1, 2, 4, 8, 16, 32, 64}:
        raise ValueError("batch_size must be a power of two up to 64")
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
