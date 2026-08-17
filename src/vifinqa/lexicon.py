"""Resolve a question's line item to the account codes the corpus files it under."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from vifinqa.statements import normalize_label


def load_lexicon(path: Path) -> dict[str, Counter]:
    """Row label to the account codes observed against it, by frequency."""
    labels = defaultdict(Counter)
    for code, variants in json.loads(Path(path).read_text(encoding="utf-8")).items():
        for label, count in variants.items():
            labels[label][code] += count
    return labels


def resolve(item: str, labels: dict[str, Counter], limit: int = 2) -> list[str]:
    """The codes a named line item most likely refers to.

    Questions abbreviate — "lợi nhuận sau thuế" for "lợi nhuận sau thuế thu nhập
    doanh nghiệp" — so an exact label is preferred and containment is the
    fallback, shortest containing label first because it adds the least.
    """
    exact = labels.get(normalize_label(item))
    if exact:
        return [code for code, _ in exact.most_common(limit)]
    key = normalize_label(item)
    holders = sorted(((len(label), label) for label in labels if key in label))
    if not holders:
        holders = sorted(((-len(label), label) for label in labels if label in key))
    counts = Counter()
    for _, label in holders[:20]:
        counts.update(labels[label])
    return [code for code, _ in counts.most_common(limit)]


def item_row(rows: list[list[str]], items: list[str]) -> int | None:
    """The row a named line item occupies, by the label the corpus writes for it.

    The sparse ranker picks the row that made the table look relevant, which is
    the row its own matching found rather than the row the question asks about.
    The label identifies the row directly, but not always in the first column —
    some filings put `Mã số` there — and OCR runs labels into their neighbours, so
    the search covers the leading cells and accepts containment. An exact label
    beats a containment anywhere later in the table; row 0 is the header and can
    never be a value.
    """
    contained = loose = None
    words = [(item, set(item.split())) for item in items]
    for index, row in enumerate(rows):
        if index == 0 or not row:
            continue
        for cell in row[:3]:
            label = normalize_label(cell)
            if not label:
                continue
            tokens = set(label.split())
            for item, item_words in words:
                if label == item:
                    return index
                if item in label and contained is None:
                    contained = index
                elif item_words <= tokens and loose is None:
                    loose = index
    return contained if contained is not None else loose
