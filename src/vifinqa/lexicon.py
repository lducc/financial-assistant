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
