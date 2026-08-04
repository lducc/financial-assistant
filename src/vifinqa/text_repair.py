"""E011: loss-averse repair for common UTF-8-as-Western-text mojibake."""

from __future__ import annotations


# These are evidence of likely UTF-8 bytes rendered through Latin-1/CP1252.
MARKERS = ("Ã", "Ä", "Å", "Æ", "á»", "áº", "â€", "Â")


def mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MARKERS)


def repair_text(text: str) -> tuple[str, bool, str | None]:
    """Return (selected_text, adopted, codec); never replaces a non-improving candidate."""

    raw_score = mojibake_score(text)
    if not raw_score:
        return text, False, None
    candidates: list[tuple[int, str, str]] = []
    for codec in ("latin-1", "cp1252"):
        try:
            candidate = text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        score = mojibake_score(candidate)
        if score < raw_score:
            candidates.append((score, candidate, codec))
    if not candidates:
        return text, False, None
    _, candidate, codec = min(candidates, key=lambda item: (item[0], item[2]))
    return candidate, True, codec
