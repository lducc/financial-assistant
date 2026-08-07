from __future__ import annotations

import re
import unicodedata


LEGAL_WORDS = {
    "cong", "ty", "co", "phan", "ctcp", "tong", "tap", "doan", "ngan",
    "hang", "tmcp", "thuong", "mai", "trach", "nhiem", "huu", "han",
}


def fold(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def tokens(text: str) -> list[str]:
    return fold(text).split()


def content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if token not in LEGAL_WORDS]

