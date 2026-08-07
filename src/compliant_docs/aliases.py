from __future__ import annotations

import re
from pathlib import Path

from .normalize import fold


# Historical names and brands used by questions but absent from code_stock.csv.
# Every entry must be verifiable inside a report owned by the target ticker.
ALIAS_SPECS = (
    ("hang tieu dung masan", "MCH"),
    ("masan meatlife", "MML"),
    ("masan high tech materials", "MSR"),
    ("hoa phat", "HPG"),
    ("hoa sen", "HSG"),
    ("nam kim", "NKG"),
    ("masan", "MSN"),
    ("dai duong", "OGC"),
    ("vinamilk", "VNM"),
    ("dabaco", "DBC"),
    ("sao mai", "ASM"),
    ("thuy san minh phu", "MPC"),
    ("dam phu my", "DPM"),
    ("dam ca mau", "DCM"),
    ("do thi kinh bac", "KBC"),
    ("cong nghiep cao su viet nam", "GVR"),
    ("van phu", "VPI"),
    ("hai phat", "HPX"),
    ("tap doan dat xanh", "DXG"),
    ("dia oc sai gon thuong tin", "SCR"),
    ("ngan hang tmcp sai gon thuong tin", "STB"),
    ("sai gon thuong tin", "STB"),
    ("mbbank", "MBB"),
    ("eximbank", "EIB"),
    ("dau tu phat trien xay dung", "DIG"),
    ("tan binh", "PRT"),
    ("hoang huy", "HHS"),
    ("dien luc gelex", "GEE"),
    ("da nhim ham thuan da mi", "DNH"),
    ("nong nghiep quoc te hoang anh gia lai", "HNG"),
    ("go truong thanh", "TTF"),
    ("the gioi di dong", "MWG"),
    ("san bay viet nam", "ACV"),
    ("vicem ha tien", "HT1"),
    ("phu nhuan jewelry", "PNJ"),
    ("bac a", "BAB"),
    ("quoc dan", "NVB"),
    ("sai gon cong thuong", "SGB"),
    ("quan doi", "MBB"),
    ("a chau", "ACB"),
    ("minh phu", "MPC"),
    ("dau khi ca mau", "DCM"),
    ("tap doan gelex", "GEX"),
    ("loc hoa dau binh son", "BSR"),
    ("vingroup", "VIC"),
    ("vincom retail", "VRE"),
    ("song da", "SJG"),
    ("viglacera", "VGC"),
    ("kien long", "KLB"),
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def match_aliases(question: str) -> list[str]:
    normalized = fold(question)
    matches = []
    occupied: list[tuple[int, int]] = []
    for alias, ticker in sorted(ALIAS_SPECS, key=lambda item: len(item[0]), reverse=True):
        for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            span = match.span()
            if any(start <= span[0] and span[1] <= end for start, end in occupied):
                continue
            matches.append(ticker)
            occupied.append(span)
    return list(dict.fromkeys(matches))


def verify_alias_sources(statements_root: Path) -> list[dict]:
    evidence = []
    folded_cache: dict[Path, str] = {}
    for alias, ticker in ALIAS_SPECS:
        source = None
        for path in sorted((statements_root / ticker).glob("**/*_extracted.txt"), reverse=True):
            if path not in folded_cache:
                folded_cache[path] = fold(path.read_text("utf-8", errors="replace"))
            if _contains_phrase(folded_cache[path], alias):
                source = path.relative_to(statements_root).as_posix()
                break
        if source is None:
            raise ValueError(f"Alias has no official report evidence: {alias} -> {ticker}")
        evidence.append({"alias": alias, "ticker": ticker, "source_path": source})
    return evidence