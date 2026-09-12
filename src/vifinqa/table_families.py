from .retrieval import Table, tokenize


FAMILY_ANCHORS: dict[str, tuple[str, ...]] = {
    "balance": (
        "tai san", "no phai tra", "von chu so huu", "tien mat",
        "cho vay", "tien gui", "phai thu", "phai tra",
    ),
    "income": (
        "doanh thu", "thu nhap", "chi phi", "loi nhuan", "lai", "lo",
        "thue thu nhap",
    ),
    "cashflow": (
        "luu chuyen tien", "dong tien", "hoat dong kinh doanh",
        "hoat dong dau tu", "hoat dong tai chinh", "tien thu", "tien chi",
    ),
    "equity": ("von chu so huu", "co phieu", "co tuc", "co dong"),
    "note": ("thuyet minh", "du phong", "ky han", "phan loai", "chi tiet"),
}


def _families(text: str) -> frozenset[str]:
    return frozenset(
        family for family, anchors in FAMILY_ANCHORS.items()
        if any(" ".join(tokenize(anchor)) in text for anchor in anchors)
    )


def table_families(table: Table) -> frozenset[str]:
    text = " ".join((table.title, *table.context, *(" ".join(row) for row in table.rows[:120])))
    return _families(" ".join(tokenize(text))) or frozenset({"other"})


def query_families(tokens: list[str]) -> frozenset[str]:
    return _families(" ".join(tokens))


def family_match_score(query: list[str], table: Table) -> float:
    wanted = query_families(query)
    return 1.0 if wanted and wanted & table_families(table) else 0.0
