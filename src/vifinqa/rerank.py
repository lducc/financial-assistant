from .retrieval import Table, column_paths, row_paths


INVENTORY = 600

# the corpus stores an English scope tag; questions say it in Vietnamese
SCOPE_WORDING = {"consolidated": "hợp nhất", "separate": "công ty mẹ"}


def table_representation(
    table: Table, row_index: int, *, inventory: int = 0, identity=None,
    hierarchy: bool = False,
) -> str:
    headers = " | ".join(" | ".join(row) for row in table.headers)
    periods = " | ".join(table.periods)
    row = " | ".join(table.rows[row_index])
    paths = row_paths(table) if hierarchy else ()
    path = " > ".join(paths[row_index]) if hierarchy else ""
    columns = "; ".join(
        " > ".join(parts) for parts in column_paths(table) if parts
    ) if hierarchy else ""
    listed = ""
    if inventory:
        header_rows = set(table.headers)
        labels = dict.fromkeys(
            " > ".join(paths[index]) if hierarchy
            else " ".join(cell for cell in item[:1] if cell).strip()
            for index, item in enumerate(table.rows)
            if item not in header_rows and any(cell.strip() for cell in item)
        )
        listed = "; ".join(label for label in labels if label)[:inventory]
    filing = ""
    if identity is not None:
        scope = SCOPE_WORDING.get(identity.scope, identity.scope)
        filing = f"{identity.ticker} {identity.year} {scope}".strip()
    # decisive parts first since truncation eats the tail
    parts = [
        filing,
        table.title,
        f"Đường dẫn dòng: {path}" if path else "",
        row,
        f"Các chỉ tiêu: {listed}" if listed else "",
        f"Đường dẫn cột: {columns}" if columns else headers,
        periods,
        table.unit,
    ]
    return "\n".join(part for part in dict.fromkeys(part.strip() for part in parts) if part)
