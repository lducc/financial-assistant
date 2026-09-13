from vifinqa.notes import code_column, normalize, note_column, note_table_line, row_label, row_note


def linked_note_lines(
    report_text: str, grid: list[list[str]], line_items: list[str], source_line: int,
) -> list[tuple[str, int]]:
    code_index = code_column(grid)
    if code_index is None or not line_items:
        return []
    note_index = note_column(grid, code_index)
    if note_index is None:
        return []
    found: dict[str, int] = {}
    for row in grid:
        if not row:
            continue
        label = normalize(row_label(row, code_index, note_index))
        if not label or not any(f" {item} " in f" {label} " for item in line_items):
            continue
        note = row_note(row, note_index)
        if not note or note in found:
            continue
        line = note_table_line(report_text, note)
        if line is not None and line != source_line:
            found[note] = line
    return sorted(found.items(), key=lambda pair: pair[1])


def select_slot_tables(
    record: dict, base_order: list[str], line_items: list[str], budget: int,
) -> tuple[list[str], list[dict]]:
    candidates = {candidate["table_id"]: candidate for candidate in record["candidates"]}
    ordered = [table_id for table_id in base_order if table_id in candidates]
    seen = set(ordered)
    ordered.extend(table_id for table_id in candidates if table_id not in seen)
    position = {table_id: rank for rank, table_id in enumerate(ordered)}

    slots = []
    reserved = set()
    for report_id in record.get("selected_docs", []):
        report_tables = [t for t in ordered if t.rsplit("|", 1)[0] == report_id]
        for item in line_items:
            matches = [
                t for t in report_tables
                if f" {item} " in f" {normalize(candidates[t].get('text', ''))} "
            ]
            chosen = min(matches, key=position.get) if matches else None
            slots.append({"report_id": report_id, "line_item": item, "table_id": chosen})
            if chosen:
                reserved.add(chosen)

    ranked_reserved = sorted(reserved, key=position.get)
    selected = ranked_reserved + [t for t in ordered if t not in reserved]
    return selected[:budget], slots
