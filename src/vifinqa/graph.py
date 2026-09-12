from dataclasses import dataclass

from .notes import code_column, note_column, note_table_line, row_label, row_note
from .slots import normalize


SOURCE_TABLE = "SOURCE_TABLE"
NOTE_REF = "NOTE_REF"


@dataclass(frozen=True)
class EvidenceSlot:
    report_id: str
    line_item: str


@dataclass(frozen=True)
class RowAnchor:
    report_id: str
    table_id: str
    row: int
    label: str
    line_item: str
    code: str = ""
    note: str = ""


@dataclass(frozen=True)
class Edge:
    table_id: str
    kind: str
    anchor: RowAnchor


@dataclass(frozen=True)
class EvidencePath:
    slot: EvidenceSlot
    anchor: RowAnchor
    edge: Edge


def evidence_slots(report_ids: list[str], line_items: list[str]) -> list[EvidenceSlot]:
    return [EvidenceSlot(report_id, item) for report_id in report_ids for item in line_items]


def row_anchors(
    grid: list[list[str]], report_id: str, table_id: str, line_items: list[str],
) -> list[RowAnchor]:
    code_index = code_column(grid)
    note_index = note_column(grid, code_index)
    anchors = []
    for index, row in enumerate(grid):
        raw_label = row_label(row, code_index, note_index)
        padded = f" {normalize(raw_label)} "
        item = next((item for item in line_items if f" {item} " in padded), None)
        if item is None:
            continue
        code, note = "", ""
        if code_index is not None and code_index < len(row):
            cell = row[code_index].strip()
            code = cell if cell.isdigit() else ""
            note = row_note(row, note_index) or ""
        anchors.append(RowAnchor(
            report_id=report_id, table_id=table_id, row=index,
            label=raw_label, line_item=item, code=code, note=note,
        ))
    return anchors


def anchor_edges(anchor: RowAnchor, report_text: str, source_line: int) -> list[Edge]:
    """The row's own table, plus the note table it cites."""
    edges = [Edge(anchor.table_id, SOURCE_TABLE, anchor)]
    if anchor.note:
        line = note_table_line(report_text, anchor.note)
        if line is not None and line != source_line:
            edges.append(Edge(f"{anchor.report_id}|{line}", NOTE_REF, anchor))
    return edges


def table_paths(
    slot: EvidenceSlot, table_id: str, grid: list[list[str]], report_text: str,
    source_line: int,
) -> list[EvidencePath]:
    if table_id.rsplit("|", 1)[0] != slot.report_id:
        return []
    paths = []
    for anchor in row_anchors(grid, slot.report_id, table_id, [slot.line_item]):
        paths.extend(
            EvidencePath(slot, anchor, edge)
            for edge in anchor_edges(anchor, report_text, source_line)
        )
    return paths


def select_paths(
    paths: list[EvidencePath], already: set[str], limit: int,
) -> list[EvidencePath]:
    """Cover each slot once before taking a second path from any slot."""
    seen_tables = set(already)
    selected = []
    remaining = list(paths)
    for slot in dict.fromkeys(path.slot for path in paths):
        chosen = next(
            (path for path in remaining
             if path.slot == slot and path.edge.table_id not in seen_tables),
            None,
        )
        if chosen is None:
            continue
        selected.append(chosen)
        seen_tables.add(chosen.edge.table_id)
        remaining.remove(chosen)
        if len(selected) == limit:
            return selected
    for path in remaining:
        if path.edge.table_id in seen_tables:
            continue
        selected.append(path)
        seen_tables.add(path.edge.table_id)
        if len(selected) == limit:
            break
    return selected
