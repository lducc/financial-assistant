from collections import Counter
import re

from .retrieval import tokenize


CODE = re.compile(r"^\d{2,3}[A-Za-z]?$")
# "6", "12", "5.1", "V.01"
NOTE = re.compile(r"^(?:[IVX]+\.)?\d{1,2}(?:\.\d{1,2})?$")
TABLE_TAG = re.compile(r"<table")
NUMBER = re.compile(r"^\(?-?\d[\d.,\s]*%?\)?$")
NOTE_HEADING = re.compile(
    r"^\s*(?:[IVX]+\.)?\d{1,2}(?:\.\d{1,2})?[.)]?\s+\S[^\n]{5,70}$",
    re.M,
)


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


# found by its repeated codes, since OCR moves the column
def code_column(grid: list[list[str]]) -> int | None:
    for row in grid[:3]:
        for index, cell in enumerate(row):
            if normalize(cell) == "ma so":
                return index

    counts: Counter[int] = Counter()
    for row in grid[:30]:
        for index, cell in enumerate(row):
            if CODE.match(cell.strip()):
                counts[index] += 1
    return next((index for index in sorted(counts) if counts[index] >= 3), None)


def note_column(grid: list[list[str]], code_index: int | None) -> int | None:
    for row in grid[:3]:
        for index, cell in enumerate(row):
            if normalize(cell) == "thuyet minh":
                return index
    return code_index + 1 if code_index is not None else None


def row_label(row: list[str], code_index: int | None, note_index: int | None) -> str:
    blocked = {index for index in (code_index, note_index) if index is not None}
    return " ".join(
        cell.strip() for index, cell in enumerate(row)
        if index not in blocked and cell.strip() and not NUMBER.fullmatch(cell.strip())
    )


def row_note(row: list[str], note_index: int) -> str | None:
    if note_index >= len(row):
        return None
    cell = row[note_index].strip()
    return cell if NOTE.match(cell) else None


# keeps sub-note levels: `V.04`, `04`, `05.01`
def normalize_note(note: str) -> str:
    parts = note.split(".")
    if re.fullmatch(r"[IVX]+", parts[0]):
        parts = parts[1:]
    return ".".join(str(int(part)) for part in parts if part.isdigit())


# heading length is bounded to skip prose
def note_table_line(report_text: str, note: str) -> int | None:
    number = normalize_note(note)
    heading = re.search(
        rf"^\s*{re.escape(number)}[.)]?\s+\S[^\n]{{5,70}}$", report_text, re.M,
    )
    if not heading:
        return None
    following = report_text[heading.end():]
    table = TABLE_TAG.search(following)
    if not table:
        return None
    next_heading = NOTE_HEADING.search(following)
    if next_heading and next_heading.start() < table.start():
        return None
    return report_text.count("\n", 0, heading.end() + table.start()) + 1
