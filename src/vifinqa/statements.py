"""The structure Circular 200 puts in every filing, read straight off the page.

Vietnamese filers all present the same four primary statements with the same
mandated account codes, so `Mã số` 110 is "Tiền và các khoản tương đương tiền"
in every company and every year. The corpus therefore labels itself: 146,246
tables carry millions of (label, code) observations, which is a lexicon of the
line-item vocabulary — including its OCR variants — built without a single gold
label.

Each statement row also carries a `Thuyết minh` cell holding the number of the
note that details it, and each note table is preceded by a heading line that
opens with that same number. The pair is a literal string join from a statement
row to the note table that expands it, which is the second and third gold table
of a question that the ranker has to guess at today.
"""

import re

ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL = re.compile(r"<td>(.*?)</td>", re.S)
CODE = re.compile(r"^\d{2,3}$")
NOTE_REF = re.compile(r"^(?:[IVX]+\.)?\d+(?:\.\d+)*$")
HEADING = re.compile(r"^\s*((?:[IVX]{1,4}\.)?\d+(?:\.\d+)*)\.?\s+(\S.*)$")
LEADER = re.compile(r"^(?:[-–—•+*]|\(?[a-zA-Z][).]|[IVXivx]+\.|\d+[.)])\s*")


def rows(table: str) -> list[list[str]]:
    return [[cell.strip() for cell in CELL.findall(row)] for row in ROW.findall(table)]


def normalize_label(text: str) -> str:
    """Strip the outline numbering filers use so labels compare across reports."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    while True:
        shorter = LEADER.sub("", text).strip()
        if shorter == text:
            return shorter
        text = shorter


def header_columns(table_rows: list[list[str]]) -> tuple[int | None, int | None]:
    """Where the account code and the note reference sit, if the table has them.

    Inferred from the column contents rather than the header, because statements
    run over pages and a continuation carries the codes without repeating the
    header, and because the header itself is the part OCR damages most.
    """
    for row in table_rows[:3]:
        code = next((i for i, cell in enumerate(row) if "Mã s" in cell), None)
        note = next((i for i, cell in enumerate(row) if "Thuyết minh" in cell), None)
        if code is not None:
            return code, note
    width = max((len(row) for row in table_rows), default=0)
    code = note = None
    for column in range(1, width):
        values = [row[column] for row in table_rows if len(row) > column and row[column]]
        if len(values) < 2:
            continue
        if code is None and sum(map(bool, map(CODE.match, values))) >= 0.6 * len(values):
            code = column
        elif code is not None and note is None and column == code + 1:
            if sum(map(bool, map(NOTE_REF.match, values))) >= 0.6 * len(values):
                note = column
    return code, note


def statement_rows(table: str):
    """(label, code, note reference) for every row that carries an account code."""
    table_rows = rows(table)
    code_at, note_at = header_columns(table_rows)
    if code_at is None:
        return
    for row in table_rows:
        if len(row) <= code_at or not CODE.match(row[code_at]):
            continue
        label = normalize_label(row[0])
        note = row[note_at].strip() if note_at is not None and len(row) > note_at else ""
        yield label, row[code_at], note if NOTE_REF.match(note) else ""


def table_heading(lines: list[str], index: int, reach: int = 8) -> tuple[str, str]:
    """The number and label of the line that introduces the table at `index`.

    Notes are titled the way the statement row that points at them is worded, so
    the heading is the join key even when its numbering is unreadable. Page
    furniture — the running company name, the page marker — is skipped rather
    than mistaken for a title.
    """
    for line in reversed(lines[max(0, index - reach):index]):
        text = line.strip()
        if not text or text.startswith("=====") or text.isupper():
            continue
        if text.startswith("<table>") or text.startswith("<"):
            break
        match = HEADING.match(text)
        if match:
            return match.group(1).rstrip("."), normalize_label(match.group(2))
        return "", normalize_label(text)
    return "", ""
