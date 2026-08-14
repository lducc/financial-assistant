"""Parse a listwise permutation back into a candidate order.

The pointwise reranker scores each table alone, so two tables in the same report
that both carry the asked line item both score near 1.0 and nothing ever
compares them. That is the measured failure: of the questions whose top-1 is not
gold, the median score gap between the wrong pick and the best gold is 0.067 and
47% are within 0.05. A listwise pass shows the candidates together and asks for
an order instead of a verdict.

The cost of that is a generated permutation, which can repeat a label, invent one
out of range, or stop early. None of those may lose a table: a candidate dropped
here disappears from the submission and takes its retrieval score with it. So
parsing keeps what the model said, in the order it said it, and appends
everything it did not mention in the order it already had. A garbled generation
degrades to the incoming ranking rather than to a shorter one.
"""

import re


LABEL = re.compile(r"\[(\d+)\]")


def parse_permutation(text: str, size: int) -> list[int]:
    """Zero-based positions named by `[n]` labels, first mention winning.

    Repeats are dropped rather than reordered because the model's first mention
    is its confident one, and out-of-range labels are dropped because there is no
    candidate to attach them to.
    """
    seen: dict[int, None] = {}
    for match in LABEL.finditer(text or ""):
        position = int(match.group(1)) - 1
        if 0 <= position < size:
            seen.setdefault(position, None)
    return list(seen)


def reorder(candidates: list[str], text: str) -> list[str]:
    """The listwise order, with every unmentioned candidate kept behind it."""
    order = parse_permutation(text, len(candidates))
    ranked = [candidates[position] for position in order]
    mentioned = set(order)
    ranked.extend(
        candidate for position, candidate in enumerate(candidates) if position not in mentioned
    )
    return ranked


def borda(orders: list[list[str]]) -> list[str]:
    """Merge several orders of the same candidates by Borda count.

    An LLM asked to rank a list is biased toward the order it was shown, so a
    single pass cannot distinguish a model that judged from one that echoed its
    input. Permutation self-consistency answers that by ranking the same
    candidates under different input orders and aggregating. Borda is the
    aggregation that suits it: each order awards points by position, so a
    candidate has to place well under every presentation to finish high, and one
    pass cannot carry it alone.

    A candidate missing from an order is scored as if it were last there, which
    is what an omission means — the model declined to rank it.
    """
    orders = [order for order in orders if order]
    if not orders:
        return []
    candidates = list(dict.fromkeys(table for order in orders for table in order))
    points: dict[str, float] = {table: 0.0 for table in candidates}
    for order in orders:
        size = len(order)
        position = {table: index for index, table in enumerate(order)}
        for table in candidates:
            # Normalized by length so a shorter pass — one where the model
            # stopped early — still counts for as much as a complete one.
            points[table] += (size - position.get(table, size)) / size
    # Ties break on the first order, which is the incoming ranking: absent a
    # reason to move a candidate, it keeps the place it already had.
    first = {table: index for index, table in enumerate(orders[0])}
    return sorted(candidates, key=lambda table: (-points[table], first.get(table, len(candidates))))


def agreement(left: list[str], right: list[str]) -> float:
    """Spearman rank correlation between two orders of the same candidates.

    Reported alongside the score because the two readings need different
    responses: a flat result with high agreement means the model ranked
    consistently and simply cannot separate these tables, which is an argument
    for training it; a flat result with low agreement means it is drifting with
    presentation order, which is an argument for more passes or a smaller window.
    """
    shared = [table for table in left if table in set(right)]
    size = len(shared)
    if size < 2:
        return 1.0
    lookup_left = {table: index for index, table in enumerate(left)}
    lookup_right = {table: index for index, table in enumerate(right)}
    squared = sum((lookup_left[table] - lookup_right[table]) ** 2 for table in shared)
    return 1 - 6 * squared / (size * (size ** 2 - 1))


def splice(ranking: list[str], head: list[str]) -> list[str]:
    """Replace the head of a ranking, keeping the tail and dropping nothing.

    The listwise pass only sees the top of the order, so the tail keeps whatever
    the pointwise stage decided. Anything in `head` that is not in `ranking` is
    ignored rather than inserted: the submission may only contain tables that
    were actually retrieved.
    """
    known = set(ranking)
    reordered = [table for table in dict.fromkeys(head) if table in known]
    covered = set(reordered)
    return reordered + [table for table in ranking if table not in covered]
