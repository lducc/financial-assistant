#!/usr/bin/env python3
"""Verify whether selected values in the public leader artifact are precomputed."""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

CID = re.compile(r"candidate_id['\"\]]*\s*==\s*['\"]([^'\"]+)['\"]")
SOURCES = re.compile(r"source_id['\"\]]*\s*==\s*['\"]([^'\"]+)['\"]")
ISIN = re.compile(r"candidate_id['\"\]]*\.isin\(\[([^]]+)\]\)")
QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")

def rows(root: Path, qid: int) -> list[dict[str, str]]:
    out = []
    for p in sorted((root / "data").glob(f"q{qid:04d}_df*.csv")):
        with p.open(encoding="utf-8", newline="") as h: out += list(csv.DictReader(h))
    return out

def close(a: object, b: object) -> bool:
    try: return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError): return False

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path); a = p.parse_args()
    records = json.loads((a.root / "submission.json").read_text(encoding="utf-8"))
    counts = Counter(); deltas = Counter(); examples = []
    for r in records:
        q, rs = r.get("pandas_query", ""), rows(a.root, int(r["id"]))
        target = None; expected = None; mode = "unmatched"
        m = CID.search(q)
        if m:
            target = next((x for x in rs if x.get("candidate_id") == m.group(1)), None); expected = target.get("answer_value") if target else None; mode = "candidate_id"
        else:
            m = SOURCES.search(q)
            if m:
                target = next((x for x in rs if x.get("source_id") == m.group(1)), None); expected = target.get("computed_answer") if target else None; mode = "source_id"
            else:
                m = ISIN.search(q)
                if m:
                    ids = set(QUOTED.findall(m.group(1))); chosen = [x for x in rs if x.get("candidate_id") in ids]
                    expected = sum(float(x["answer_value"]) for x in chosen) if chosen and all("answer_value" in x for x in chosen) else None
                    target = chosen[0] if chosen else None; mode = "isin_sum"
        counts[f"{mode}_matched" if target is not None else f"{mode}_unmatched"] += 1
        if target is not None and expected is not None:
            counts["answer_equals_precomputed"] += int(close(expected, r.get("answer")))
            counts["answer_checked"] += 1
            try:
                if r.get("relevant_tables"):
                    d = int(str(r["relevant_tables"][0]).rsplit("|",1)[1]) - int(float(target["table_id"]))
                    deltas[d] += 1
            except (IndexError, KeyError, TypeError, ValueError): pass
            if len(examples) < 5:
                examples.append({"id": r["id"], "mode": mode, "answer": r.get("answer"), "precomputed": expected, "table_delta": next(iter(deltas)) if deltas else None})
    print(json.dumps({"records": len(records), "counts": counts, "table_delta_distribution": deltas, "examples": examples}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
