#!/usr/bin/env python3
"""Summarize query and evidence schemas in a public submission artifact."""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

CID = re.compile(r"candidate_id['\"\]]*\s*==\s*['\"]([^'\"]+)['\"]")
VAR = re.compile(r"\b([A-Za-z_]\w*)\.")

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path); a = p.parse_args()
    records = json.loads((a.root / "submission.json").read_text(encoding="utf-8"))
    query_forms, missing = Counter(), []
    colsets, answer_colsets = Counter(), Counter()
    for r in records:
        q = r.get("pandas_query", "")
        if CID.search(q):
            query_forms["candidate_id"] += 1
        elif "iloc" in q or "iat" in q:
            query_forms["positional_index"] += 1
        elif any(op in q for op in ["sum(", "mean(", "/", "*", "+", "-"]):
            query_forms["arithmetic_or_aggregate"] += 1
        else:
            query_forms["other"] += 1
        paths = sorted((a.root / "data").glob(f"q{int(r['id']):04d}_df*.csv"))
        rows = []
        for path in paths:
            with path.open(encoding="utf-8", newline="") as h: rows += list(csv.DictReader(h))
        cols = tuple(sorted(set().union(*(row.keys() for row in rows)))) if rows else ()
        colsets[cols] += 1
        answer_colsets["answer_value" if "answer_value" in cols else "no_answer_value"] += 1
        if len(missing) < 8 and not CID.search(q):
            missing.append({"id": r["id"], "query": q, "answer": r.get("answer"), "columns": cols, "rows": len(rows)})
    print(json.dumps({"records": len(records), "query_forms": query_forms, "csv_schema_counts": answer_colsets, "examples_without_candidate_id": missing}, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
