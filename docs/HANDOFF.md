# ViFinQA Project Handoff

**Last updated:** 2026-08-12
**Branch:** `feat/cleanup` (cleaned, tests green, pushed)

---

## 1. Competition Overview

**R2AI2026 — Build AI Financial Data Assistant**

Text-to-Pandas task on Vietnamese financial statements (OCR). System must answer questions by retrieving tables from 1,973 OCR reports (100 tickers, 2015–2025), generating executable Pandas code, and computing numeric answers.

| Detail | Value |
|---|---|
| Questions | 1,012 |
| Reports | 1,973 |
| Tables in corpus | ~146,246 |
| Submissions/day | 10 (public), 5 (private total) |
| Docker | `codalab/codalab-legacy:py37` |
| Model limit | ≤14B params, released before 2026-06-01 |
| No GPT/Gemini | Only open models |

### Scoring (all macro-averaged)

| Metric | What it measures |
|---|---|
| **Execution Accuracy** | Pandas code runs + produces correct number (primary rank) |
| **Tables F2-Macro** | Table retrieval quality (beta=2, recall-heavy) |
| **Docs F2-Macro** | Document retrieval quality (beta=2) |
| **Tables/Docs Precision, Recall, MRR@5** | Individual retrieval sub-metrics |
| **Answer Accuracy** | Numeric answer within 0.02% tolerance |

F2 weights recall ~4x precision. Missing a relevant table hurts more than
including an extra one. When in doubt, include the table.

---

## 2. Current Leaderboard (public, 506/1,012 scored)

| # | Team | Exec Acc | Tables F2 | Docs F2 | Docs MRR5 | Answer Acc |
|---|---|---|---|---|---|---|
| 1 | **synera** (sub 2333) | **0.660** | 0.000 | 0.968 | 0.965 | 0.660 |
| 2 | BM25 Baseline | 0.358 | 0.893 | 0.894 | 0.894 | 1.000 |
| 3 | nguyenvuhoanglong | 0.291 | 0.550 | 0.961 | 0.975 | 0.293 |
| 4 | yyy | 0.289 | 0.353 | 0.937 | 0.961 | 0.289 |
| 5 | vilamiu | 0.285 | 0.413 | 0.951 | 0.978 | 0.287 |
| 6 | aifinance | 0.255 | 0.000 | 0.927 | 0.963 | 0.255 |

### Key correlations

- Docs F2 <-> Exec Acc: strong (r ~ 0.63). Document retrieval is the main execution driver.
- Tables F2 <-> Exec Acc: weak (r ~ 0.19). Table retrieval matters less than expected.
- Winner profile: synera has 0.968 Docs F2 but 0.0 Tables F2. Document retrieval quality is what wins.

---

## 3. Winner Forensics: synera (sub 2333)

**Source:** `notes/conversation.md`, E016 experiment.

### What they did

1. Precomputed `answer_value` inside every CSV evidence file.
2. Pandas query selects answer by `candidate_id` from the CSV. No runtime computation.
3. Query mix: 359 `candidate_id` selectors, 651 `source_id -> computed_answer` selectors, 2 sums.
4. Every table address is exactly 1 line below the CSV `table_id` (systematic -1 offset).

### Why tables F2 = 0

Scorer reports `gold=506, pred=1012`. The -1 offset means every submitted table
address is wrong from the scorer's perspective. Document retrieval is excellent;
table-ID serialization is broken.

### What this means for us

- Document retrieval quality is the #1 score driver. Do not skip it.
- The precomputed-answer shortcut is fragile (scorer-specific, breaks on private phase).
- Do not copy the shortcut directly. Use grounded evidence tables but compute answers from raw values.
- Recommended path: metadata-gated hybrid retrieval -> evidence closure -> raw candidate table -> typed program proposer -> restricted Pandas execution -> unit/provenance validation -> canonical table-ID serializer.

---

## 4. Competitor Repos (Deep Dive)

### 4.1 TanKai-247/financial-text-to-pandas

**Most complete competitor.** Modular, documented, production-quality codebase.

| Aspect | Detail |
|---|---|
| Status | Baseline retrieval + submission builder working. Answer generation stubbed. |
| Structure | `src/finpandas/` 10 modules: ingestion, parsing, storage, retrieval, query understanding, schema linking, planning, compilation, execution, validation |
| Key pattern | Typed ProgramPlan DSL -> deterministic Pandas compiler. No free-form LLM code gen. |
| Query understanding | `compound.py`: two-stage compound queries: selection phase (argmax/argmin/year-axis) -> target phase (lookup/growth/difference). Regex + heuristic, no LLM. |
| Schema linking | `linker.py` + `multi.py`: binds row/column references before compilation. Coordinates only (`iloc[row, col]`). |
| Compiler safety | No imports, no file/network access, no loops, no user-defined functions. Sandboxed. |
| Coverage planning | Per `(ticker, year, scope, metric)` slot quotas. Prevents one strong company/year from starving other operands. |
| Hierarchical retrieval | SQLite FTS5: report BM25 -> table BM25 -> row BM25 -> deterministic score fusion. |
| Evidence planner | Reserves one table per slot, fills rest by relevance. Pilot: 1.000 report recall, 1.000 table recall, 0.900 row recall on 20-question pilot. |

**Reusable patterns:**

- Typed DSL/DAG for program planning (not free-form LLM)
- `ProgramPlan` with explicit `CellReference` coordinates
- Coverage-aware slot allocation
- Hierarchical BM25 (report -> table -> row)
- Deterministic compiler with safety constraints
- `CompoundQueryIntent` model for multi-stage questions (argmax/argmin selection -> target lookup)
- `SelectionStage` / `TargetStage` decomposition

### 4.2 minhduonq/dsdai

**Best architectural analysis.** Detailed survey of 1,012 questions with real stats.

| Aspect | Detail |
|---|---|
| Status | Scaffold complete (skills, config, package). Core modules are empty shells. |
| Architecture | 6-stage: parse question -> retrieve docs -> retrieve tables -> normalize tables -> generate pandas query -> execute + repair |
| Key insight | Two processing paths: fast path (~65% of Qs: single company, single year, simple lookup) needs no LLM; LLM path (~35%: multi-year, derived metrics, multi-company). |
| Question stats | 21.9% ty dong, 20.3% trieu dong, 19.6% %/phan tram, 7.0% nghin ty, 5.7% tram ty. 35.2% multi-year. 36.0% no ticker (name only). 16.3% >=3 tickers. |
| doc_stock.csv | Sole authority for company names. LLM must never guess ticker from memory. |
| Unit handling | Full vocabulary: nghin ty, tram ty, lan, co phieu, USD. OCR often corrupts "Don vi" row. |
| Fiscal year | Not always Dec 31. HSG ends Sep 30. Must read from report header. |
| Retrieval | Doc retrieval by law (metadata filter), not embedding. Embedding only at table level within filtered reports. |

**Reusable patterns:**

- Two-path routing (fast/LLM) based on question complexity
- `code_stock.csv` as sole ticker authority
- Unit vocabulary with OCR-corruption handling
- Plan coverage metrics: question_parsed -> report_retrieved -> table_retrieved -> evidence_complete -> schema_linked -> plan_valid -> execution_success
- Per-phase error taxonomy
- `PLAN.md` survey data: hard stats on what actually drives difficulty

### 4.3 minhmnq/KMA

**5-day team sprint plan.** Three roles: Testing Lead, Data/ML Lead, Tech Lead.

| Aspect | Detail |
|---|---|
| Status | Day 1-2 done. Full catalog: 146,246 tables, 6.2M cells indexed. |
| Architecture | SQLite FTS5 BM25 + financial synonyms + RapidFuzz reranking |
| Key stat | 1,012/1,012 questions have candidates. Top-1 metadata consistency 100%. |
| Extraction | HTML table parser, number normalizer, context enricher, cell index, document metadata |
| Retrieval | `fts_retriever.py`: FTS5 with synonym expansion + fuzzy reranking |
| Reasoning | Template-based for common ops; LLM (<=14B) only for schema linking edge cases |
| Rules | Open models only, released before 2026-06-01, <=14B params |

**Reusable patterns:**

- RapidFuzz reranking after BM25 (cheap, effective)
- Financial synonym dictionary for Vietnamese terms
- Cell index for row-level retrieval
- Day-based sprint plan with clear gates
- Stratified sampling for test sets (banks, securities, enterprises)

### 4.4 NQKhaixyz/aiguru

**Not found.** Repo deleted or private. Referenced in docs only.

---

## 5. Literature and Research Insights

### From conversation notes

| Source | Key Insight |
|---|---|
| VLSP ViNumQA (2025) | Subtask constrained <=13B; effective patterns: Markdown table, program/execution-oriented training, n-sampling, majority vote. ViFinQA harder because retrieval required first. |
| Financial table QA (2026) | BM25 can beat dense retrieval on financial docs. Hybrid + neural reranking outperforms single retrievers. HyDE/multi-query not always beneficial for precise numeric queries. |
| Leaderboard analysis | Docs F2 is the main execution driver. Tables F2 is surprisingly weak. Winner profile: perfect docs, broken table IDs. |

### Recommended architecture (synthesized)

```
Question
  -> ticker/year/scope metadata extraction
  -> hybrid report/table/row retrieval (BM25 + dense + metadata)
  -> evidence closure + raw CSV normalization
  -> typed program DSL (not free-form LLM)
  -> restricted Pandas compiler (sandboxed)
  -> unit/provenance validation
  -> canonical table-ID serialization
  -> vote across N candidates (n-sampling + majority)
```

---

## 6. Our Codebase State

### What we have

| Module | File | Purpose |
|---|---|---|
| OCR table parsing | `src/vifinqa/tables.py` | Parse HTML tables from OCR .txt, extract metadata |
| BM25 retrieval | `src/vifinqa/retrieval.py` | Metadata-gated BM25 over OCR table rows (749 lines) |
| Document gate | `src/docs.py` | Company/year/scope filtering, submission packaging |
| Numeric extraction | `src/vifinqa/answers.py` | Parse OCR numbers, first-numeric-cell extraction |
| Dense retrieval | `src/vifinqa/dense.py` | E5 dense index + fusion with sparse |
| Reranking | `src/vifinqa/rerank.py` | mMARCO reranker integration |
| V2 evaluation | `src/vifinqa/evaluation_v2.py` | Slot-level retrieval evaluation (592 lines) |
| Source binding | `src/vifinqa/review.py` | Validate source-cell bindings |
| Production run | `run.py` | End-to-end: questions -> submission.zip |
| Validator | `scripts/validate_submission.py` | Strict package schema check |

### Retrieval modes implemented

- `baseline` -- fixed top-5 contextual BM25
- `report-coverage` -- **default**; reserves 1 table per gated report, fills rest by relevance
- `role-coverage` -- multi-year role fusion
- `evidence-slots` -- slot-aware selection
- `metric-coverage` -- metric-focused query tokens
- `field-aware` / `rank-fusion` -- field-weighted scoring
- `dense-hybrid` -- sparse + dense union + RRF

### What we lack (vs competitors)

| Gap | Impact | Difficulty |
|---|---|---|
| No question parser (entity/year/scope extraction) | Cannot route fast vs LLM path | Medium |
| No unit vocabulary / normalization | Wrong answer scale | Medium |
| No typed program DSL | Must use free-form LLM | Medium-High |
| No safe Pandas compiler | No execution validation | Medium |
| No coverage-aware slot allocation | Multi-year Qs fail | Medium |
| No multi-report answer logic | ~35% of Qs need it | High |
| No n-sampling / voting | Single-shot answers only | High |
| No local eval loop (gold labels) | Cannot measure improvements | Easy |
| Gold-150 has no answer programs | Cannot test execution accuracy | Blocker |

### Test suite

33 tests passing. Covers: retrieval modes, dense fusion, reranker representation,
answer parsing, source bindings, evaluation v2, pipeline packaging.

---

## 7. Next Tasks (Priority Order)

### P0: Unblocks everything else

1. **V2 calibration gates** -- Run 20-record calibration, hit gates (Jaccard >= 0.80,
   slot agree >= 0.75, operation agree >= 0.80). Without this, no development labels exist.
2. **Local eval loop** -- `evaluate_v2_retrieval` on frozen dev set. Every change must
   be measured against a baseline. No gold labels for execution yet, but retrieval
   metrics are measurable.

### P1: Retrieval quality (main score driver)

3. **Question parser** -- Extract `(ticker, years, scope, metric, unit, operation)` from
   question text. Use `code_stock.csv` as sole ticker authority. Route fast path vs LLM path.
4. **Report-gate repair** -- Improve entity/year/scope resolution. Document selection is
   strong but table report coverage still misses on some multi-report Qs.
5. **Slot obligations** -- Entity x year x scope x metric x role. Reserve 1 table per
   obligation, fill rest by baseline. Beat fixed top-5 on multi-report Qs.
6. **Unit vocabulary** -- Full Vietnamese unit mapping: ty, trieu, nghin ty, tram ty,
   %, lan, co phieu. Handle OCR corruption of "Don vi" rows.

### P2: Execution path (after retrieval stable)

7. **Typed program DSL** -- `ProgramPlan` with `CellReference` coordinates (copy TanKai
   pattern). Operations: lookup, difference, growth, ratio, sum, average, min, max,
   ROE, ROA, debt-to-equity.
8. **Safe Pandas compiler** -- Sandboxed: no imports, no file access, no loops. Read only
   bound DataFrame cells via `.iloc[row, col]`. Convert monetary inputs to base VND,
   apply operation, convert to requested output unit.
9. **Multi-report answer logic** -- Extend `answers.py` beyond single-report lookup.
   Handle sum/delta/ratio across reports.
10. **N-sampling + voting** -- Generate N candidate programs, execute each, vote on
    consistent answers (VLSP ViNumQA pattern).

### P3: Optional upgrades (gate first)

11. **Dense union** -- Only if sparse top-50 miss rate is material on v2 dev.
12. **Rerank (mmarco)** -- Same gate; no default until measured.
13. **Blueprint rewrite** -- Rewrite `table_retrieval_next_approaches.md` to reflect
    text2pandas reality (drop SQL-first assumptions).

### Not now

- More retrieval modes without measurement
- Bigger models (stay <=14B)
- Answer guessing without cell binding
- Precomputed-answer shortcut (scorer-specific, breaks on private)

---

## 8. Key Data Facts

| Fact | Detail |
|---|---|
| Corpus path | `data/raw/vifinqa/financial_statements/<TICKER>/<YEAR>/<DOC_ID>/<DOC_ID>_extracted.txt` |
| Table ID format | `<doc_id>\|<line_no>` where line_no is 1-indexed `<table>` start line |
| Questions | `data/raw/vifinqa/questions/questions.jsonl` -- only `id` + `question` |
| Company map | `data/raw/vifinqa/code_stock.csv` -- ticker <-> company name (sole authority) |
| Gold-150 | `annotations/gold_150.jsonl` -- retrieval labels only, no answer programs |
| Gold-150 evidence | `data/results/e027_gold_150/evidence.csv` -- preserved E027 bindings |
| V2 queues | `annotations/v2_queues/` -- sampling manifest + calibration/development splits |
| OCR format | Page markers `===== PAGE N =====`, tables as inline `<table>` HTML on single lines |
| Scope convention | "cong ty me" -> separate; otherwise -> consolidated (default) |
| Year convention | "cuoi nam X" -> balance sheet at 31/12/X; "nam X" -> income/cash flow for year X |
| Fiscal year edge | HSG ends Sep 30, not Dec 31. Read from report header. |
| Submissions limit | 10/day public, 5 total private. Always validate locally first. |
| Answer tolerance | <=0.02% relative; % returned as percentage points |
| Private scoring | Retrieval + answer + execution; no empty tables, no constant-return Pandas |

---

## 9. Git State

```
feat/cleanup  530359a  [origin/feat/cleanup]  chore: land retrieval modes, drop local run bloat
main          bb7a295  research(results): audit public top submission
```

Tests: 33 passing. Output junk wiped. Research dump removed from repo (was 451M).
