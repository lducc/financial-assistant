# ViFinQA internal gold-150

Generated: `2026-08-09`

## Status

- 150 distinct official questions.
- 150/150 annotation-contract validation passes.
- 150/150 pandas programs recompute their numeric answer.
- 150/150 source table IDs, start lines, raw cells, numeric parsing, and units rechecked from raw reports.
- Evidence split: {'1': 50, '2': 50, '3+': 50}.
- Difficulty split: {'easy': 46, 'hard': 82, 'medium': 22}.

This is self-reviewed internal gold, not organizer-released ground truth or independent double annotation. Candidate evidence was seeded from submission 2333, then every retained operand was relocated in raw source and every formula was reconstructed and executed. This avoids treating submission predictions as truth while preserving useful candidate discovery.

## Operations

| Operation | Questions |
| --- | ---: |
| lookup | 50 |
| difference | 26 |
| selector | 19 |
| average | 16 |
| growth | 15 |
| ratio | 10 |
| sum | 8 |
| select_then_ratio | 3 |
| conditional_next_period_ratio | 1 |
| conditional_count | 1 |
| rank_filter_count | 1 |

## Expected functions

| Function | Questions |
| --- | ---: |
| lookup | 150 |
| scale_unit | 115 |
| join | 100 |
| sum | 60 |
| subtract | 26 |
| argmax_or_argmin | 24 |
| average | 17 |
| divide | 15 |
| percent_change | 15 |
| multi_hop | 5 |
| filter | 3 |
| count | 2 |
| rank | 1 |

## Intended use

Use fixed IDs for table retrieval recall@k, precision@k, MRR, operand coverage, routing accuracy, and executable-answer regression. Do not tune and report on same set without a frozen holdout. `gold_tables` uses `report_id|source_start_line`; `row_column_bindings` uses span-expanded logical grids while retaining raw cell strings.
