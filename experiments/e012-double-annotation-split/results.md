# E012 results — reviewer subset ready

The selection test passed and produced 30 distinct, reproducible IDs from the
frozen 120-record E003 queue.  All seven operation-hint families appear:
aggregate 5, average 6, difference 6, selector 4, growth/change 3, lookup 3,
and ratio/percent 3.  The subset also includes every primary stratum represented
in the source queue, with extra capacity allocated round-robin to larger strata.

The local reviewer file is
`data/derived/proxy_queue/double_annotation_queue.jsonl`.  It is still blank;
this result establishes coverage and reproducibility, not inter-annotator
agreement or system accuracy.
