# Fine-tuning the reranker on a rented GPU

The pool holds 87.4% of the gold live and the shipped order surfaces 61.5% of it,
so 0.22 F2 sits in the ranking stage and every hand-written rule for it has lost:
coverage ordering -0.008, account-code promotion -0.050, listwise -0.046, every
cap negative. A better scorer is what is left.

## The box

Any 24 GB card. 8B in fp16 is 16.4 GB, which fits one 4090 and removes the int8
outlier path that made scores depend on how batches were packed. Allow 50 GB of
disk for the model.

At about $0.40/hour: model pull 0.4 h, training 1 h, both scoring cells 1 h.
Roughly a dollar, so a $3 balance covers the experiment and a retry.

## Run it

```
scp output/rerank/training_linked.jsonl output/rerank/pairs_bench_v4.jsonl box:~/data/
ssh box 'cd repo && bash vast/run.sh'
```

Set `MODEL_NAME = "Qwen/Qwen3-Reranker-8B"` and `QUANTIZATION = "fp16"` at the top
of both scripts first — those hold across the three runs, so they are edited once
rather than passed. `MAX_GROUPS` is the session budget: one group is one optimiser
step, so read the step time the trainer prints and set it to the hours you want to
spend.

Both cells are scored in the same process on the same pairs file, so the
comparison carries no packing drift — the A/A stratum of the last such pair read
exactly 0.0000.

## Then, locally

```
python3 scripts/compare_rerank_runs.py --pairs output/rerank/pairs_bench_v4.jsonl \
    --control output/rerank/scores_base.jsonl \
    --treatment output/rerank/scores_tuned.jsonl --aa-items 99
```

Accept on the standing rule and nothing softer: the mean up by more than 0.02, the
interval excluding zero, no difficulty tier down. The live leaderboard resolves
about 0.017, so a smaller margin cannot be confirmed where it counts.

## The data

`output/rerank/training_linked.jsonl`, from `scripts/synthesize_training.py`:
10,135 groups, 2.39 positives each, benchmark reports excluded.

Positives are linked by the figure rather than the label — a filing restates the
same number in the note and the comparative — which recovers 0.903 of the
benchmark's gold tables where label matching recovers 0.775. Negatives are BM25's
top tables in the same report that do not carry the figure, so 0.069 of them share
no content word with the item, against 0.635 for the first attempt. Queries are
the real questions slotted on item, year and company and sampled back.

Known gaps, none of them correctness: positives are 0.644 precise against the
benchmark's gold, every query names one line item where 55% of real questions name
two or three, and anchors need a `Mã số` row so note-only items are absent.
