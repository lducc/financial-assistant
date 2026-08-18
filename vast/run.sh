#!/usr/bin/env bash
# Fine-tune the reranker and settle both open questions on a rented GPU.
#
#   bash vast/run.sh
#
# Expects ./data to hold training_linked.jsonl, pairs_bench_v6.jsonl,
# pairs_bench_v7.jsonl, pairs_v6.jsonl and the existing scores_v4.jsonl.
# The scripts are already set to the 8B: fp16 for scoring, nf4 for training.
set -euo pipefail

# The template ships torch built against its own CUDA; upgrading it here is how
# a paid box breaks. Install only what the template lacks.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -q transformers peft accelerate bitsandbytes

# --- the benchmark, three readings of one candidate set ------------------------
# One process, so nothing but the named change differs between them. base is the
# control for both: scores_tuned changes the weights, sweep_full changes the text.
python kaggle/rerank_qwen_8b.py data/pairs_bench_v6.jsonl scores_base.jsonl
python kaggle/train_reranker.py data/training_linked.jsonl adapter
python kaggle/rerank_qwen_8b.py data/pairs_bench_v6.jsonl scores_tuned.jsonl adapter
python kaggle/rerank_qwen_8b.py data/pairs_bench_v7.jsonl sweep_full.jsonl

# --- the full corpus -----------------------------------------------------------
# The sixth argument is SKIP_PATH: scores_v4 already holds the original pool, so
# only the item-carrying tables are judged, about 4,400 pairs rather than 55,000.
# This ships the scored-expansion configuration whether or not anything above won.
python kaggle/rerank_qwen_8b.py data/pairs_v6.jsonl scores_full_new.jsonl \
    "" full v1 data/scores_v4.jsonl

echo "done: scores_base scores_tuned sweep_full scores_full_new adapter/"
echo "if the adapter or the re-texting cleared the gate, run its full pass next"
