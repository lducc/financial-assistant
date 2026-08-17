#!/usr/bin/env bash
# Fine-tune the reranker and score both A/B cells on a rented GPU, unattended.
#
#   bash vast/run.sh
#
# Expects training_linked.jsonl and pairs_bench_v4.jsonl in ./data.
# Edit MODEL_NAME and QUANTIZATION at the top of the two scripts once; the paths
# that change between runs arrive as arguments.
set -euo pipefail

pip install -q -U torch transformers peft accelerate

python kaggle/train_reranker.py  data/training_linked.jsonl adapter
python kaggle/rerank_qwen_8b.py  data/pairs_bench_v4.jsonl scores_tuned.jsonl adapter
python kaggle/rerank_qwen_8b.py  data/pairs_bench_v4.jsonl scores_base.jsonl

echo "done: scores_tuned.jsonl scores_base.jsonl adapter/"
