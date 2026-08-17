#!/usr/bin/env bash
# Fine-tune the reranker and score both A/B cells on a rented GPU, unattended.
#
#   bash vast/run.sh
#
# Expects training_linked.jsonl, pairs_bench_v6.jsonl, pairs_v6.jsonl and the
# existing scores_v4.jsonl in ./data.
# Edit MODEL_NAME and QUANTIZATION at the top of the two scripts once; the paths
# that change between runs arrive as arguments.
set -euo pipefail

# The template ships torch built against its own CUDA; upgrading it here is how
# a paid box breaks. Only what the template lacks.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install -q transformers peft accelerate

# The benchmark A/B first: it is the cheap half and it decides whether the rest
# of the box's time is worth spending.
python kaggle/train_reranker.py  data/training_linked.jsonl adapter
python kaggle/rerank_qwen_8b.py  data/pairs_bench_v6.jsonl scores_tuned.jsonl adapter
python kaggle/rerank_qwen_8b.py  data/pairs_bench_v6.jsonl scores_base.jsonl

# The augmented candidates at full corpus, base model. SKIP_PATH means only the
# item-carrying tables are judged — about 4,400 pairs rather than 55,000 — so the
# scored-expansion configuration ships whether or not the adapter works. Set
# SKIP_PATH = "data/scores_v4.jsonl" at the top of the scorer before running.
python kaggle/rerank_qwen_8b.py  data/pairs_v6.jsonl scores_full_new.jsonl

# Only if the adapter cleared the gate on the benchmark. Costs the full pass.
python kaggle/rerank_qwen_8b.py  data/pairs_v6.jsonl scores_full_tuned.jsonl adapter

echo "done: scores_tuned scores_base scores_full_new scores_full_tuned adapter/"
