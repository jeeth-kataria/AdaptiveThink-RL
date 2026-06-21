#!/usr/bin/env bash
# Legacy router path: full GRPO router training (efficiency layer, NOT the headline).
# Model-agnostic — defaults to Qwen2.5-1.5B-Instruct; switch via MODEL=... .
# The headline single-run pipeline is scripts/run_all_24h.sh (rl/ Dr.GRPO core);
# this trains the standalone router adapter consumed by eval --route-mode model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"   # 3B/7B = one-flag switch
SEED="${1:-0}"
OUT="outputs/router-seed${SEED}"
VERIFIER="${VERIFIER:-outputs/verifier-400m/best.pt}"
mkdir -p "${OUT}" logs

# Materialize GRPO training data if missing (GSM8K train {question, answer};
# difficulty is computed at train time via the verifier).
if [ ! -f data/gsm8k_train_labelled.jsonl ]; then
  echo "data/gsm8k_train_labelled.jsonl missing — running scripts/02b_prep_train_data.sh"
  bash scripts/02b_prep_train_data.sh
fi

python src/adaptivethink/router/train_grpo.py \
  --model "${MODEL}" \
  --data data/gsm8k_train_labelled.jsonl \
  --verifier-ckpt "${VERIFIER}" \
  --output-dir "${OUT}" \
  --steps 1500 --batch 1 --group-size 8 \
  --lr 1e-5 --kl-beta 5e-3 --max-seq-len 2048 \
  --seed "${SEED}" 2>&1 | tee "logs/grpo_seed${SEED}.log"
