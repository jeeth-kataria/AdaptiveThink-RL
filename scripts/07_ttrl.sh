#!/usr/bin/env bash
# Day 8 (optional Idea A): Test-Time RL on unlabeled MMLU, warm-started from the router.
# Usage: bash scripts/07_ttrl.sh [SEED]
set -e
SEED=${1:-0}
mkdir -p outputs/ttrl-seed${SEED} logs

python src/adaptivethink/ttrl/ttrl.py \
  --adapter outputs/router-seed${SEED} \
  --output-dir outputs/ttrl-seed${SEED} \
  --n 500 --steps 300 --group-size 8 \
  --kl-beta 1e-3 --lambda-tok 1e-3 \
  --seed ${SEED} 2>&1 | tee logs/ttrl_seed${SEED}.log

echo "TTRL adapter -> outputs/ttrl-seed${SEED}"
echo "Eval the TTRL ablation:"
echo "  python eval/run_benchmarks.py --benchmark mmlu --adapter outputs/ttrl-seed${SEED} \\"
echo "    --verifier-ckpt outputs/verifier-400m/best.pt --route-mode model \\"
echo "    --tag ttrl --out results/ttrl.json"
