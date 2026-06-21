#!/usr/bin/env bash
# Full benchmark eval + Pareto/KPI plots, multi-seed.
# Usage: bash scripts/05_eval.sh [N_PER_BENCHMARK] [ADAPTER_DIR]
#   N_PER_BENCHMARK : items/benchmark (0 or empty = full held-out set)
#   ADAPTER_DIR     : trained router/GRPO adapter (default outputs/router-seed0)
# Model-agnostic via MODEL=... ; seeds via SEEDS=... (default 0,1,2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"   # 3B/7B = one-flag switch
N="${1:-200}"                                   # items per benchmark (0 = full)
ADAPTER="${2:-outputs/router-seed0}"
VERIFIER="${VERIFIER:-outputs/verifier-400m/best.pt}"
SEEDS="${SEEDS:-0,1,2}"                          # multi-seed Pass@1 variance
BENCH="${BENCH:-all}"
mkdir -p results logs
NFLAG=""; [ "${N}" != "0" ] && NFLAG="--n ${N}"
VERIFIER_FLAG=""; [ -f "${VERIFIER}" ] && VERIFIER_FLAG="--verifier-ckpt ${VERIFIER}"

# 1) Baselines: base model, no router, the pre-RL anchors.
python eval/run_benchmarks.py --benchmark "${BENCH}" --model-name "${MODEL}" \
  --route-mode never_think --seeds "${SEEDS}" \
  --tag baseline_never_think --out results/baseline_never_think.json ${NFLAG} \
  2>&1 | tee logs/eval_baseline_never_think.log
python eval/run_benchmarks.py --benchmark "${BENCH}" --model-name "${MODEL}" \
  --route-mode always_think --seeds "${SEEDS}" \
  --tag baseline_always_think --out results/baseline_always_think.json ${NFLAG} \
  2>&1 | tee logs/eval_baseline_always_think.log

# 2) Router: trained adapter + verifier, adaptive routing (the system).
python eval/run_benchmarks.py --benchmark "${BENCH}" --model-name "${MODEL}" \
  --adapter "${ADAPTER}" ${VERIFIER_FLAG} --route-mode model \
  --seeds "${SEEDS}" --tag router --out results/router.json ${NFLAG} \
  2>&1 | tee logs/eval_router.log

# 3) Fixed-policy points (post-RL adapter) for the Pareto chart.
python eval/run_benchmarks.py --benchmark "${BENCH}" --model-name "${MODEL}" \
  --adapter "${ADAPTER}" --route-mode never_think \
  --seeds "${SEEDS}" --tag system1 --out results/system1.json ${NFLAG} \
  2>&1 | tee logs/eval_system1.log
python eval/run_benchmarks.py --benchmark "${BENCH}" --model-name "${MODEL}" \
  --adapter "${ADAPTER}" --route-mode always_think \
  --seeds "${SEEDS}" --tag system2 --out results/system2.json ${NFLAG} \
  2>&1 | tee logs/eval_system2.log

# 4) Plots + KPI delta table (baseline anchor = always_think pre-RL number).
python eval/plots.py --baseline results/baseline_always_think.json \
  --runs results/router.json results/system1.json results/system2.json \
         results/baseline_never_think.json \
  --outdir results/figures 2>&1 | tee logs/plots.log

echo "Eval done. See results/figures/kpi_table.md"
