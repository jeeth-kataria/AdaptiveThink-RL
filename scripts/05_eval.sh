#!/usr/bin/env bash
# Day 8-9: full benchmark eval + Pareto/KPI plots.
# Usage: bash scripts/05_eval.sh [N_PER_BENCHMARK] [ADAPTER_DIR]
set -e
N=${1:-200}                       # items per benchmark (use 0/empty for full)
ADAPTER=${2:-outputs/router-seed0}
VERIFIER=outputs/verifier-400m/best.pt
mkdir -p results logs
NFLAG=""; [ "$N" != "0" ] && NFLAG="--n $N"

# 1) Baseline: base reasoner, full CoT, no router (the pre-RL number).
python eval/run_benchmarks.py --benchmark all --route-mode always_think \
  --tag baseline --out results/baseline.json $NFLAG 2>&1 | tee logs/eval_baseline.log

# 2) Router: trained adapter + verifier, adaptive routing (the system).
python eval/run_benchmarks.py --benchmark all --adapter "$ADAPTER" \
  --verifier-ckpt "$VERIFIER" --route-mode model \
  --tag router --out results/router.json $NFLAG 2>&1 | tee logs/eval_router.log

# 3) Fixed-policy points for the Pareto chart.
python eval/run_benchmarks.py --benchmark all --adapter "$ADAPTER" \
  --route-mode never_think --tag system1 --out results/system1.json $NFLAG
python eval/run_benchmarks.py --benchmark all --adapter "$ADAPTER" \
  --route-mode always_think --tag system2 --out results/system2.json $NFLAG

# 4) Plots + KPI delta table.
python eval/plots.py --baseline results/baseline.json \
  --runs results/router.json results/system1.json results/system2.json \
  --outdir results/figures

echo "Eval done. See results/figures/kpi_table.md"
