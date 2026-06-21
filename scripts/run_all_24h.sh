#!/usr/bin/env bash
# =============================================================================
# run_all_24h.sh — the ONE 24h AdaptiveThink-RL pipeline.
#
# Target: Qwen2.5-1.5B-Instruct, Dr.GRPO RLVR, single NVIDIA A6000 (48 GB).
# API-free. No teacher. Pure RLVR (exact-match) reward reused from
# src/adaptivethink/router/reward.py. Novelty = CCDD self-difficulty.
#
# Pipeline stages (each banner-delimited, logged to logs/, resumable):
#   0  optional smoke (50-step GRPO sanity)            [SMOKE=1 to enable]
#   1  baseline eval: never_think + always_think       (gsm8k/strategyqa/mmlu)
#   2  CCDD self-difficulty -> data/self_difficulty.jsonl
#   3  Dr.GRPO RLVR training -> outputs/grpo-seed0
#   4  optional self-distilled verifier on self-difficulty labels
#   5  final eval (--seeds 0,1,2) + KPI table via eval/plots.py
#   6  quantize merged adapter -> GGUF Q4_K_M
#
# RESUMABILITY: every stage skips itself if its primary output already exists.
# Re-running the script after an interruption continues from the first
# incomplete stage. Delete a stage's output dir/file to force a re-run.
#
# TIME-BUDGET CUT ORDER (if the 24h wall-clock is tight, drop from the top):
#   1) STAGE 6 quantize        (export QUANTIZE=0)   — demo polish, not a KPI
#   2) STAGE 5 multi-seed      (set EVAL_SEEDS=0)    — single seed still valid
#   3) STAGE 2/4 CCDD+verifier (export CCDD=0)       — falls back to plain RLVR
#   4) STAGE 3 GRPO: stop at last checkpoint >= 250  (lower GRPO_STEPS or just
#      Ctrl-C; --save-steps 50 means a >=250 ckpt is the minimum shippable run)
# =============================================================================
set -euo pipefail

# ── repo root (this script lives in scripts/) ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── activate the project venv (built by ./run.sh setup) so `python` resolves ──
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "[run_all_24h] WARNING: venv missing at ${VENV_DIR} — run './run.sh setup' first." >&2
fi
# PYTHONPATH fallback so 'adaptivethink' imports even if the editable install failed.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
# Optional creds (HF_TOKEN / WANDB_*), if present.
[ -f "${REPO_ROOT}/.env" ] && { set -a; . "${REPO_ROOT}/.env"; set +a; }

# ── tunables (all overridable from the environment) ──────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"   # 3B/7B = one-flag switch
DATASETS="${DATASETS:-gsm8k,strategyqa,aqua}"  # train pool (45/30/25)
GRPO_OUT="${GRPO_OUT:-outputs/grpo-seed0}"
GRPO_STEPS="${GRPO_STEPS:-400}"                 # Dr.GRPO RLVR steps
GROUP_SIZE="${GROUP_SIZE:-8}"
KL_BETA="${KL_BETA:-0.0}"                        # modern default: no KL/ref model
LR="${LR:-1e-6}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1024}"
SEED="${SEED:-0}"

SELF_DIFF_FILE="${SELF_DIFF_FILE:-data/self_difficulty.jsonl}"
CCDD_K="${CCDD_K:-8}"                            # rollouts/question for solve-rate
CCDD_N="${CCDD_N:-2000}"                         # questions to self-label

VERIFIER_OUT="${VERIFIER_OUT:-outputs/verifier-400m/best.pt}"

EVAL_N="${EVAL_N:-200}"                          # items/benchmark (0 = full)
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2}"               # set to 0 to drop multi-seed
EVAL_BENCH="${EVAL_BENCH:-all}"                 # gsm8k/strategyqa/mmlu maintained

GGUF_OUT="${GGUF_OUT:-outputs/gguf/router-1p5b-Q4_K_M.gguf}"

# ── stage toggles ────────────────────────────────────────────────────────────
SMOKE="${SMOKE:-0}"          # 1 = run the 50-step GRPO smoke first
CCDD="${CCDD:-1}"            # 1 = CCDD self-difficulty + self-distilled verifier
QUANTIZE="${QUANTIZE:-1}"   # 1 = export GGUF at the end

mkdir -p logs data outputs results

# ── helpers ──────────────────────────────────────────────────────────────────
banner() {
  echo ""
  echo "============================================================================="
  echo "== $* "
  echo "==   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================================="
}

NFLAG=""; [ "${EVAL_N}" != "0" ] && NFLAG="--n ${EVAL_N}"

# ── ETA banner ───────────────────────────────────────────────────────────────
banner "AdaptiveThink-RL 24h pipeline | model=${MODEL}"
cat <<EOF
  Datasets (train) : ${DATASETS}    (eval: gsm8k/strategyqa/mmlu, held-out)
  Dr.GRPO          : steps=${GRPO_STEPS} group=${GROUP_SIZE} kl=${KL_BETA} lr=${LR}
                     max_prompt=${MAX_PROMPT_LEN} max_completion=${MAX_COMPLETION_LEN}
  CCDD             : k=${CCDD_K} n=${CCDD_N} -> ${SELF_DIFF_FILE}   (enabled=${CCDD})
  Eval             : n=${EVAL_N} seeds=${EVAL_SEEDS} bench=${EVAL_BENCH}
  Quantize         : enabled=${QUANTIZE}

  Rough ETA on a single A6000 (1.5B):
    [1] baseline eval ....... ~0.5 h
    [2] CCDD self-difficulty  ~2-3 h   (k=${CCDD_K} x n=${CCDD_N} rollouts)
    [3] Dr.GRPO ${GRPO_STEPS} steps .. ~14-18 h   (the headline stage)
    [4] verifier ............ ~0.5 h
    [5] final eval x3 seeds .. ~1.5 h
    [6] quantize GGUF ....... ~0.5 h
    -----------------------------------------------------------------
    TOTAL ................... ~19-24 h   (cut order in the header)
EOF

# =============================================================================
# STAGE 0 — optional smoke test (50-step GRPO sanity)
# =============================================================================
if [ "${SMOKE}" = "1" ]; then
  banner "STAGE 0/6 — smoke test (50-step GRPO)"
  bash scripts/00_smoke_grpo.sh 2>&1 | tee logs/00_smoke.log
else
  banner "STAGE 0/6 — smoke test SKIPPED (set SMOKE=1 to enable)"
fi

# =============================================================================
# STAGE 1 — baseline eval (no router): never_think + always_think
# These are the pre-RL anchors for the Pareto/KPI table.
# =============================================================================
banner "STAGE 1/6 — baseline eval (never_think + always_think)"
if [ -f results/baseline_never_think.json ] && [ -f results/baseline_always_think.json ]; then
  echo "[skip] baseline eval outputs already exist."
else
  python eval/run_benchmarks.py --benchmark "${EVAL_BENCH}" --model-name "${MODEL}" \
    --route-mode never_think --tag baseline_never_think \
    --out results/baseline_never_think.json ${NFLAG} \
    2>&1 | tee logs/01_eval_baseline_never_think.log
  python eval/run_benchmarks.py --benchmark "${EVAL_BENCH}" --model-name "${MODEL}" \
    --route-mode always_think --tag baseline_always_think \
    --out results/baseline_always_think.json ${NFLAG} \
    2>&1 | tee logs/01_eval_baseline_always_think.log
fi

# =============================================================================
# STAGE 2 — CCDD self-difficulty (the API-free novelty)
# difficulty = 1 - base-model empirical solve-rate over K rollouts/question.
# Writes {question, answer, difficulty} rows (verifier-compatible schema).
# =============================================================================
if [ "${CCDD}" = "1" ]; then
  banner "STAGE 2/6 — CCDD self-difficulty (k=${CCDD_K} n=${CCDD_N})"
  if [ -f "${SELF_DIFF_FILE}" ]; then
    echo "[skip] ${SELF_DIFF_FILE} already exists."
  else
    python -m adaptivethink.rl.self_difficulty \
      --model "${MODEL}" --datasets "${DATASETS}" \
      --k "${CCDD_K}" --n "${CCDD_N}" --out "${SELF_DIFF_FILE}" \
      2>&1 | tee logs/02_self_difficulty.log
  fi
else
  banner "STAGE 2/6 — CCDD self-difficulty SKIPPED (CCDD=0; plain RLVR)"
fi

# =============================================================================
# STAGE 3 — Dr.GRPO RLVR training (the headline stage)
# Reward = pure RLVR exact-match (reused from router/reward.py). No teacher.
# Curriculum filter (solve_rate in {0,1} dropped) is applied inside the trainer
# when --self-difficulty-file is provided.
# =============================================================================
banner "STAGE 3/6 — Dr.GRPO RLVR (steps=${GRPO_STEPS})"
if [ -f "${GRPO_OUT}/adapter_model.safetensors" ] || [ -f "${GRPO_OUT}/adapter_model.bin" ]; then
  echo "[skip] GRPO adapter already present in ${GRPO_OUT}."
else
  # CCDD off by default (empty file -> trainer trains the full pool); use an
  # array so the empty value survives as a real argv element (no word-split).
  GRPO_DIFF_ARGS=( --self-difficulty-file "" )
  if [ "${CCDD}" = "1" ] && [ -f "${SELF_DIFF_FILE}" ]; then
    GRPO_DIFF_ARGS=( --self-difficulty-file "${SELF_DIFF_FILE}" )
  fi
  python -m adaptivethink.rl.drgrpo_train \
    --model "${MODEL}" --datasets "${DATASETS}" \
    "${GRPO_DIFF_ARGS[@]}" \
    --out "${GRPO_OUT}" \
    --steps "${GRPO_STEPS}" --seed "${SEED}" \
    --loss dr_grpo --kl "${KL_BETA}" \
    --group-size "${GROUP_SIZE}" --lr "${LR}" \
    --max-prompt-length "${MAX_PROMPT_LEN}" \
    --max-completion-length "${MAX_COMPLETION_LEN}" \
    --save-steps 50 \
    2>&1 | tee logs/03_drgrpo.log
fi

# =============================================================================
# STAGE 4 — optional self-distilled verifier on CCDD labels
# Trains the difficulty verifier on data/self_difficulty.jsonl (no teacher).
# =============================================================================
if [ "${CCDD}" = "1" ]; then
  banner "STAGE 4/6 — self-distilled verifier (on ${SELF_DIFF_FILE})"
  if [ -f "${VERIFIER_OUT}" ]; then
    echo "[skip] verifier checkpoint ${VERIFIER_OUT} already exists."
  elif [ ! -f "${SELF_DIFF_FILE}" ]; then
    echo "[skip] ${SELF_DIFF_FILE} missing — cannot train verifier without it."
  else
    mkdir -p "$(dirname "${VERIFIER_OUT}")"
    python src/adaptivethink/verifier/train.py \
      --train "${SELF_DIFF_FILE}" \
      --eval  "${SELF_DIFF_FILE}" \
      --out   "${VERIFIER_OUT}" \
      --epochs 3 --batch 32 --lr 2e-5 \
      2>&1 | tee logs/04_verifier.log
  fi
else
  banner "STAGE 4/6 — verifier SKIPPED (CCDD=0)"
fi

# =============================================================================
# STAGE 5 — final eval (multi-seed) + KPI table
# Router (route-mode model) vs the never/always baselines, then plots.
# =============================================================================
banner "STAGE 5/6 — final eval (--seeds ${EVAL_SEEDS}) + KPI table"
VERIFIER_FLAG=""; [ -f "${VERIFIER_OUT}" ] && VERIFIER_FLAG="--verifier-ckpt ${VERIFIER_OUT}"

if [ -f results/router.json ]; then
  echo "[skip] results/router.json already exists."
else
  python eval/run_benchmarks.py --benchmark "${EVAL_BENCH}" --model-name "${MODEL}" \
    --adapter "${GRPO_OUT}" ${VERIFIER_FLAG} --route-mode model \
    --seeds "${EVAL_SEEDS}" --tag router \
    --out results/router.json ${NFLAG} \
    2>&1 | tee logs/05_eval_router.log
fi

# Fixed-policy points for the Pareto chart (post-RL adapter).
if [ ! -f results/system1.json ]; then
  python eval/run_benchmarks.py --benchmark "${EVAL_BENCH}" --model-name "${MODEL}" \
    --adapter "${GRPO_OUT}" --route-mode never_think \
    --seeds "${EVAL_SEEDS}" --tag system1 \
    --out results/system1.json ${NFLAG} \
    2>&1 | tee logs/05_eval_system1.log
fi
if [ ! -f results/system2.json ]; then
  python eval/run_benchmarks.py --benchmark "${EVAL_BENCH}" --model-name "${MODEL}" \
    --adapter "${GRPO_OUT}" --route-mode always_think \
    --seeds "${EVAL_SEEDS}" --tag system2 \
    --out results/system2.json ${NFLAG} \
    2>&1 | tee logs/05_eval_system2.log
fi

# Plots + KPI delta table (baseline anchor = always_think pre-RL number).
python eval/plots.py --baseline results/baseline_always_think.json \
  --runs results/router.json results/system1.json results/system2.json \
         results/baseline_never_think.json \
  --outdir results/figures 2>&1 | tee logs/05_plots.log

# =============================================================================
# STAGE 6 — quantize merged adapter -> GGUF Q4_K_M (on-device demo)
# =============================================================================
if [ "${QUANTIZE}" = "1" ]; then
  banner "STAGE 6/6 — quantize -> GGUF Q4_K_M"
  if [ -f "${GGUF_OUT}" ]; then
    echo "[skip] ${GGUF_OUT} already exists."
  else
    mkdir -p "$(dirname "${GGUF_OUT}")"
    python src/adaptivethink/quantize/export_gguf.py \
      --base-model "${MODEL}" \
      --adapter "${GRPO_OUT}" \
      --merged-dir outputs/router-merged \
      --out "${GGUF_OUT}" \
      --quant-type Q4_K_M \
      2>&1 | tee logs/06_quantize.log
  fi
else
  banner "STAGE 6/6 — quantize SKIPPED (QUANTIZE=0)"
fi

banner "PIPELINE COMPLETE"
echo "  Adapter   : ${GRPO_OUT}"
echo "  KPI table : results/figures/kpi_table.md"
echo "  GGUF      : ${GGUF_OUT}  (if QUANTIZE=1)"
echo "  Logs      : logs/"
