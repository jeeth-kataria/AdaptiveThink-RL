#!/usr/bin/env bash
# ============================================================================
# AdaptiveThink-RL — THE SINGLE MASTER SCRIPT
# ----------------------------------------------------------------------------
# One entry point you run on the GPU box. It owns dependency resolution
# (the #1 pain), training, evaluation, and deployment for the
# "distillation cold-start -> Dr.GRPO/GRPO with rule-based verifiable rewards"
# pipeline on Qwen2.5-3B-Instruct (single Linux GPU, CUDA 12.x, ~50GB VRAM).
#
#   Usage:  ./run.sh <subcommand> [flags]
#   Subcommands: setup | smoke | baseline | sft | grpo | eval | router
#                quantize | all | help
#
# Everything activates the project .venv first. Stages are idempotent and
# resumable (a stage is skipped if its output already exists; force with
# --force / FORCE=1). Logs go to logs/<stage>.log.
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Repo-root detection + global paths/constants
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
LOG_DIR="$REPO_ROOT/logs"
RESULTS_DIR="$REPO_ROOT/results"
OUT_DIR="$REPO_ROOT/outputs"
LOCK_FILE="$REPO_ROOT/requirements.lock"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$OUT_DIR" "$RESULTS_DIR/baseline" "$RESULTS_DIR/trained" "$RESULTS_DIR/figures"

# --- Model + benchmark defaults (overridable via flags / env) --------------
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"        # fallback: Qwen/Qwen2.5-1.5B-Instruct
DATASETS="${DATASETS:-gsm8k,strategyqa}"          # improve targets; MMLU = maintain
SEEDS="${SEEDS:-0,1,2}"                            # multi-seed RL (high variance)
STEPS="${STEPS:-1500}"                             # GRPO steps
SFT_STEPS="${SFT_STEPS:-500}"
LOSS="${LOSS:-dr_grpo}"                            # dr_grpo (default) | grpo
KL="${KL:-0.0}"                                    # KL off by default (Lavaee SLM finding)
ENTROPY="${ENTROPY:-0.0}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-4096}"
CLIP_HIGHER="${CLIP_HIGHER:-1}"                    # 1 -> --clip-higher, 0 -> --no-clip-higher
DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-1}"        # filter unsolvable-hard for sub-3B
ONE_SHOT="${ONE_SHOT:-0}"                          # 1-shot RLVR ablation flag
SFT_TRACES="${SFT_TRACES:-open-thoughts/OpenThoughts-114k}"  # cold-start CoT traces
RUN_SFT="${RUN_SFT:-0}"                            # 'all' runs SFT only when --sft / RUN_SFT=1

# --- Eval (lm-eval-harness) constants from EVAL RESEARCH --------------------
LM_EVAL_VERSION="0.4.12"
# gsm8k_cot = 8-shot CoT (baked in YAML); mmlu needs explicit --num_fewshot 5;
# bigbench_strategyqa_multiple_choice is zero-shot by construction (acc metric).
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
# Chat model: the Qwen2.5-Instruct base + chat-formatted training -> apply chat
# template to BOTH baseline and trained for an apples-to-apples comparison.
APPLY_CHAT="${APPLY_CHAT:-1}"

# --- Pinned dependency stack (DEPENDENCY RESEARCH, verified 2026-06-18) -----
# vLLM is the strictest torch pinner -> install it FIRST so it brings torch 2.10.0,
# then pin the rest INTO the resolved window. lm-eval LAST with --no-deps.
PY_VER="${PY_VER:-3.12}"                           # vllm 0.19.1 needs >=3.10,<3.14
PIN_VLLM="vllm==0.19.1"
PIN_TORCH="torch==2.10.0"
PIN_TORCHVISION="torchvision==0.25.0"
PIN_TORCHAUDIO="torchaudio==2.10.0"
PIN_UNSLOTH="unsloth==2026.6.7"
PIN_UNSLOTH_ZOO="unsloth_zoo>=2026.6.5"
PIN_TRANSFORMERS="transformers==4.57.6"
PIN_TRL="trl==0.24.0"
PIN_PEFT="peft==0.19.1"
PIN_ACCELERATE="accelerate==1.14.0"
PIN_BNB="bitsandbytes==0.49.2"
PIN_DATASETS="datasets==3.6.0"
PIN_LM_EVAL="lm-eval==${LM_EVAL_VERSION}"
# Repo runtime deps not pulled by the RL stack (kept loose so they never move torch).
EXTRA_RUNTIME_DEPS=(huggingface-hub sentencepiece openai httpx python-dotenv \
  scipy numpy matplotlib pyyaml wandb)
# lm-eval runtime deps (no pins so they cannot upgrade torch/transformers/vllm).
LM_EVAL_RUNTIME_DEPS=(sqlitedict jsonlines tqdm-multiprocess more-itertools \
  evaluate pytablewriter)

# ---------------------------------------------------------------------------
# 1. Logging / banner helpers
# ---------------------------------------------------------------------------
c_reset='\033[0m'; c_blue='\033[1;34m'; c_green='\033[1;32m'
c_yellow='\033[1;33m'; c_red='\033[1;31m'
banner()  { printf "\n${c_blue}========================================================================${c_reset}\n${c_blue}  %s${c_reset}\n${c_blue}========================================================================${c_reset}\n" "$*"; }
info()    { printf "${c_green}[run.sh]${c_reset} %s\n" "$*"; }
warn()    { printf "${c_yellow}[run.sh WARN]${c_reset} %s\n" "$*" >&2; }
die()     { printf "${c_red}[run.sh ERROR]${c_reset} %s\n" "$*" >&2; exit 1; }

# Run a command teeing its output to logs/<stage>.log (and stdout).
log_run() {
  local stage="$1"; shift
  info "logging to $LOG_DIR/$stage.log"
  ( "$@" ) 2>&1 | tee "$LOG_DIR/$stage.log"
  return "${PIPESTATUS[0]}"
}

# Idempotency: skip a stage when its sentinel output exists (unless --force).
FORCE="${FORCE:-0}"
should_skip() {
  local sentinel="$1" stage="$2"
  if [[ "$FORCE" != "1" && -e "$sentinel" ]]; then
    info "SKIP $stage — output already exists: $sentinel  (use --force / FORCE=1 to rerun)"
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# 2. GPU / CUDA detection
# ---------------------------------------------------------------------------
detect_gpu() {
  banner "GPU / CUDA detection"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    nvidia-smi | head -n 12 || true
  else
    warn "nvidia-smi not found. This pipeline targets a single Linux CUDA 12.x GPU."
    warn "setup will still build the venv, but training/eval will fail without a GPU."
  fi
  if command -v nvcc >/dev/null 2>&1; then
    info "nvcc: $(nvcc --version | grep -i release || true)"
  else
    info "nvcc not on PATH (OK — wheels ship their own CUDA runtime)."
  fi
}

# ---------------------------------------------------------------------------
# 3. venv resolution + activation
# ---------------------------------------------------------------------------
pick_python() {
  for p in "python${PY_VER}" python3.12 python3.11 python3.10; do
    if command -v "$p" >/dev/null 2>&1; then echo "$p"; return 0; fi
  done
  # Last resort: a generic python3 in the 3.10-3.13 window.
  if command -v python3 >/dev/null 2>&1; then
    if python3 -c 'import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<(3,14) else 1)' 2>/dev/null; then
      echo python3; return 0
    fi
  fi
  return 1
}

activate_venv() {
  [[ -f "$VENV_DIR/bin/activate" ]] || die "venv missing at $VENV_DIR — run './run.sh setup' first."
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  export PYTHONNOUSERSITE=1   # ignore ~/.local site-packages that could shadow pins
  # Load .env (HF_TOKEN / WANDB_API_KEY / ...) if present, without clobbering env.
  if [[ -f "$REPO_ROOT/.env" ]]; then set -a; # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"; set +a; fi
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"   # opt in by exporting WANDB_MODE=online
}

# Create .env from the template on first setup so there is a file to fill in.
ensure_env_file() {
  if [[ ! -f "$REPO_ROOT/.env" && -f "$REPO_ROOT/.env.template" ]]; then
    cp "$REPO_ROOT/.env.template" "$REPO_ROOT/.env"
    info "created .env from .env.template — all keys are OPTIONAL; fill in only what you need."
  fi
}

# Load .env into the current shell (setup uses this; activate_venv does the same).
load_dotenv() {
  if [[ -f "$REPO_ROOT/.env" ]]; then set -a; # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"; set +a; fi
}

# Print which optional credentials are present (never prints the values themselves).
print_key_status() {
  banner "credential status (all OPTIONAL — training/eval/quantize work without them)"
  for k in HF_TOKEN WANDB_API_KEY DEEPINFRA_API_KEY OPENAI_API_KEY; do
    if [[ -n "${!k:-}" ]]; then info "  $k: set"; else info "  $k: (unset)"; fi
  done
  info "  WANDB_MODE=${WANDB_MODE:-disabled}  (set WANDB_MODE=online + WANDB_API_KEY to log curves)"
}

# ---------------------------------------------------------------------------
# 4. SETUP — the dependency-hell fix
# ---------------------------------------------------------------------------
cmd_setup() {
  banner "SETUP — build .venv and resolve the full pinned dependency stack"
  detect_gpu
  ensure_env_file

  local PYBIN; PYBIN="$(pick_python)" || die "No suitable Python (need 3.10-3.13; 3.12 preferred). Install python${PY_VER}."
  info "using interpreter: $PYBIN ($($PYBIN --version 2>&1))"

  if [[ ! -d "$VENV_DIR" ]]; then
    info "creating venv at $VENV_DIR"
    "$PYBIN" -m venv "$VENV_DIR"
  else
    info "venv already exists at $VENV_DIR (reusing; --force recreates is not auto — rm -rf .venv to rebuild)"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  export PYTHONNOUSERSITE=1
  load_dotenv   # make HF_TOKEN/WANDB_* available to the smoke run below

  python -m pip install --upgrade pip setuptools wheel

  # Prefer uv (Unsloth's recommended resolver) but fall back to plain pip.
  local USE_UV=0
  if python -m pip install -q --upgrade uv 2>/dev/null && python -m uv --version >/dev/null 2>&1; then
    USE_UV=1; info "uv available — using uv for resolution."
  else
    warn "uv unavailable — falling back to pip (still deterministic with these pins)."
  fi
  pip_install() {
    if [[ "$USE_UV" == "1" ]]; then python -m uv pip install "$@"; else python -m pip install "$@"; fi
  }
  # CUDA wheel index for the plain-pip fallback (uv uses --torch-backend=auto).
  # Without this, plain pip can resolve a CPU-only torch and CUDA silently fails.
  local PIP_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

  # ---- STEP 1: vLLM FIRST (strictest torch pinner -> brings torch 2.10.0) ----
  banner "deps 1/6 — vLLM (pins torch 2.10.0)"
  if [[ "$USE_UV" == "1" ]]; then
    pip_install "$PIN_VLLM" --torch-backend=auto || die "vLLM install failed (step 1)."
  else
    pip_install "$PIN_VLLM" --extra-index-url "$PIP_CUDA_INDEX" || die "vLLM install failed (step 1)."
  fi

  # ---- STEP 2: pin torch trio explicitly (defensive; vllm already brought it) -
  banner "deps 2/6 — torch / torchvision / torchaudio (defensive re-pin)"
  if [[ "$USE_UV" == "1" ]]; then
    pip_install "$PIN_TORCH" "$PIN_TORCHVISION" "$PIN_TORCHAUDIO" --torch-backend=auto \
      || die "torch trio pin failed (step 2)."
  else
    pip_install "$PIN_TORCH" "$PIN_TORCHVISION" "$PIN_TORCHAUDIO" --extra-index-url "$PIP_CUDA_INDEX" || die "torch trio pin failed (step 2)."
  fi

  # ---- STEP 3: Unsloth + training stack, pinned INTO the resolved window -----
  banner "deps 3/6 — Unsloth + transformers/trl/peft/accelerate/bnb/datasets"
  pip_install \
    "$PIN_UNSLOTH" "$PIN_UNSLOTH_ZOO" "$PIN_TRANSFORMERS" "$PIN_TRL" "$PIN_PEFT" \
    "$PIN_ACCELERATE" "$PIN_BNB" "$PIN_DATASETS" \
    || die "training stack install failed (step 3)."

  # ---- STEP 4: repo runtime deps (loose; must not move torch) ---------------
  banner "deps 4/6 — repo runtime deps"
  pip_install "${EXTRA_RUNTIME_DEPS[@]}" || warn "some runtime deps failed (non-fatal)."

  # ---- STEP 5: lm-eval LAST + isolated (--no-deps so it can't upgrade torch) -
  banner "deps 5/6 — lm-eval (--no-deps) + its runtime deps"
  pip_install --no-deps "$PIN_LM_EVAL" || die "lm-eval install failed (step 5)."
  pip_install "${LM_EVAL_RUNTIME_DEPS[@]}" || warn "some lm-eval runtime deps failed (non-fatal)."

  # ---- STEP 6: flash-attn — SKIP (no official torch-2.10 wheel; Unsloth uses
  #             xformers + Triton). Only attempt a community prebuilt wheel if
  #             the user explicitly opts in via FLASH_ATTN_WHEEL=<url>. ---------
  banner "deps 6/6 — flash-attn (intentionally SKIPPED)"
  if [[ -n "${FLASH_ATTN_WHEEL:-}" ]]; then
    warn "FLASH_ATTN_WHEEL set — attempting community prebuilt wheel (no build)."
    pip_install --no-build-isolation --no-deps "$FLASH_ATTN_WHEEL" \
      || warn "flash-attn wheel failed (OK — Unsloth uses xformers/Triton)."
  else
    info "flash-attn skipped (expected & fine — Unsloth uses xformers + Triton kernels)."
  fi

  # ---- install this repo as an editable package (adaptivethink.*) -----------
  banner "installing repo package (editable)"
  pip_install -e . || warn "editable install failed — falling back to PYTHONPATH=src at runtime."

  # ---- derive requirements.lock from the resolved env -----------------------
  banner "writing requirements.lock"
  write_lock

  # ---- verification: import everything + assert torch<2.11 ------------------
  banner "VERIFY — imports + versions + torch<2.11 assertion"
  verify_imports || die "Dependency verification FAILED. See guidance above."

  # ---- 1-step GRPO smoke to prove the box trains ----------------------------
  banner "SETUP smoke — 1-step GRPO (proves training works end-to-end)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    grpo_smoke 1 || die "1-step GRPO smoke FAILED — the stack imports but cannot train. See $LOG_DIR/setup_smoke.log"
    info "1-step GRPO smoke PASSED."
  else
    warn "No GPU detected — skipping GRPO smoke (it needs CUDA). Run './run.sh smoke' on the GPU box."
  fi

  print_key_status

  banner "SETUP COMPLETE"
  info "venv: $VENV_DIR"
  info "lock: $LOCK_FILE"
  info "next: ./run.sh smoke   (tiny end-to-end)   then   ./run.sh all   (full pipeline)"
}

write_lock() {
  {
    echo "# requirements.lock — generated by run.sh setup on $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# Pinned, verified-compatible stack for single-GPU GRPO on Qwen2.5-3B (CUDA 12.x)."
    echo "# Install ORDER matters: vLLM first (torch pinner), lm-eval last (--no-deps)."
    echo "# Reproduce exactly with: ./run.sh setup"
    echo "#"
    echo "# --- core pinned set (the resolved window) ---"
    echo "$PIN_VLLM"
    echo "$PIN_TORCH"
    echo "$PIN_TORCHVISION"
    echo "$PIN_TORCHAUDIO"
    echo "$PIN_UNSLOTH"
    echo "$PIN_UNSLOTH_ZOO"
    echo "$PIN_TRANSFORMERS"
    echo "$PIN_TRL"
    echo "$PIN_PEFT"
    echo "$PIN_ACCELERATE"
    echo "$PIN_BNB"
    echo "$PIN_DATASETS"
    echo "$PIN_LM_EVAL  # install with --no-deps"
    echo "#"
    echo "# --- full frozen environment (pip freeze) ---"
    python -m pip freeze
  } > "$LOCK_FILE"
  info "wrote $LOCK_FILE"
}

verify_imports() {
  python - <<'PY'
import importlib, sys
mods = ["torch","transformers","trl","peft","accelerate","bitsandbytes",
        "datasets","vllm","unsloth","unsloth_zoo","lm_eval"]
print("="*56)
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m:16s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:16s} IMPORT FAILED: {type(e).__name__}: {e}")
        failed.append(m)
import torch
print("="*56)
print("CUDA available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
maj, minv = map(int, torch.__version__.split('+')[0].split('.')[:2])
if (maj, minv) >= (2, 11):
    print(f"FATAL: torch {torch.__version__} >= 2.11 will conflict with Unsloth!")
    sys.exit(2)
try:
    import flash_attn; print("flash_attn:", flash_attn.__version__, "(present, optional)")
except Exception:
    print("flash_attn: NOT installed (expected & fine — Unsloth uses xformers/Triton)")
if failed:
    print("\nGUIDANCE: the modules above failed to import. Most common cause is a")
    print("  later package having upgraded torch. Fix: rm -rf .venv && ./run.sh setup")
    print("  (install order: vLLM first, lm-eval last with --no-deps).")
    sys.exit(1)
print("OK: full stack imports and torch<2.11.")
PY
}

# 1-/N-step GRPO smoke using Unsloth fast_inference (in-process vLLM rollout).
# import unsloth FIRST (applies patches) — else GRPO shape-mismatch at train time.
grpo_smoke() {
  local steps="${1:-1}"
  GRPO_SMOKE_STEPS="$steps" log_run "setup_smoke" python - <<'PY'
import os
import unsloth                                   # MUST be first (applies patches)
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

steps = int(os.environ.get("GRPO_SMOKE_STEPS", "1"))
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=1024, load_in_4bit=True,
    fast_inference=True,                          # in-process vLLM rollouts
    max_lora_rank=16, gpu_memory_utilization=0.6,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, use_gradient_checkpointing="unsloth",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
ds = Dataset.from_list(
    [{"prompt":[{"role":"user","content":f"Reply with the number {i}."}]} for i in range(8)])
def reward_len(completions, **kw):               # toy reward: shorter is better
    return [max(0.0, 1.0 - len(c[0]["content"])/200.0) for c in completions]
cfg = GRPOConfig(use_vllm=True, learning_rate=5e-6, per_device_train_batch_size=2,
                 gradient_accumulation_steps=1, num_generations=2,
                 max_prompt_length=256, max_completion_length=64,
                 max_steps=steps, logging_steps=1, output_dir="outputs/grpo_smoke",
                 report_to="none")
GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward_len],
            args=cfg, train_dataset=ds).train()
print(f"GRPO SMOKE ({steps} step) PASSED")
PY
}

# ---------------------------------------------------------------------------
# 5. SMOKE — tiny end-to-end (load 3B 4-bit, 2-step GRPO, 5-item eval)
# ---------------------------------------------------------------------------
cmd_smoke() {
  activate_venv
  banner "SMOKE — load 3B 4-bit + 2-step GRPO + 5-item eval"
  command -v nvidia-smi >/dev/null 2>&1 || die "smoke needs a CUDA GPU (nvidia-smi not found)."

  info "[1/2] 2-step GRPO rollout/train"
  grpo_smoke 2 || die "GRPO smoke failed — see $LOG_DIR/setup_smoke.log"

  info "[2/2] 5-item eval via existing eval/run_benchmarks.py (gsm8k, always_think)"
  log_run "smoke_eval" python eval/run_benchmarks.py \
    --model-name "$MODEL" --benchmark gsm8k --route-mode always_think \
    --n 5 --tag smoke --out "$RESULTS_DIR/smoke.json" \
    || die "smoke eval failed — see $LOG_DIR/smoke_eval.log"

  banner "SMOKE PASSED — the box is ready for a full run (./run.sh all)"
}

# ---------------------------------------------------------------------------
# 6. BASELINE — lm-eval-harness baselines (GSM8K 8-shot CoT / MMLU 5-shot / StrategyQA)
#    Separate invocations per EVAL RESEARCH to avoid cross-task --num_fewshot ambiguity.
# ---------------------------------------------------------------------------
lm_eval_common_args() {
  # Echo the flags shared by baseline and trained so the comparison is valid.
  local extra=()
  [[ "$APPLY_CHAT" == "1" ]] && extra+=(--apply_chat_template --fewshot_as_multiturn)
  printf '%s\n' --batch_size auto --log_samples "${extra[@]}"
}

cmd_baseline() {
  activate_venv
  banner "BASELINE — lm-eval-harness on base model (NO adapter): $MODEL"
  command -v lm_eval >/dev/null 2>&1 || die "lm_eval not found — run './run.sh setup'."
  local margs="pretrained=${MODEL},dtype=${EVAL_DTYPE}"
  mapfile -t COMMON < <(lm_eval_common_args)

  # GSM8K — gsm8k_cot is 8-shot CoT (baked in its YAML; do NOT pass --num_fewshot).
  if should_skip "$RESULTS_DIR/baseline_gsm8k" "baseline:gsm8k"; then :; else
    log_run "baseline_gsm8k" lm_eval --model hf --model_args "$margs" \
      --tasks gsm8k_cot "${COMMON[@]}" --output_path "$RESULTS_DIR/baseline_gsm8k" \
      || die "baseline GSM8K failed."
  fi
  # MMLU — needs explicit --num_fewshot 5 (template has no built-in shots).
  if should_skip "$RESULTS_DIR/baseline_mmlu" "baseline:mmlu"; then :; else
    log_run "baseline_mmlu" lm_eval --model hf --model_args "$margs" \
      --tasks mmlu --num_fewshot 5 "${COMMON[@]}" --output_path "$RESULTS_DIR/baseline_mmlu" \
      || die "baseline MMLU failed."
  fi
  # StrategyQA — bigbench multiple_choice (acc); zero-shot by construction.
  if should_skip "$RESULTS_DIR/baseline_sqa" "baseline:strategyqa"; then :; else
    log_run "baseline_sqa" lm_eval --model hf --model_args "$margs" \
      --tasks bigbench_strategyqa_multiple_choice "${COMMON[@]}" \
      --output_path "$RESULTS_DIR/baseline_sqa" \
      || die "baseline StrategyQA failed."
  fi
  banner "BASELINE COMPLETE — results under $RESULTS_DIR/baseline_{gsm8k,mmlu,sqa}/"
}

# ---------------------------------------------------------------------------
# 7. SFT — distillation cold-start (per the interface contract)
# ---------------------------------------------------------------------------
cmd_sft() {
  activate_venv
  banner "SFT cold-start — adaptivethink.rl.sft_coldstart"
  local out="$OUT_DIR/sft"
  if should_skip "$out/adapter_config.json" "sft"; then return 0; fi
  log_run "sft" python -m adaptivethink.rl.sft_coldstart \
    --model "$MODEL" \
    --traces "$SFT_TRACES" \
    --out "$out" \
    --steps "$SFT_STEPS" \
    --max-seq-len "$SFT_MAX_SEQ_LEN" \
    || die "SFT cold-start failed — see $LOG_DIR/sft.log"
  banner "SFT COMPLETE — adapter at $out"
}

# ---------------------------------------------------------------------------
# 8. GRPO — Dr.GRPO/GRPO training (per the interface contract), multi-seed
# ---------------------------------------------------------------------------
grpo_one_seed() {
  local seed="$1"
  local out="$OUT_DIR/grpo-seed${seed}"
  if should_skip "$out/adapter_config.json" "grpo:seed${seed}"; then return 0; fi

  local clip_flag="--clip-higher"; [[ "$CLIP_HIGHER" == "1" ]] || clip_flag="--no-clip-higher"
  local diff_flag="--difficulty-filter"; [[ "$DIFFICULTY_FILTER" == "1" ]] || diff_flag="--no-difficulty-filter"
  local oneshot_flag="--no-one-shot"; [[ "$ONE_SHOT" == "1" ]] && oneshot_flag="--one-shot"

  banner "GRPO seed=${seed} — loss=${LOSS} kl=${KL} steps=${STEPS} group=${GROUP_SIZE}"
  log_run "grpo_seed${seed}" python -m adaptivethink.rl.drgrpo_train \
    --model "$MODEL" \
    --datasets "$DATASETS" \
    --out "$out" \
    --steps "$STEPS" \
    --seed "$seed" \
    --loss "$LOSS" \
    --kl "$KL" \
    "$clip_flag" \
    --entropy "$ENTROPY" \
    "$diff_flag" \
    "$oneshot_flag" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --group-size "$GROUP_SIZE" \
    || die "GRPO seed=${seed} failed — see $LOG_DIR/grpo_seed${seed}.log"
}

cmd_grpo() {
  activate_venv
  banner "GRPO — multi-seed RL (seeds=${SEEDS})"
  # SEED env/arg overrides the multi-seed loop for a single targeted run.
  if [[ -n "${SEED:-}" ]]; then
    info "SEED=$SEED set — running a single seed."
    grpo_one_seed "$SEED"
  else
    IFS=',' read -r -a seed_arr <<< "$SEEDS"
    for s in "${seed_arr[@]}"; do
      s="$(echo "$s" | tr -d '[:space:]')"
      [[ -n "$s" ]] && grpo_one_seed "$s"
    done
  fi
  banner "GRPO COMPLETE — adapters under $OUT_DIR/grpo-seed*/"
}

# Pick the first available trained adapter (lowest seed) for eval/quantize/router.
resolve_adapter() {
  if [[ -n "${ADAPTER:-}" && -e "$ADAPTER" ]]; then echo "$ADAPTER"; return 0; fi
  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  for s in "${seed_arr[@]}"; do
    s="$(echo "$s" | tr -d '[:space:]')"
    local cand="$OUT_DIR/grpo-seed${s}"
    [[ -e "$cand/adapter_config.json" ]] && { echo "$cand"; return 0; }
  done
  # SFT-only fallback.
  [[ -e "$OUT_DIR/sft/adapter_config.json" ]] && { echo "$OUT_DIR/sft"; return 0; }
  return 1
}

# ---------------------------------------------------------------------------
# 9. EVAL — lm-eval-harness on the trained adapter (IDENTICAL config to baseline)
#    + existing eval/run_benchmarks.py for the efficiency Pareto.
# ---------------------------------------------------------------------------
cmd_eval() {
  activate_venv
  banner "EVAL — trained adapter (identical lm-eval config) + efficiency Pareto"
  command -v lm_eval >/dev/null 2>&1 || die "lm_eval not found — run './run.sh setup'."
  local adapter; adapter="$(resolve_adapter)" \
    || die "no trained adapter found (run './run.sh grpo' first, or set ADAPTER=<dir>)."
  info "trained adapter: $adapter"

  # Same flags as baseline; the ONLY change is adding peft=<adapter>.
  local margs="pretrained=${MODEL},peft=${adapter},dtype=${EVAL_DTYPE}"
  mapfile -t COMMON < <(lm_eval_common_args)

  if should_skip "$RESULTS_DIR/trained_gsm8k" "eval:gsm8k"; then :; else
    log_run "eval_gsm8k" lm_eval --model hf --model_args "$margs" \
      --tasks gsm8k_cot "${COMMON[@]}" --output_path "$RESULTS_DIR/trained_gsm8k" \
      || die "trained GSM8K eval failed."
  fi
  if should_skip "$RESULTS_DIR/trained_mmlu" "eval:mmlu"; then :; else
    log_run "eval_mmlu" lm_eval --model hf --model_args "$margs" \
      --tasks mmlu --num_fewshot 5 "${COMMON[@]}" --output_path "$RESULTS_DIR/trained_mmlu" \
      || die "trained MMLU eval failed."
  fi
  if should_skip "$RESULTS_DIR/trained_sqa" "eval:strategyqa"; then :; else
    log_run "eval_sqa" lm_eval --model hf --model_args "$margs" \
      --tasks bigbench_strategyqa_multiple_choice "${COMMON[@]}" \
      --output_path "$RESULTS_DIR/trained_sqa" \
      || die "trained StrategyQA eval failed."
  fi

  # Efficiency Pareto via the repo's own harness (accuracy vs tokens/latency).
  banner "EVAL — efficiency Pareto (eval/run_benchmarks.py)"
  local n="${PARETO_N:-200}"
  log_run "eval_pareto_baseline" python eval/run_benchmarks.py \
    --model-name "$MODEL" --benchmark all --route-mode always_think \
    --n "$n" --seeds "$SEEDS" --tag baseline --out "$RESULTS_DIR/trained/pareto_baseline.json" \
    || warn "Pareto baseline run failed (non-fatal for headline KPI)."
  log_run "eval_pareto_system" python eval/run_benchmarks.py \
    --model-name "$MODEL" --adapter "$adapter" --benchmark all --route-mode model \
    --n "$n" --seeds "$SEEDS" --tag trained --out "$RESULTS_DIR/trained/pareto_trained.json" \
    || warn "Pareto trained run failed (non-fatal for headline KPI)."

  banner "EVAL COMPLETE — headline under $RESULTS_DIR/trained_{gsm8k,mmlu,sqa}/, Pareto under $RESULTS_DIR/trained/"
}

# ---------------------------------------------------------------------------
# 10. ROUTER — optional efficiency/wow layer (reuse repo verifier + pipeline)
# ---------------------------------------------------------------------------
cmd_router() {
  activate_venv
  banner "ROUTER — efficiency/Pareto layer (verifier + adaptive routing)"
  local adapter; adapter="$(resolve_adapter)" \
    || die "no trained adapter found — run './run.sh grpo' first."
  local verifier="${VERIFIER_CKPT:-$OUT_DIR/verifier-400m/best.pt}"
  local n="${ROUTER_N:-200}"
  local route_args=()
  if [[ -e "$verifier" ]]; then
    info "verifier checkpoint: $verifier (route-mode=model)"
    route_args=(--verifier-ckpt "$verifier" --route-mode model)
  else
    warn "verifier checkpoint not found at $verifier — using threshold routing on the adapter's own difficulty signal."
    route_args=(--route-mode threshold)
  fi

  # Fixed-policy + adaptive points for the Pareto chart, then plots.
  log_run "router_model" python eval/run_benchmarks.py \
    --model-name "$MODEL" --adapter "$adapter" "${route_args[@]}" \
    --benchmark all --n "$n" --seeds "$SEEDS" \
    --tag router --out "$RESULTS_DIR/trained/router.json" \
    || die "router eval failed — see $LOG_DIR/router_model.log"
  log_run "router_system1" python eval/run_benchmarks.py \
    --model-name "$MODEL" --adapter "$adapter" --route-mode never_think \
    --benchmark all --n "$n" --seeds "$SEEDS" \
    --tag system1 --out "$RESULTS_DIR/trained/system1.json" || warn "system1 point failed (non-fatal)."
  log_run "router_system2" python eval/run_benchmarks.py \
    --model-name "$MODEL" --adapter "$adapter" --route-mode always_think \
    --benchmark all --n "$n" --seeds "$SEEDS" \
    --tag system2 --out "$RESULTS_DIR/trained/system2.json" || warn "system2 point failed (non-fatal)."

  if [[ -f "$REPO_ROOT/eval/plots.py" ]]; then
    log_run "router_plots" python eval/plots.py \
      --baseline "$RESULTS_DIR/trained/pareto_baseline.json" \
      --runs "$RESULTS_DIR/trained/router.json" "$RESULTS_DIR/trained/system1.json" "$RESULTS_DIR/trained/system2.json" \
      --outdir "$RESULTS_DIR/figures" || warn "plot generation failed (non-fatal)."
  fi
  banner "ROUTER COMPLETE — Pareto points + figures under $RESULTS_DIR/{trained,figures}/"
}

# ---------------------------------------------------------------------------
# 11. QUANTIZE — merge adapter -> GGUF Q4_K_M (reuse existing script/exporter)
# ---------------------------------------------------------------------------
cmd_quantize() {
  activate_venv
  banner "QUANTIZE — merge trained adapter -> GGUF Q4_K_M (on-device)"
  local adapter; adapter="$(resolve_adapter)" \
    || die "no trained adapter found — run './run.sh grpo' first."
  local gguf_out="$OUT_DIR/gguf/router-Q4_K_M.gguf"
  if should_skip "$gguf_out" "quantize"; then return 0; fi
  mkdir -p "$OUT_DIR/gguf"

  if [[ -f "$REPO_ROOT/scripts/06_quantize.sh" ]]; then
    info "using scripts/06_quantize.sh"
    log_run "quantize" bash "$REPO_ROOT/scripts/06_quantize.sh" "$adapter" \
      || die "quantize failed — see $LOG_DIR/quantize.log"
  elif [[ -f "$REPO_ROOT/src/adaptivethink/quantize/export_gguf.py" ]]; then
    info "using src/adaptivethink/quantize/export_gguf.py"
    log_run "quantize" python src/adaptivethink/quantize/export_gguf.py \
      --adapter "$adapter" --merged-dir "$OUT_DIR/router-merged" \
      --out "$gguf_out" --quant-type Q4_K_M \
      || die "quantize failed — see $LOG_DIR/quantize.log"
  else
    die "no quantize entry point found (scripts/06_quantize.sh or export_gguf.py)."
  fi
  banner "QUANTIZE COMPLETE — GGUF under $OUT_DIR/gguf/"
}

# ---------------------------------------------------------------------------
# 12. ALL — full resumable pipeline with banner-delimited stages
# ---------------------------------------------------------------------------
cmd_all() {
  activate_venv
  banner "ALL — full pipeline: baseline -> (sft) -> grpo -> eval -> router -> quantize"
  info "model=$MODEL datasets=$DATASETS seeds=$SEEDS loss=$LOSS kl=$KL steps=$STEPS sft=${RUN_SFT}"

  cmd_baseline
  if [[ "$RUN_SFT" == "1" ]]; then cmd_sft; else info "SKIP sft (pass --sft or RUN_SFT=1 to enable cold-start)"; fi
  cmd_grpo
  cmd_eval
  cmd_router
  cmd_quantize

  banner "ALL STAGES COMPLETE"
  info "headline KPIs: $RESULTS_DIR/{baseline,trained}_{gsm8k,mmlu,sqa}/"
  info "efficiency Pareto + figures: $RESULTS_DIR/{trained,figures}/"
  info "on-device GGUF: $OUT_DIR/gguf/"
  info "logs: $LOG_DIR/"
}

# ---------------------------------------------------------------------------
# 13. Usage / help
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
AdaptiveThink-RL — master script (single Linux GPU, CUDA 12.x, ~50GB VRAM)

USAGE
  ./run.sh <subcommand> [flags]

SUBCOMMANDS
  setup       Build .venv + resolve the FULL pinned dependency stack (the
              dependency-hell fix), write requirements.lock, verify imports,
              and run a 1-step GRPO smoke. Run this FIRST.
  smoke       Tiny end-to-end: load 3B 4-bit, 2-step GRPO, 5-item eval.
  baseline    lm-eval-harness baselines: GSM8K(8-shot CoT)/MMLU(5-shot)/StrategyQA.
  sft         Distillation cold-start (adaptivethink.rl.sft_coldstart).
  grpo        Dr.GRPO/GRPO training (adaptivethink.rl.drgrpo_train), multi-seed.
  eval        lm-eval on the trained adapter (identical config) + efficiency Pareto.
  router      Optional efficiency/wow layer (verifier + adaptive routing + plots).
  quantize    Merge adapter -> GGUF Q4_K_M (on-device).
  all         baseline -> (sft) -> grpo -> eval -> router -> quantize (resumable).
  help        This message.

COMMON FLAGS (also settable as env vars)
  --model <id>         default: $MODEL  (fallback Qwen/Qwen2.5-1.5B-Instruct)
  --datasets <csv>     default: $DATASETS
  --seeds <csv>        default: $SEEDS        (multi-seed RL)
  --seed <int>         single-seed GRPO run (overrides --seeds)
  --steps <int>        GRPO steps (default: $STEPS)
  --loss <grpo|dr_grpo>default: $LOSS
  --kl <float>         KL coefficient (default: $KL, off)
  --entropy <float>    entropy bonus (default: $ENTROPY)
  --group-size <int>   GRPO group size (default: $GROUP_SIZE)
  --max-seq-len <int>  default: $MAX_SEQ_LEN
  --clip-higher / --no-clip-higher           (default: clip-higher on)
  --difficulty-filter / --no-difficulty-filter (default: on)
  --one-shot / --no-one-shot                 1-shot RLVR ablation (default: off)
  --sft                run SFT cold-start in 'all' (default: off)
  --adapter <dir>      explicit trained adapter for eval/router/quantize
  --force              rerun a stage even if its output exists (or FORCE=1)

ENV VARS
  VENV_DIR, PY_VER(=$PY_VER), HF_TOKEN, WANDB_MODE(=${WANDB_MODE:-disabled}),
  FLASH_ATTN_WHEEL (opt-in community wheel URL), PARETO_N, ROUTER_N,
  VERIFIER_CKPT, EVAL_DTYPE(=$EVAL_DTYPE), APPLY_CHAT(=$APPLY_CHAT)

EXAMPLES
  ./run.sh setup
  ./run.sh smoke
  ./run.sh all --sft
  ./run.sh grpo --model Qwen/Qwen2.5-1.5B-Instruct --seeds 0,1,2 --loss dr_grpo --kl 0.0
  SEED=0 ./run.sh grpo
  ./run.sh eval --adapter outputs/grpo-seed0
EOF
}

# ---------------------------------------------------------------------------
# 14. Flag parsing (shared across subcommands)
# ---------------------------------------------------------------------------
parse_flags() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)             MODEL="$2"; shift 2;;
      --datasets)          DATASETS="$2"; shift 2;;
      --seeds)             SEEDS="$2"; shift 2;;
      --seed)              SEED="$2"; shift 2;;
      --steps)             STEPS="$2"; shift 2;;
      --sft-steps)         SFT_STEPS="$2"; shift 2;;
      --loss)              LOSS="$2"; shift 2;;
      --kl)                KL="$2"; shift 2;;
      --entropy)           ENTROPY="$2"; shift 2;;
      --group-size)        GROUP_SIZE="$2"; shift 2;;
      --max-seq-len)       MAX_SEQ_LEN="$2"; shift 2;;
      --traces)            SFT_TRACES="$2"; shift 2;;
      --adapter)           ADAPTER="$2"; shift 2;;
      --clip-higher)       CLIP_HIGHER=1; shift;;
      --no-clip-higher)    CLIP_HIGHER=0; shift;;
      --difficulty-filter) DIFFICULTY_FILTER=1; shift;;
      --no-difficulty-filter) DIFFICULTY_FILTER=0; shift;;
      --one-shot)          ONE_SHOT=1; shift;;
      --no-one-shot)       ONE_SHOT=0; shift;;
      --sft)               RUN_SFT=1; shift;;
      --force)             FORCE=1; shift;;
      --) shift; break;;
      -*) die "unknown flag: $1 (see './run.sh help')";;
      *)  die "unexpected argument: $1 (see './run.sh help')";;
    esac
  done
}

# ---------------------------------------------------------------------------
# 15. Dispatch
# ---------------------------------------------------------------------------
main() {
  local sub="${1:-help}"; shift || true
  parse_flags "$@"
  case "$sub" in
    setup)    cmd_setup;;
    smoke)    cmd_smoke;;
    baseline) cmd_baseline;;
    sft)      cmd_sft;;
    grpo)     cmd_grpo;;
    eval)     cmd_eval;;
    router)   cmd_router;;
    quantize) cmd_quantize;;
    all)      cmd_all;;
    help|-h|--help) usage;;
    *) warn "unknown subcommand: $sub"; usage; exit 1;;
  esac
}

main "$@"
