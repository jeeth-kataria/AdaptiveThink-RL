# RUNBOOK — Strategy 1 staged plan (copy-paste commands)

The exact commands to run every strategy from `new.md`, in the staged **Day 1–14**
order. This is the "task programs to run" deliverable: each block is copy-paste
ready for a **single Linux GPU, CUDA 12.x, ~50 GB VRAM**.

**Goal framing (do not lose this):**
- **IMPROVE** → **GSM8K** + **StrategyQA** (rule-based exact-match reward; the +5% bar lives here).
- **MAINTAIN** → **MMLU** (knowledge MC; RL on math reasoning barely moves it — just keep it above the floor).
- **Always** run the *identical* `lm-eval` harness config for baseline AND trained model, and report
  **multiple seeds (0, 1, 2) with confidence intervals** — RL is high-variance and small eval sets swing on one question.

Two equivalent ways to invoke everything:

1. **Master script (preferred):** `./run.sh <subcommand>` — subcommands
   `setup | smoke | baseline | sft | grpo | eval | router | quantize | all`.
   `run.sh all` runs `baseline → (sft) → grpo → eval → router → quantize`.
2. **Direct module calls** (shown beneath each `run.sh` line) — use these if you need to
   override a specific flag or `run.sh` is unavailable.

All hyperparameters live in [`configs/strategy1.yaml`](configs/strategy1.yaml).
Logs are written to `logs/`; every stage is idempotent and resumable (re-running
resumes from the latest `checkpoint-*`).

---

## Day 0 — Setup (resolve ALL dependencies once)

The #1 pain is dependency resolution. `run.sh setup` builds a venv at `.venv` and
installs the verified-compatible pinned stack (vLLM first → torch 2.10.0, then
Unsloth + TRL 0.24, transformers 4.57.6; lm-eval last with `--no-deps`; flash-attn
skipped — Unsloth uses xformers/Triton).

```bash
./run.sh setup
# then activate for any manual module calls below:
source .venv/bin/activate
```

<details><summary>What setup pins (for reference / manual install)</summary>

```bash
# vLLM FIRST (strictest torch pinner) -> torch 2.10.0
uv pip install "vllm==0.19.1" --torch-backend=auto
uv pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" --torch-backend=auto
# training stack, pinned into the resolved window (trl<=0.24, NOT 1.x; transformers 4.57.6)
uv pip install "unsloth==2026.6.7" "unsloth_zoo>=2026.6.5" "transformers==4.57.6" \
  "trl==0.24.0" "peft==0.19.1" "accelerate==1.14.0" "bitsandbytes==0.49.2" "datasets==3.6.0"
# lm-eval LAST, isolated so its loose floors can't upgrade torch/transformers/vllm
uv pip install --no-deps "lm-eval==0.4.12"
# flash-attn: SKIP (no official wheel for torch 2.10; Unsloth doesn't need it)
```
GRPO runtime gotcha: **`import unsloth` must precede `trl`/`transformers`** — both
trainers already do this internally.
</details>

**Smoke test the whole wiring (2 GRPO steps, ~5–7 GB VRAM):**

```bash
./run.sh smoke
# == one-item-per-dataset, 2 steps, no vLLM, fast:
python -m adaptivethink.rl.drgrpo_train --model Qwen/Qwen2.5-1.5B-Instruct \
  --datasets gsm8k,strategyqa --out outputs/smoke --steps 2 --seed 0 \
  --loss dr_grpo --kl 0.0 --no-clip-higher --entropy 1.0 \
  --no-difficulty-filter --one-shot --max-seq-len 1024 --group-size 4
```

---

## Day 1–2 — Baseline (lock the base model + fixed eval harness)

Establish baselines on GSM8K, MMLU, StrategyQA with a **fixed** `lm-eval` config —
the same config you will reuse after training. **Threshold:** if baseline
GSM8K > 80%, switch the base to `Qwen2.5-1.5B-Instruct` for more headroom (edit
`model:` in `configs/strategy1.yaml` or pass `--model`).

```bash
./run.sh baseline
# == lm-eval, identical config used for baseline AND trained model:
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=bfloat16 \
  --tasks gsm8k,mmlu,strategyqa --num_fewshot 5 --batch_size auto \
  --output_path results/baseline --log_samples 2>&1 | tee logs/baseline.log
```

> Keep `--num_fewshot` and the prompt template FIXED across baseline and trained
> eval, or the before/after delta is confounded (published GSM8K varies 4-shot vs
> 8-shot-CoT). The `<think>/<answer>` system prompt used in training is the same
> one in `configs/strategy1.yaml`.

---

## Day 3–5 — Strategy 1 core: SFT cold-start → Dr.GRPO RLVR

### Step 3a — SFT cold-start (install the `<think>/<answer>` format)

Short SFT on ~1k DeepSeek-R1 CoT traces (`simplescaling/s1K-1.1`) so the format is
reliable before RL. (Optional but recommended; set `sft.enabled: false` to skip.)

```bash
./run.sh sft
# == direct:
python -m adaptivethink.rl.sft_coldstart \
  --model Qwen/Qwen2.5-3B-Instruct \
  --traces simplescaling/s1K-1.1 \
  --out outputs/sft --steps 500 --max-seq-len 4096
```

Alternative right-sized trace sources (swap `--traces`, add `--config default`,
cap with `--max-examples`):
```bash
# math-diverse R1 traces, subsampled to a short cold-start:
python -m adaptivethink.rl.sft_coldstart --model Qwen/Qwen2.5-3B-Instruct \
  --traces open-r1/OpenR1-Math-220k --config default --max-examples 4000 \
  --out outputs/sft --steps 500 --max-seq-len 4096
# ShareGPT-chat R1 traces:
python -m adaptivethink.rl.sft_coldstart --model Qwen/Qwen2.5-3B-Instruct \
  --traces open-thoughts/OpenThoughts-114k --config default --max-examples 4000 \
  --out outputs/sft --steps 500 --max-seq-len 4096
# Optional: drop traces whose <answer> != gold (reuses router.reward matcher):
#   add  --validate-answers
```

### Step 3b — Dr.GRPO RL (the accuracy core), multi-seed

Dr.GRPO loss (constant normalization) + binary correctness + small format reward,
**KL off**, clip-higher off, entropy mask off — the small-model-safe config
(Strategy 6 toolkit). Run **all three seeds** for confidence intervals.

```bash
./run.sh grpo            # runs seeds 0,1,2 from configs/strategy1.yaml
# == direct, one seed (loop S in 0 1 2):
for S in 0 1 2; do
  python -m adaptivethink.rl.drgrpo_train \
    --model Qwen/Qwen2.5-3B-Instruct --datasets gsm8k,strategyqa \
    --out outputs/grpo-seed${S} --steps 1500 --seed ${S} \
    --loss dr_grpo --kl 0.0 --no-clip-higher --entropy 1.0 \
    --no-difficulty-filter --no-one-shot --max-seq-len 2048 --group-size 8 \
    2>&1 | tee logs/grpo-seed${S}.log
done
```

> The trainer starts from `--model` (the base). To RL **from the SFT cold-start
> adapter**, point `--model` at the merged SFT output (or pass the adapter dir if
> your loader supports it): `--model outputs/sft`.

---

## Day 6–8 — Strategy 6: stability toolkit + StrategyQA + multi-seed evals

Tune stability and confirm gains hold across seeds. The stability knobs are all
on the `drgrpo_train` arg surface.

**Watch `frac_reward_zero_std`** (logged by the trainer) — high values mean many
groups have all-equal rewards (no gradient). If high, enable the **difficulty
filter** (offline: drop base-unsolvable `p_solve==0` and trivial `p_solve==1`):

```bash
python -m adaptivethink.rl.drgrpo_train \
  --model Qwen/Qwen2.5-3B-Instruct --datasets gsm8k,strategyqa \
  --out outputs/grpo-filtered-seed0 --steps 1500 --seed 0 \
  --loss dr_grpo --kl 0.0 --no-clip-higher --entropy 1.0 \
  --difficulty-filter --no-one-shot --max-seq-len 2048 --group-size 8 \
  2>&1 | tee logs/grpo-filtered-seed0.log
```

**If entropy collapses or reward gets hacked**, A/B the stability levers (one change at a time):

```bash
# (a) small KL anchor instead of none (DeepSeek-R1 used a KL term):
python -m adaptivethink.rl.drgrpo_train ... --kl 1.0e-3 ...
# (b) DAPO Clip-Higher (epsilon_high=0.28) — note: inert at on-policy mu=1, and the
#     SLM ablation says it HURTS tiny models; use only as an experiment:
python -m adaptivethink.rl.drgrpo_train ... --clip-higher ...
# (c) plain GRPO loss instead of Dr.GRPO (bias-fix ablation):
python -m adaptivethink.rl.drgrpo_train ... --loss grpo ...
```

**Multi-seed eval of the trained model** (identical harness as Day 1–2 baseline):

```bash
./run.sh eval
# == per-seed lm-eval, then aggregate with CIs:
for S in 0 1 2; do
  lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=bfloat16,enable_lora=True,lora_local_path=outputs/grpo-seed${S} \
    --tasks gsm8k,mmlu,strategyqa --num_fewshot 5 --batch_size auto \
    --output_path results/grpo-seed${S} --log_samples 2>&1 | tee logs/eval-seed${S}.log
done
```

> **Report:** "+X% GSM8K, +Y% StrategyQA (mean ± 95% CI over seeds 0/1/2),
> MMLU maintained above floor." Plot reward & entropy curves to prove stability.

---

## Day 9–11 — Strategy 2: router / efficiency layer + on-device (the "wow")

The adaptive-compute router (route easy → short/no-think, hard → full CoT) is the
**efficiency + deployment** story bolted on top of the Strategy-1 accuracy core.
It uses the existing repo router pipeline (verifier-gated `<think>/<no_think>` GRPO).

```bash
./run.sh router          # verifier + GRPO router (existing pipeline)
# == existing router stages (see README pipeline):
bash scripts/03_train_verifier.sh                 # 400M difficulty verifier
bash scripts/04_train_grpo_router.sh 0            # GRPO router, seed 0 (repeat 1,2)
# Efficiency Pareto (accuracy vs tokens/latency) via the existing harness:
python eval/run_benchmarks.py --model-name Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/router-seed0 --route-mode model \
  --benchmark all --seeds 0,1,2 --tag router --out results/router_eval.json
```

**Quantize to GGUF Q4_K_M for the Samsung Galaxy on-device latency numbers:**

```bash
./run.sh quantize
# == direct:
python -m adaptivethink.quantize.export_gguf \
  --base-model Qwen/Qwen2.5-3B-Instruct --adapter outputs/grpo-seed0 \
  --quant-type Q4_K_M --out outputs/gguf/adaptivethink-Q4_K_M.gguf
```

> **Threshold (Day 12–14):** if the router degrades accuracy below the +5% bar,
> ship it as an optional inference mode and keep the Strategy-1 accuracy model as
> the headline.

---

## Day 12–14 — Ablations + presentation

### Strategy 3 — one-shot / minimal-data RLVR flex

Show that ~1 carefully chosen verifiable example nearly matches large-data RL — a
cheap, reproducible "insight" ablation. Uses the trainer's `--one-shot` flag
(1 item per dataset).

```bash
python -m adaptivethink.rl.drgrpo_train \
  --model Qwen/Qwen2.5-3B-Instruct --datasets gsm8k \
  --out outputs/grpo-oneshot-seed0 --steps 1500 --seed 0 \
  --loss dr_grpo --kl 0.0 --no-clip-higher --entropy 1.0 \
  --no-difficulty-filter --one-shot --max-seq-len 2048 --group-size 8 \
  2>&1 | tee logs/grpo-oneshot-seed0.log
# evaluate the one-shot run with the SAME harness, compare to full-data RL:
lm_eval --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=bfloat16,enable_lora=True,lora_local_path=outputs/grpo-oneshot-seed0 \
  --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path results/grpo-oneshot
```

### Strategy 4 — TTRL (test-time RL) ablation — clearly labeled

Majority-vote pseudo-labels on unlabeled eval data. **Optics caution:** disclose
methodology; never the headline. Uses the existing TTRL module.

```bash
python -m adaptivethink.ttrl.ttrl \
  --adapter outputs/grpo-seed0 --output-dir outputs/ttrl-seed0 \
  --n 500 --steps 300 --group-size 8 --max-seq-len 2048 \
  --lr 1.0e-5 --kl-beta 1.0e-3 --seed 0 2>&1 | tee logs/ttrl-seed0.log
# (config: configs/ttrl_ablation.yaml)
```

### Strategy 5 — advanced variants (GSPO / full DAPO) — NOTE, do not run by default

Per the ablation, GSPO's benefit is MoE/large-model stability and full DAPO's
extra tricks (Clip-Higher, token-level loss, dynamic sampling, soft overlong
punishment) **hurt** sub-3B dense models. We deliberately **cherry-pick only**
Dr.GRPO's bias fixes + overlong filtering and **skip the rest**. If you want to
demonstrate the ablation that these hurt, reuse the Day 6–8 A/B levers
(`--clip-higher`, `--loss grpo`) rather than a separate trainer. (Dynamic
sampling is not supported in TRL; difficulty filtering is the offline equivalent.)

### Final — whole pipeline + KPI/Pareto charts

```bash
./run.sh all             # baseline -> (sft) -> grpo -> eval -> router -> quantize
# KPI delta table + Pareto charts (existing harness):
python eval/run_benchmarks.py --benchmark all --seeds 0,1,2 --tag final --out results/final.json
```

**Deck checklist (this wins hackathons):** headline metric front and center
("+X% GSM8K, +Y% StrategyQA, MMLU maintained, ~50% fewer tokens"); an ablation
table isolating each reward/stability component; a Pareto chart (accuracy vs
tokens/latency); **multi-seed runs with 95% confidence intervals**; reward &
entropy curves; one clean qualitative `<think>/<answer>` example; on-device
latency from the quantized GGUF.

---

## Strategy → command quick index

| # | Strategy (new.md)              | Command                                                                 |
|---|--------------------------------|-------------------------------------------------------------------------|
| 1 | Dr.GRPO core (cold-start + RL) | `./run.sh sft` → `./run.sh grpo` (Day 3–5)                               |
| 2 | Router / efficiency + on-device| `./run.sh router` → `./run.sh quantize` (Day 9–11)                       |
| 3 | One-shot RLVR flex             | `drgrpo_train ... --one-shot` (Day 12–14)                                |
| 4 | TTRL ablation                  | `python -m adaptivethink.ttrl.ttrl ...` (Day 12–14)                      |
| 5 | GSPO / DAPO variants (NOTE)    | Not run by default; A/B via `--clip-higher` / `--loss grpo`             |
| 6 | Stability toolkit (cross-cut)  | `--difficulty-filter`, `--kl`, `--clip-higher`, `--entropy`, `--loss`    |
|   | Baseline / eval (fixed harness)| `./run.sh baseline` / `./run.sh eval` (multi-seed + CI)                  |
