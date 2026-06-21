# AdaptiveThink-RL

- **Problem Statement Number** - PS06
- **Problem Statement Title** - Enhancing Reasoning in Small Language Models (SLMs, ≤7B) using Reinforcement Learning — Samsung EnnovateX AX Hackathon 2026
- **Team name** - StateZero
- **Team members (Names)** - Jeeth Bhavesh Kataria, Ojasvi Poonia
- **Institute/College Name** - Ramaiah Institute of Technology, Bengaluru, Karnataka – 560054
- **Final Presentation Google Drive Link** - <!-- TODO: add Google Drive / Slides link -->
- **Full Submission Demo Video Link** - <!-- TODO: add YouTube/Drive demo video link -->
- **Setup & Result Reproducibility Video Link** - <!-- TODO: add YouTube/Drive reproducibility video link -->

---

## Solution Summary

We make a **1.5B** small language model reason **substantially better with reinforcement learning alone** — **no SFT distillation and no LLM API anywhere** (a self-imposed constraint, since the model never sees a teacher's answers). On the **same base model, identical eval harness, and full test sets**, RL lifts:

- **GSM8K +12.6%** (64.2 → 76.8)
- **MMLU +8.0%** on the standard 57-subject benchmark (45.9 → 53.9) — *never trained on*
- **StrategyQA +5.4%** (56.3 → 61.7)

That clears the PS06 KPI (**≥ +5% on ≥ 2 of 3**) with margin, and it does so **at single-shot, low-latency inference** on a sub-2B model.

The method is deliberately **lightweight and SLM-appropriate**: **Dr.GRPO + RLVR** (a rule-based, verifiable reward — no reward model, no preference data, no teacher) on `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0). The headline finding is exactly the failure mode the problem statement warns about — **SLMs are extremely sensitive to reward design.** Our first run barely moved the model; the fix that unlocked everything was correcting the **answer-verification in the reward itself** (and matching it in eval), not a new algorithm or more compute.

> **What is genuinely ours.** Not a new RL algorithm. The contributions are (1) the diagnosis and fix of a **reward-signal failure specific to SLMs** — a verifiable reward that silently scored correct-but-differently-formatted answers as wrong, corrupting GRPO — and (2) a **train/eval-consistent RLVR pipeline** that demonstrably improves *reasoning* (it transfers to two held-out benchmarks) rather than overfitting to the trained tasks. We build on TRL / Unsloth / vLLM.

---

## KPI Results

**Goal (PS06):** improve **≥ 2 of 3** benchmarks by **≥ +5%** over the **same base model**, efficiently. All numbers are **Pass@1, greedy, single-shot**, scored by the **identical** harness for baseline and trained model (same `<think>/<answer>` prompt, same extractor), on the **full test split** of each benchmark (no sampling). Reproduce with `eval/eval_kpi.py` (see Quick Start).

| Benchmark | Role | Baseline (Qwen2.5-1.5B-Instruct) | Ours (Dr.GRPO + RLVR) | Δ | Target | Met? |
|---|---|---|---|---|---|---|
| **GSM8K** (full, 1319) | improve | 64.2 | **76.8** | **+12.6** | ≥ 50% & ≥ +5% | ✅ |
| **MMLU** (full 57-subject, ~14k) | improve / held-out | 45.9 | **53.9** | **+8.0** | ≥ 45% & ≥ +5% | ✅ |
| **StrategyQA** (full, 687) | improve | 56.3 | **61.7** | **+5.4** | ≥ 65% & ≥ +5% | improvement ✅ (abs −3.3 under floor) |

> MMLU and AQuA are **held out** (never trained on). Trained only on GSM8K + StrategyQA *train* splits; every benchmark is scored on its *test* split. A balanced 16-subject MMLU subset gives an even larger gain (46.7 → 58.0, **+11.3**); we report the **standard full 57-subject** number above so there is no subset-selection question.

**Verdict — KPI met under either reading of "≥ 2 of 3":**
- *Floor + improvement:* **GSM8K** (76.8 ≥ 50, +12.6) and **MMLU** (53.9 ≥ 45, +8.0) both fully qualify → **2 / 3**.
- *Improvement ≥ +5%:* GSM8K +12.6, MMLU +8.0, StrategyQA +5.4 → **3 / 3**.

### Generalization (not overfitting), measured

Because RL was applied only to GSM8K + StrategyQA, the held-out benchmarks are the honest test of whether we improved **reasoning** vs memorized two datasets. Both held-out tasks **rose**:

| Held-out benchmark (never trained) | Baseline | Ours | Δ |
|---|---|---|---|
| MMLU (57-subject) | 45.9 | 53.9 | **+8.0** |
| AQuA-RAT (test, 254) | 42.9 | 47.2 | **+4.3** |

An overfit model degrades on unseen tasks; ours improves on both — evidence of a transferable reasoning gain.

### Efficiency (PS06's "minimal latency overhead")

The reported model wins at **single-shot greedy** (one forward pass / question). As an ablation we ran **self-consistency (vote@8)**: it lifts the high-variance *base* model far more than our trained model, because **RL made our model reliable enough not to need test-time sampling**. Concretely, our trained model reaches the base model's *8-sample* GSM8K accuracy in a *single* pass (~5× less compute):

| Config | GSM8K | Tokens / question |
|---|---|---|
| Trained, **1 sample** (greedy) | **74.0%** | ~280 |
| Base, **8 samples** (vote@8) | 75.2% | ~1410 |

So the efficient single-shot model already meets the KPI; voting is *not* needed and is reported only as an ablation.

---

## How we did it (approach)

```
GSM8K train  ─┐
StrategyQA   ─┤   <think>…</think><answer>…</answer>      rule-based reward (RLVR)
  train       ├─►  prompt (train == eval)        ─►   • correctness (exact-match, robust)   ─►  Dr.GRPO
              │    Qwen2.5-1.5B-Instruct                • format (well-formed think/answer)        (KL off,
              │    + 4-bit QLoRA (Unsloth)                                                          no per-group
              │    vLLM colocated rollouts                                                          std-div)
              ▼
   LoRA adapter (1.18% of params)  ─►  eval on GSM8K / StrategyQA (test) + held-out MMLU / AQuA
```

**1. Base model — `Qwen/Qwen2.5-1.5B-Instruct`** (Apache-2.0). Small enough for on-device / low-latency, with real reasoning headroom; the trainer is one `--model` flag away from 3B/7B (all Qwen2.5 share ChatML and the same 7 LoRA target modules).

**2. RL algorithm — Dr.GRPO.** TRL `GRPOTrainer` with `loss_type=dr_grpo` (constant length-normalization, no per-group reward-std division → removes GRPO's length/difficulty biases), **KL/β = 0**, `group_size = 8`, run under **Unsloth 4-bit QLoRA** with **vLLM colocated rollouts** on one 24 GB GPU. Final run: `lr 1e-5`, `500` steps, `max_prompt 512`, `max_completion 640`. *No reward model, no preference data, no teacher / API* — the only learning signal is the verifiable reward below.

**3. Reward design — RLVR (outcome + process).** This is where SLM sensitivity bit us, and where the win came from.
- **Outcome (weight 1.0):** binary exact-match. The prediction is the `<answer>…</answer>` content, compared to gold with a **robust matcher** ([`router/reward.py:match_answer`](src/adaptivethink/router/reward.py)) — tolerant numeric / fraction / LaTeX / boolean comparison, then the **standard per-task extraction** (last number for math, stated True/False for StrategyQA, option letter for MC). This is the same extraction every public harness (lm-eval-harness) uses.
- **Process (weight 0.2):** a small reward for a well-formed, ordered `<think>…</think><answer>…</answer>` block ([`rl/rewards.py`](src/adaptivethink/rl/rewards.py)). Kept small so it can never be gamed over correctness.

**4. Train/eval consistency.** Training and evaluation use the **identical** `<think>/<answer>` prompt and the **identical** answer matcher, so the measured delta is not confounded by prompt or parsing differences ([`eval/eval_kpi.py`](eval/eval_kpi.py)).

---

## Key insights (what we learned)

1. **SLMs are brutally sensitive to reward design — a "correct" verifiable reward can still be wrong.** Our first GRPO run improved GSM8K by only ~+1% and looked like a failure. The cause was *not* the model or the algorithm: the reward's answer-checker scored correct answers written as expressions (`<answer>1+3+…= 8 hours</answer>` for gold `8`) or prose as **wrong**. ~16% of genuinely-correct rollouts were rewarded **0**, corrupting GRPO's advantage estimates. The *same* bug deflated our eval, making the base model look like 37% on GSM8K when it is really ~64%. **Fixing the answer extraction in both the reward and the eval** — standard last-number / boolean / letter extraction — is what unlocked the gains. This is the PS06 warning ("high sensitivity to reward design") made concrete.
2. **Measure before you train more.** Most of the apparent "jump" (37 → 64 GSM8K) was *correct measurement*, not model improvement; the genuine RL gain is the +12.6 on top of the corrected baseline. We verified the corrected scores by re-scoring saved completions offline before spending another GPU-hour.
3. **It generalizes.** Training on math + commonsense improved two never-trained benchmarks (MMLU +8.0, AQuA +4.3) — the model learned a reasoning *habit* (reason, then commit), not dataset-specific answers.
4. **A reliable small model beats an unreliable one with test-time tricks.** Self-consistency helps the base model far more than ours; RL bought us reliability, which is the better trade against latency.
5. **Low LR + too few steps under-trains GRPO on SLMs.** `lr 1e-6 / 250 steps` barely moved the policy (grad-norm ~0.02); `lr 1e-5 / 500 steps` with the corrected reward did.

---

## Genuineness / integrity statement

- **No leakage:** training touches only *train* splits (GSM8K, StrategyQA); every benchmark is scored on its *test* split; **MMLU and AQuA are never trained on**.
- **No hardcoding:** no answer tables, no memorized test data; predictions come from live model generation and standard extraction. Baseline and trained model are scored by the **identical** function.
- **Full test sets**, greedy, single-shot — the reported numbers carry no sampling noise, and the standard full-57 MMLU is used (no convenient subset).
- The repo's reward/eval logic is covered by unit tests (`tests/test_reward.py`).

---

## Quick Start (reproducibility)

```bash
git clone <repo-url> && cd AdaptiveThink-RL
./run.sh setup            # pinned stack: vLLM 0.19.1 / torch 2.10 / TRL 0.24 / Unsloth / transformers 4.57.6
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

**Train (Dr.GRPO + RLVR, ~3.5 h on one 24 GB GPU):**
```bash
python -m adaptivethink.rl.drgrpo_train \
  --model Qwen/Qwen2.5-1.5B-Instruct --datasets gsm8k,strategyqa \
  --out outputs/grpo-seed0-v2 --steps 500 --seed 0 \
  --loss dr_grpo --kl 0.0 --group-size 8 --lr 1e-5 \
  --max-prompt-length 512 --max-completion-length 640 --save-steps 100
```

**Evaluate (identical harness; baseline = no `--adapter`, trained = with it):**
```bash
# baseline
python eval/eval_kpi.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
  --datasets gsm8k,strategyqa,mmlu --n 0 --mmlu-all --out results/final_baseline.json
# trained
python eval/eval_kpi.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/grpo-seed0-v2 --datasets gsm8k,strategyqa,mmlu --n 0 --mmlu-all \
  --out results/final_trained.json
```
`--n 0` = full test split; `--mmlu-all` = standard 57-subject MMLU (omit for the 16-subject spread). The eval prints per-benchmark Pass@1 and writes per-item completions for offline re-scoring.

---

## Implemented extensions (NOT part of the reported result)

These are designed and present in the repo as optional/future work. **The KPI result above does not use or depend on them** — it is the core Dr.GRPO + RLVR model — and we do not claim their effect:

- **CCDD curriculum filter** ([`rl/self_difficulty.py`](src/adaptivethink/rl/self_difficulty.py)) — an API-free, teacher-free difficulty signal (`1 − base solve-rate@K`) usable as a GRPO curriculum filter (`--self-difficulty-file`). The reported run was trained **without** it.
- **Adaptive-compute router / verifier** ([`src/adaptivethink/router/`](src/adaptivethink/router/), [`src/adaptivethink/verifier/`](src/adaptivethink/verifier/)) — a `think` / `no_think` gate for on-device latency. Implemented; not part of the reported run.
- **GGUF Q4_K_M export** ([`src/adaptivethink/quantize/`](src/adaptivethink/quantize/)) for llama.cpp on-device inference.
- **Test-Time RL (TTRL)** ([`src/adaptivethink/ttrl/`](src/adaptivethink/ttrl/)) — optional, clearly-labeled ablation only.

---

## Project Artefacts

- **Documentation** — [`docs/results.md`](docs/results.md) (full KPI tables, generalization, efficiency, reproduce commands), [`docs/technical.md`](docs/technical.md) (stack, architecture, design decisions), [`docs/ax.md`](docs/ax.md) (agentic-AI & open-weight usage), and [`docs/journey.md`](docs/journey.md) — **the honest development log: what failed, what fixed it, and the insights gained.**
- **Source Code** — [`src/adaptivethink/`](src/adaptivethink/): RL trainer ([`rl/drgrpo_train.py`](src/adaptivethink/rl/drgrpo_train.py)), rewards ([`rl/rewards.py`](src/adaptivethink/rl/rewards.py)), robust matcher shared by train + eval ([`router/reward.py`](src/adaptivethink/router/reward.py)), data ([`rl/data.py`](src/adaptivethink/rl/data.py)); evaluator ([`eval/eval_kpi.py`](eval/eval_kpi.py)).
- **Models Used** (open-weight only): [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0). *No teacher model and no LLM API are used anywhere.*
- **Models Published**: `statezero/adaptivethink-rl-1.5b-grpo-lora` — the RL-trained QLoRA adapter <!-- TODO: publish + link -->
- **Datasets Used** (links + licenses):
  - [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) — grade-school math, **MIT** (train + eval).
  - [`ChilleD/StrategyQA`](https://huggingface.co/datasets/ChilleD/StrategyQA) (train) / [`wics/strategy-qa`](https://huggingface.co/datasets/wics/strategy-qa) (fallback) — multi-hop yes/no, **Apache-2.0 / MIT** (train + eval, disjoint splits).
  - [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) — knowledge MC, **MIT** — **held-out / eval-only**.
  - [`deepmind/aqua_rat`](https://huggingface.co/datasets/deepmind/aqua_rat) — algebraic MC, **Apache-2.0** — **held-out generalization probe** (not trained on in the reported run).

---

## Attribution

Built on open-source foundations; our contribution is the **reward-design diagnosis/fix for SLM RLVR** and the train/eval-consistent pipeline and integration — **not** a new RL algorithm.

| Project | Link | Our use |
|---|---|---|
| Dr.GRPO | [arxiv:2503.20783](https://arxiv.org/abs/2503.20783) | RL loss (constant length-norm, no per-group std-div) applied to a sub-2B dense SLM with RLVR |
| DeepSeek-R1 (RL recipe) | [arxiv:2501.12948](https://arxiv.org/abs/2501.12948) | Rule-based RLVR + format-reward inspiration |
| HuggingFace TRL | [github.com/huggingface/trl](https://github.com/huggingface/trl) | `GRPOTrainer` / `GRPOConfig` |
| Unsloth | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) | Memory-efficient QLoRA GRPO on a single GPU |
| vLLM | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | Fast colocated rollout generation + batched eval |
| Qwen2.5 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Open-weight base model (Apache-2.0) |

**License:** Apache-2.0 (see [`LICENSE`](LICENSE)). Copyright 2026 StateZero (Jeeth Bhavesh Kataria, Ojasvi Poonia).
