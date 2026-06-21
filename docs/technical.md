# Technical Documentation

## 1. Overview

AdaptiveThink-RL improves the reasoning of a **1.5B** small language model with **reinforcement
learning only** — no SFT distillation, no teacher model, no LLM API. We apply **Dr.GRPO** with a
**rule-based verifiable reward (RLVR)** to `Qwen/Qwen2.5-1.5B-Instruct`, and demonstrate +12.6% GSM8K
/ +8.0% MMLU / +5.4% StrategyQA over the same base model, at single-shot (low-latency) inference.
Full numbers: [`results.md`](results.md).

## 2. Technical Stack

| Component | Technology |
|---|---|
| Base reasoning SLM | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) |
| RL algorithm | **Dr.GRPO** via HuggingFace **TRL 0.24** `GRPOTrainer` (`loss_type=dr_grpo`) |
| Memory-efficient training | **Unsloth** + 4-bit **QLoRA** (LoRA on the 7 attn/MLP projections) |
| Rollout generation (train) | **vLLM 0.19.1**, colocated in the trainer process |
| Reward | rule-based RLVR — robust exact-match (`match_answer`) + format reward; **no reward model** |
| Evaluation | vLLM batched generation, greedy, train-consistent prompt (`eval/eval_kpi.py`) |
| Pinned runtime | torch 2.10 + cu130, transformers 4.57.6, peft, bitsandbytes |

There is **no** difficulty verifier, no DeepSeek teacher, and no external API in the reported system —
the only learning signal is the verifiable reward, computed from the base model's own rollouts.

## 3. Method & Architecture

```
GSM8K(train) + StrategyQA(train)
        │  map to the RLVR prompt (identical at train & eval):
        │     <|im_start|>system  Reason step by step inside <think></think>,
        │       then give the final answer inside <answer></answer>. <|im_end|>
        │     <|im_start|>user  Question: ... <|im_end|>
        │     <|im_start|>assistant
        ▼
  Qwen2.5-1.5B-Instruct  ──(4-bit QLoRA, Unsloth)──►  policy π_θ
        │
        │  Dr.GRPO loop (group size 8, KL/β = 0):
        │    1. vLLM samples a GROUP of completions per prompt
        │    2. reward each (RLVR, below)
        │    3. group-relative advantage  →  policy-gradient update on the LoRA params
        ▼
  LoRA adapter  outputs/grpo-seed0-v2   ──►  eval on GSM8K/StrategyQA (test) + held-out MMLU/AQuA
```

### Reward (RLVR — outcome + process)

```
reward = 1.0 · correctness  +  0.2 · format
```
- **correctness** (`rl/rewards.py::correctness_reward`): binary exact-match. The prediction is the
  `<answer>…</answer>` content; it is compared to gold with
  [`router/reward.py::match_answer`](../src/adaptivethink/router/reward.py) — a tolerant
  numeric/fraction/LaTeX/boolean comparison, then the **standard per-task extraction** (last number for
  math, stated True/False for StrategyQA, option letter for MC). No reward model, no API.
- **format** (`rl/rewards.py::format_reward`): small reward for a well-formed, ordered
  `<think>…</think><answer>…</answer>` block. Kept small (0.2) so it can never be gamed over correctness.

### Train/eval consistency

The exact same prompt builder and the exact same `match_answer` are used in the reward **and** in the
evaluator. This is deliberate: it guarantees the measured before/after delta is not confounded by a
prompt or parsing mismatch — and it was the root cause we had to fix (see §5).

## 4. OSS Libraries Used

| Library | Link | Purpose |
|---|---|---|
| HuggingFace TRL | https://github.com/huggingface/trl | `GRPOTrainer` / `GRPOConfig` (Dr.GRPO) |
| Unsloth | https://github.com/unslothai/unsloth | Memory-efficient 4-bit QLoRA GRPO on one GPU |
| vLLM | https://github.com/vllm-project/vllm | Colocated rollout generation + batched eval |
| Transformers | https://github.com/huggingface/transformers | Model / tokenizer loading |
| PEFT | https://github.com/huggingface/peft | LoRA adapters |
| BitsAndBytes | https://github.com/bitsandbytes-foundation/bitsandbytes | 4-bit quantization (training) |
| datasets | https://github.com/huggingface/datasets | GSM8K / StrategyQA / AQuA / MMLU |

## 5. Key Design Decisions

**Why Dr.GRPO (not vanilla GRPO/PPO)?** Dr.GRPO removes GRPO's length and difficulty biases (constant
length-normalization, no per-group reward-std division). With **KL/β = 0** the policy is free to move
without anchoring to the base — appropriate when the reward is a clean, verifiable signal.

**Why RLVR (rule-based reward), not a reward model?** PS06 asks for *lightweight, efficient* reward
mechanisms. A rule-based verifiable reward needs no second model, no preference data, and no API — it
is exact-match against dataset gold. This is both cheaper and less hackable than a learned reward.

**The decisive fix — answer extraction in the reward (SLM reward-sensitivity).** Our first run barely
moved (+1% GSM8K). The cause was the reward's answer-checker: it scored correct answers written as
expressions or prose as *wrong*, so ~16% of correct rollouts got reward 0 and GRPO's advantages were
corrupted. Replacing strict string-compare with the standard extraction (`match_answer`, the same one
lm-eval-harness-style harnesses use) — in **both** the reward and the eval — plus a real learning rate
(1e-6 → 1e-5) produced the +12.6. This is PS06's "high sensitivity to reward design" made concrete.

**Why single-shot greedy at inference?** It is the low-latency operating point and it already meets the
KPI. A self-consistency (vote@8) ablation showed voting helps the *base* model more than ours — RL made
our model reliable, so it does not need expensive test-time sampling (see [`results.md`](results.md) §4).

**Why LoRA (1.18% of params)?** Cheap to train on one 24 GB GPU, fast to ship (74 MB adapter), and it
constrains how far the policy can drift — which (together with held-out evals) is part of why the model
generalized instead of overfitting.

## 6. Repository Map

| Path | What |
|---|---|
| `src/adaptivethink/rl/drgrpo_train.py` | Dr.GRPO trainer (TRL + Unsloth + vLLM) |
| `src/adaptivethink/rl/rewards.py` | RLVR reward functions (correctness + format) |
| `src/adaptivethink/rl/data.py` | dataset → `<think>/<answer>` prompt mapping |
| `src/adaptivethink/router/reward.py` | `match_answer` + tolerant matcher (shared by reward & eval) |
| `eval/eval_kpi.py` | train-consistent evaluator (vLLM/HF, greedy, full-test, `--mmlu-all`) |
| `demo.py` | Mac/CPU demo: base vs trained, side by side |
| `tests/test_reward.py` | unit tests for the matcher / reward |

## 7. Installation

```bash
./run.sh setup     # builds the pinned venv (vLLM 0.19.1 / torch 2.10 / TRL 0.24 / Unsloth)
```
For inference-only on a laptop (Apple Silicon / CPU): `pip install torch transformers peft accelerate`
and run `python demo.py --compare` — no CUDA/vLLM needed.
