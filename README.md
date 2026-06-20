# AdaptiveThink

- **Problem Statement Number** - PS06
- **Problem Statement Title** - Reinforcement Learning for Small Language Model (SLM) Reasoning
- **Team name** - StateZero
- **Team members (Names)** - Jeeth Bhavesh Kataria, Ojasvi Poonia
- **Institute/College Name** - Ramaiah Institute of Technology, Bengaluru, Karnataka - 560054
- **Final Presentation Google Drive Link** - *(to be added)*
- **Full Submission Demo Video Link** - *(YouTube link — to be added after training)*
- **Setup & Result Reproducibility Video Link** - *(YouTube link — to be added after training)*

---

### Project Artefacts

- **Technical Documentation** - [`docs/technical.md`](docs/technical.md) — stack, architecture, OSS libraries, installation, user guide. [`docs/ax.md`](docs/ax.md) — agentic AI usage.
- **[Important]** [`docs/ax.md`](docs/ax.md) — detailed explanation of open-weight model usage, agentic workflows, tool chaining via Kiro CLI, reasoning & planning pipelines, memory/context handling, what worked and what did not.
- **Source Code** - [`src/adaptivethink/`](src/adaptivethink/) — all training, evaluation, and deployment code.
- **Models Used**
  - [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) — reasoning SLM (trainee)
  - [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — verifier encoder base
  - deepseek-ai/DeepSeek-V3 (via API, inference-only for teacher labels — not in final product)
- **Models Published**
  - `statezero/verifier-400m` *(to be published after training)*
  - `statezero/router-1p5b-lora` *(to be published after training)*
- **Datasets Used**
  - [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) — MIT License
  - [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) — MIT License
  - [wics/strategy-qa](https://huggingface.co/datasets/wics/strategy-qa) — Apache 2.0
  - [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) — MIT License
- **Datasets Published**
  - `statezero/difficulty-labels` — teacher-labelled difficulty scores for 12k math/reasoning questions, Apache 2.0 *(to be published after training)*

---

### What is AdaptiveThink?

SLMs waste compute by running full chain-of-thought on every query — easy or hard. AdaptiveThink fixes this with three coupled components:

```
Question
   │
   ▼
┌──────────────────────┐
│  Difficulty Verifier  │  400M cross-encoder distilled from DeepSeek-V3
│  (external tool)      │  outputs difficulty score d ∈ [0,1]
└──────────┬───────────┘
           │ d
           ▼
┌──────────────────────┐
│  RL-Trained Router    │  2-token decision: <think> or <no_think>
│  GRPO on 1.5B SLM    │  reward = correctness − λ·tokens·(1−d)
└──────────┬───────────┘
           │
    ┌──────┴──────┐
<think>       <no_think>
Full CoT      Direct answer
    └──────┬──────┘
      \boxed{answer}
```

**Key novelty vs prior work:** The length penalty is gated by `(1−d)` where `d` comes from an *external* distilled verifier — not internal model confidence (AdaptThink) or group-rollout pass-rate (CODA). Easy questions pay full penalty; hard questions pay near-zero, preventing routing collapse on difficult items.

### Quick Start

```bash
git clone https://github.com/jeeth-kataria/AdaptiveThink.git
cd AdaptiveThink
bash scripts/01_setup.sh
```

See [`WHERE_TO_RUN.md`](WHERE_TO_RUN.md) for exact commands per environment (Colab T4 for data/verifier, Vast.ai RTX 4090 for GRPO training).

### End-to-end pipeline

```bash
bash scripts/01_setup.sh                  # env + pinned deps
bash scripts/02_gen_teacher_labels.sh     # DeepSeek-V3 difficulty labels: pool + eval (Stage 1 data)
bash scripts/02b_prep_train_data.sh       # GSM8K train {question,answer} -> data/gsm8k_train_labelled.jsonl
bash scripts/03_train_verifier.sh         # Stage 1: 400M difficulty verifier
bash scripts/04_train_grpo_router.sh 0    # Stage 2: GRPO router (seed 0; repeat 1,2)
bash scripts/05_eval.sh 200               # baseline vs router + Pareto/KPI table
bash scripts/06_quantize.sh               # Stage 3: GGUF Q4_K_M export
bash scripts/07_ttrl.sh 0                 # optional Idea A: Test-Time RL ablation
```

`scripts/05_eval.sh` writes the KPI delta table to `results/figures/kpi_table.md`
(flags whether ≥ +5% on ≥ 2 of GSM8K/MMLU/StrategyQA is met) and Pareto charts
(accuracy vs compute) per benchmark. Configs for every stage live in [`configs/`](configs/).

| Stage | Code | Config |
|---|---|---|
| 1 — Verifier distillation | [`src/adaptivethink/verifier/`](src/adaptivethink/verifier/) | [`configs/verifier_distill.yaml`](configs/verifier_distill.yaml) |
| 2 — GRPO router | [`src/adaptivethink/router/`](src/adaptivethink/router/) | [`configs/grpo_router.yaml`](configs/grpo_router.yaml) |
| 3 — Quantize + inference | [`src/adaptivethink/quantize/`](src/adaptivethink/quantize/), [`src/adaptivethink/inference/`](src/adaptivethink/inference/) | — |
| Eval harness | [`eval/run_benchmarks.py`](eval/run_benchmarks.py), [`eval/plots.py`](eval/plots.py) | — |
| Optional — TTRL | [`src/adaptivethink/ttrl/`](src/adaptivethink/ttrl/) | [`configs/ttrl_ablation.yaml`](configs/ttrl_ablation.yaml) |

### Attribution

| Project | Link | Our use / new features |
|---|---|---|
| DeepSeek-R1 | [arxiv:2501.12948](https://arxiv.org/abs/2501.12948) | Base reasoning model; we add RL routing head |
| HuggingFace TRL | [github.com/huggingface/trl](https://github.com/huggingface/trl) | GRPOTrainer |
| Unsloth | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) | Memory-efficient GRPO on single GPU |
| Weaver (Stanford) | [arxiv:2506.18203](https://arxiv.org/abs/2506.18203) | Verifier distillation architecture; we integrate it as a live reward signal |
| AdaptThink | [arxiv:2505.13417](https://arxiv.org/abs/2505.13417) | Adaptive thinking baseline; we extend with external verifier gating |
| CODA | [arxiv:2603.08659](https://arxiv.org/abs/2603.08659) | Difficulty-gated reward baseline; our `d` is external not internal |
| TTRL | [arxiv:2504.16084](https://arxiv.org/abs/2504.16084) | Test-time RL add-on |
| llama.cpp | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF quantisation + on-device inference |
