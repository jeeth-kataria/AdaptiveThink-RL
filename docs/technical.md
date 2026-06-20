# Technical Documentation

## Technical Stack

| Component | Technology |
|---|---|
| Reasoning SLM | DeepSeek-R1-Distill-Qwen-1.5B (QLoRA-4bit) |
| Difficulty Verifier | Qwen2.5-0.5B encoder + regression head (~400M) |
| RL Training | GRPO via HuggingFace TRL 0.12+ |
| Memory-efficient training | Unsloth + QLoRA-4bit |
| Fast rollout generation | vLLM (colocated, RTX 4090 only) |
| On-device inference | llama.cpp GGUF Q4_K_M |
| Experiment tracking | Weights & Biases |
| Checkpoint backup | HuggingFace Hub |

## Architecture

```
Question
   │
   ▼
┌─────────────────────┐
│  Difficulty Verifier │  ← 400M cross-encoder, distilled from DeepSeek-V3
│  (external tool)     │    outputs d ∈ [0,1]
└──────────┬──────────┘
           │ d
           ▼
┌─────────────────────┐
│   Routing Head       │  ← 2-token decision: <think> or <no_think>
│   (RL-trained)       │    trained with GRPO, reward = correctness − λ·tokens·(1−d)
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
<think>      <no_think>
    │             │
    ▼             ▼
Full CoT      Direct answer
reasoning     (1 step)
    │             │
    └──────┬──────┘
           │
      \\boxed{answer}
```

## OSS Libraries Used

| Library | Link | Purpose |
|---|---|---|
| HuggingFace TRL | https://github.com/huggingface/trl | GRPOTrainer |
| Unsloth | https://github.com/unslothai/unsloth | Memory-efficient GRPO |
| HuggingFace Transformers | https://github.com/huggingface/transformers | Model loading |
| PEFT | https://github.com/huggingface/peft | QLoRA adapters |
| vLLM | https://github.com/vllm-project/vllm | Fast rollout generation |
| llama.cpp | https://github.com/ggerganov/llama.cpp | GGUF quantisation + on-device inference |
| BitsAndBytes | https://github.com/TimDettmers/bitsandbytes | 4-bit quantisation |
| Weights & Biases | https://wandb.ai | Experiment tracking |
| datasets | https://github.com/huggingface/datasets | GSM8K / MATH-500 / StrategyQA / MMLU |
| scipy | https://scipy.org | Spearman ρ evaluation |

## Installation

See [`WHERE_TO_RUN.md`](../WHERE_TO_RUN.md) for environment-specific instructions.

```bash
git clone https://github.com/YOUR_ORG/adaptivethink
cd adaptivethink
bash scripts/01_setup.sh
```

## Key Design Decisions

**Why external verifier instead of internal confidence?**
AdaptThink uses internal model confidence; CODA uses group-rollout pass-rate. Both are policy-internal signals that can be noisy under quantisation. Our external verifier is trained independently and remains stable when the policy changes during GRPO training.

**Why `(1−d)` gating?**
Without gating, a uniform length penalty collapses routing on hard items (the model learns to always say `<no_think>` to avoid the penalty). The `(1−d)` term makes the penalty near-zero on hard items, allowing the model to use full CoT where it matters.

**Why GGUF Q4_K_M?**
Reduces the 1.5B model from ~3 GB (FP16) to ~1 GB, fitting comfortably on a Samsung Galaxy S24 (12 GB RAM). Accuracy drop is <1.5% on GSM8K vs FP16.
