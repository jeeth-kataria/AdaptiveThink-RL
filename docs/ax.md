# docs/ax.md — Open-Weight Models & Agentic Reasoning Approach

> Required by the Samsung EnnovateX AX Hackathon. Explains how we use open-weight models and an
> agentic, verification-driven reasoning approach to solve PS06 — and what worked vs what did not.

## 1. Open-weight models used

| Model | Role | Why |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | The reasoning SLM we train and ship | Strong open 1.5B reasoner, Apache-2.0, single-GPU-trainable, on-device-friendly. 3B / 7B are a one-flag `--model` switch (all Qwen2.5 share ChatML + the same LoRA target modules). |

This is the **only** model in the pipeline, and the solution is **fully API-free**:

- **No teacher model, no distillation, no hosted/closed LLM** anywhere in training, reward, or inference.
- The learning signal is a **rule-based verifiable reward** — exact-match against dataset gold.
- Difficulty and reward are computed from the model's **own** generations.

So every gain belongs to the small model itself — it never sees an external model's answers. All
supporting tooling is open-source (TRL, Unsloth, vLLM, PEFT).

## 2. Agentic, verification-driven reasoning

Our solution is "agentic" in two concrete, reproducible senses:

**(a) The model runs a plan-then-answer reasoning loop.** For every question it observes the prompt,
reasons step-by-step inside `<think>…</think>`, then commits a final answer inside `<answer>…</answer>`.
A process reward keeps this structure reliable, so the reasoning trace is explicit and inspectable.

**(b) Training is an autonomous generate → verify → improve loop (RLVR).** Dr.GRPO drives a closed
loop with a verifier in the loop:

```
sample a GROUP of candidate solutions (vLLM rollouts)
   → VERIFY each with the rule-based reward (correct? well-formed?)
   → compute group-relative advantage
   → update the policy (LoRA)
   → repeat
```

The **rule-based verifier is the "tool"** the loop calls at every step — there is no learned reward
model and no external service. This is what makes the reward lightweight and the whole loop
reproducible on a single GPU.

## 3. Tool use / verification

The single tool in the loop is the **answer verifier** (`src/adaptivethink/router/reward.py::match_answer`):
a deterministic checker that extracts the final answer — last number for math, True/False for
StrategyQA, option letter for multiple-choice — and compares it to gold with numeric / fraction /
boolean tolerance. It is reused **verbatim** in both the reward and the evaluator, so training and
scoring can never diverge.

## 4. Reasoning & planning pipeline

- **Outcome reward** (weight 1.0): correctness, via the verifier above.
- **Process reward** (weight 0.2): a well-formed, ordered `<think>…</think><answer>…</answer>` block.

Keeping the process reward small ensures the model optimizes for *correct reasoning*, not for gaming
the format.

## 5. Memory / context

The model carries its working state **inside the reasoning trace itself** — intermediate results live
in the `<think>` block and are consumed when it commits the `<answer>`. No external memory store is
needed; the structured trace is the context.

## 6. What worked / what did not

**Worked**
- **API-free RLVR.** A rule-based verifiable reward needs no second model — cheap, reproducible, and it
  improved reasoning measurably: GSM8K +12.6, MMLU +8.0, StrategyQA +5.4 (full test sets).
- **Getting the reward right.** The single highest-leverage change was the answer-extraction *inside the
  reward*; small models are extremely sensitive to it (full story in [`journey.md`](journey.md)).
- **Held-out validation.** MMLU and AQuA were never trained on, yet both improved — evidence of a
  genuine, transferable reasoning gain rather than dataset memorization.

**Did not work / chose to drop**
- **Self-consistency voting** lifted the high-variance base model more than our trained model, so it
  *shrank* our measured advantage — we dropped it and kept **single-shot** inference (lower latency).
- **Extra machinery** (a self-difficulty curriculum and an adaptive think/no-think router) was
  implemented but the winning run did not use it; we deliberately shipped a **single, efficient model**
  rather than a multi-component system, and document those parts honestly as optional extensions.
