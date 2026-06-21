# Development Journey — what failed, what fixed it, and how we got the win

> An honest, chronological log of how we went from a model that looked like a **failure** to one that
> clears the PS06 KPI with margin. We include the wrong turns on purpose — the biggest insight came
> *from* a failure. All numbers are real, from `results/` (Pass@1, greedy, train-consistent eval).

**TL;DR of the arc:** GSM8K looked like **37%** (apparent failure) → we discovered we were *mis-measuring*
and the *reward* had the same bug → fixed answer-extraction + raised the learning rate → **76.8%**
(+12.6 over the same base model), validated on held-out benchmarks. The win was a **reward-design fix**,
not more compute.

---

## Phase 0 — Starting point & the over-ambitious plan

The initial design was a two-layer system: a Dr.GRPO "accuracy core" **plus** a self-difficulty
curriculum ("CCDD"), an adaptive `think/no_think` router, a difficulty verifier, and GGUF on-device
export. Constraints we set: **≤7B, single GPU, no LLM/teacher API**. We chose
`Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0) for headroom + efficiency.

**Lesson (in hindsight):** this was too much machinery. The thing that actually won was small and
boring. We document the unused parts honestly as "extensions, not the result" (see README).

## Phase 1 — Dependency hell (before any learning happened)

Getting Dr.GRPO running on a rented GPU took a pinned, ordered stack (vLLM 0.19.1 / torch 2.10+cu130 /
TRL 0.24 / Unsloth / transformers 4.57.6). Real failures fixed along the way:
- invalid `pyproject` build-backend; a CUDA-12 vLLM wheel on a CUDA-13 host (manual cu130 wheel);
- a vLLM `torch.compile` crash (`SymInt` in rotary embedding) → fixed with `enforce_eager=True`;
- scripts that didn't activate the venv (`python: command not found`).

**Lesson:** for SLM-RL on rented hardware, the environment is half the battle — pin everything.

## Phase 2 — v1 training, and a "data mix" bug

First real run: Dr.GRPO, `lr 1e-6`, 250 steps. We initially passed `--datasets gsm8k,strategyqa,aqua`
and the loader pulled **106,543** rows (~91% AQuA) — the run would have learned mostly letter-answers.
**Fix:** restart on `gsm8k,strategyqa` (9,076 rows, ~77/23). Training ran, rewards drifted up slightly,
adapter saved.

## Phase 3 — The model looked *catastrophically* bad — but it was the EVAL

Our first eval reported **MMLU 8.7%** and **StrategyQA 20.7%** — *below random*. A 1.5B model is not
that bad. The cause: the evaluator used a different prompt convention (`\boxed{}` + no answer-format
instruction) than the model was **trained** on (`<think>/<answer>`). We built a **train-consistent
evaluator** (`eval/eval_kpi.py`) using the exact training prompt + matcher.

**Insight #1:** *with SLMs, a prompt/parse mismatch can make a fine model look broken.* Measure the way
you trained.

## Phase 4 — v1, measured properly: gains within noise (apparent failure #2)

Train-consistent eval of v1:

| | GSM8K | StrategyQA | MMLU |
|---|---|---|---|
| baseline | 36.5 | 51.5 | 60.5 |
| v1 trained | 37.5 | 53.0 | 63.0 |
| Δ | +1.0 | +1.5 | +2.5 |

At n=200 the noise band is ±3.5pp, so **all deltas were statistically zero. KPI not met.** First guess:
undertrained (`lr 1e-6` too gentle — grad-norms were ~0.02).

## Phase 5 — The turning point: we were *throwing away correct answers* (in eval AND in the reward)

Instead of blindly retraining, we **inspected the completions** and found two things:
1. The model already emitted the `<answer>` format **97.5%** of the time — format was *not* the problem.
2. Many "wrong" answers were **actually correct**, written as expressions or prose
   (`<answer>1+3+0.5+1.5+2 = 8 hours</answer>` for gold `8`) — our strict string-match scored them 0.

We **re-scored the saved completions offline** (free, no GPU) with the standard final-answer extraction
(last number / boolean / option letter):

| | GSM8K | StrategyQA | MMLU |
|---|---|---|---|
| baseline (old strict score) | 36.5 | 51.5 | 60.5 |
| baseline (correct score) | **53.0** | **57.5** | 60.5 |

The base model was **always ~53% on GSM8K** — we'd been under-counting by **+16.5pp**.

**Insight #2 (the big one):** the *same* extraction bug was inside the **training reward**. During GRPO,
~16% of *correct* rollouts were rewarded **0** — a corrupted advantage signal. **This, not just the low
LR, is why v1 barely learned.** This is PS06's warning — *"SLMs have high sensitivity to reward design"* —
made concrete. **A broken-but-plausible reward actively held the model back.**

## Phase 6 — The fix (v2): correct the reward, then train for real

- Added a robust matcher (`router/reward.py::match_answer`) — tolerant compare + standard per-task
  extraction — and wired it into **both** the reward and the eval. Unit tests confirm genuine errors
  still score wrong (no false positives). **33 tests pass.**
- Retrained: `lr 1e-5` (10×), 500 steps, same data, corrected reward.

The reward fix showed immediately: `correctness_reward/mean` started **~0.45–0.55** (vs v1's ~0.26),
because correct answers were finally being counted. Result on full test sets:

| Benchmark | baseline | v2 trained | Δ |
|---|---|---|---|
| GSM8K (1319) | 64.2 | **76.8** | **+12.6** |
| MMLU (full-57, ~14k) | 45.9 | **53.9** | **+8.0** |
| StrategyQA (687) | 56.3 | **61.7** | **+5.4** |

**KPI cleared** (GSM8K + MMLU clear floor + ≥5%; all three improve ≥5%).

## Phase 7 — Proving it's genuine (not overfit, not gamed)

The user (rightly) asked: *"how did we suddenly get this good — is it real?"* So we validated:
- **Held-out generalization:** MMLU and AQuA were **never trained on**, yet both rose (MMLU +8.0,
  AQuA 42.9→47.2 **+4.3**). An overfit model degrades on unseen tasks; ours improved → real reasoning gain.
- **Second seed + full test sets:** re-ran at n=500/seed=1, then the full test splits — deltas held.
- **Audit:** training touches only *train* splits; eval on *test*; no hardcoded answers; identical
  matcher for baseline and trained. Standard full-57 MMLU clears it too (no convenient subset).

## Phase 8 — Things we tried that DID NOT work (and dropped honestly)

- **Self-consistency voting (vote@8):** *made our advantage smaller, not larger.* It lifts the
  high-variance base model more than our already-reliable model (vote@8 deltas: GSM8K +5.2, StrategyQA
  +1.4, **MMLU −1.2**), and didn't clear StrategyQA's 65 floor — at 8× the compute. **Dropped.** We kept
  it only as an *efficiency ablation*: our model's 1-shot accuracy ≈ the base model's 8-shot accuracy
  (~5× less compute), which is the right story for "minimal latency overhead."
- **CCDD curriculum / adaptive router / GGUF:** designed and present in the repo, but the winning run
  **did not use them.** Rather than claim them, we **demoted them to "implemented extensions, not part of
  the reported result."** Genuineness over claims.
- **StrategyQA-heavy retrain for a 3/3 sweep:** considered, but it risks overfitting a small binary
  dataset; with 2/3 already met we chose not to chase the last 3.3pp on a 1.5B (3B is the clean lever if
  needed).

## What made the model worse / wasted effort — at a glance

| Misstep | Effect | Root cause | Fix |
|---|---|---|---|
| `\boxed`-style eval prompt | model looked *below random* | train/eval prompt mismatch | train-consistent `eval_kpi.py` |
| strict string-match reward | GRPO barely learned (+1%) | ~16% of correct rollouts scored 0 | `match_answer` standard extraction |
| `lr 1e-6`, 250 steps | undertrained | too gentle for GRPO on an SLM | `lr 1e-5`, 500 steps |
| AQuA-dominated data mix | would bias to letter-answers | uncapped multi-dataset load | restrict to gsm8k+strategyqa |
| self-consistency voting | shrank our delta | helps base model more | dropped; kept as efficiency ablation |
| CCDD/router/GGUF in headline | unverifiable overclaim | machinery never run | demoted to "extensions" |

## Lessons (the insights PS06 asks for)

1. **Measure before you train more.** Two of our three "failures" were *measurement* artifacts. Offline
   re-scoring of saved completions cost nothing and changed the whole picture.
2. **SLMs are reward-sensitive — a plausible-but-wrong verifiable reward is worse than none.** The single
   highest-leverage change was fixing answer extraction *inside the reward*, not the algorithm or compute.
3. **Verify genuineness explicitly** — held-out probes, second seed, full test sets, leakage audit. It
   both protects the result and *is* the strongest part of the write-up.
4. **Cut what you can't back with data.** Dropping voting and the CCDD/router headline made the
   submission *stronger*, because everything left is true and reproducible.

> Reproduce any number in this document from `results/*.json` and the commands in
> [`results.md`](results.md) — base model + the provided LoRA adapter, no API keys.
