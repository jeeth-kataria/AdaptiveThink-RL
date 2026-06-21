# Results & Reproducibility

- **Problem:** PS06 — "Enhancing Reasoning in SLMs (≤7B) using Reinforcement Learning" (Samsung EnnovateX AX Hackathon 2026)
- **Team:** StateZero — Jeeth Bhavesh Kataria, Ojasvi Poonia (Ramaiah Institute of Technology, Bengaluru)
- **Base model (frozen, identical for baseline and trained):** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0)
- **Method:** Dr.GRPO + rule-based verifiable rewards (RLVR). No SFT, **no LLM/teacher API anywhere**.
- **Trained artifact:** LoRA adapter `outputs/grpo-seed0-v2` (18.46M trainable params, 1.18% of the model), published at [Ojasvi-Poonia/adaptivethink-qwen2.5-1.5b-grpo](https://huggingface.co/Ojasvi-Poonia/adaptivethink-qwen2.5-1.5b-grpo).

All numbers are **Pass@1, greedy, single-shot**, produced by the **identical** evaluator
([`eval/eval_kpi.py`](../eval/eval_kpi.py)) for baseline and trained model — same `<think>/<answer>`
prompt, same answer extraction — on the **full test split** of each benchmark (no sampling).

---

## 1. Headline KPI (full test sets)

| Benchmark | Trained on? | Baseline | Ours (Dr.GRPO+RLVR) | Δ | Floor | Verdict |
|---|---|---|---|---|---|---|
| **GSM8K** (1319) | yes | 64.2 | **76.8** | **+12.6** | ≥ 50% | ✅ floor + ≥ +5% |
| **MMLU** (full 57-subject, ~14k) | **no — held out** | 45.9 | **53.9** | **+8.0** | ≥ 45% | ✅ floor + ≥ +5% |
| **StrategyQA** (687) | yes | 56.3 | **61.7** | **+5.4** | ≥ 65% | improvement ✅ (abs −3.3 under floor) |

> A balanced 16-subject MMLU subset gives an even larger gain (46.7 → **58.0**, **+11.3**); we report
> the **standard full 57-subject** number above so there is no subset-selection question.

**Verdict — KPI met under either reading of "≥ 2 of 3":**
- *Floor + ≥ +5%:* **GSM8K** and **MMLU** both fully qualify → **2 / 3**.
- *≥ +5% improvement:* GSM8K +12.6, MMLU +8.0, StrategyQA +5.4 → **3 / 3 improved**.

---

## 2. Generalization — not overfitting (held-out benchmarks)

RL touched only GSM8K + StrategyQA *train* splits. The honest test of "did we improve reasoning vs
memorize two datasets" is performance on benchmarks we **never trained on**. Both rose:

| Held-out benchmark (never trained) | Baseline | Ours | Δ |
|---|---|---|---|
| MMLU (full 57-subject) | 45.9 | 53.9 | **+8.0** |
| AQuA-RAT (test, 254) | 42.9 | 47.2 | **+4.3** |

An overfit model degrades on unseen tasks; ours improves on both → the gain is a transferable
reasoning habit (reason step-by-step, then commit), not dataset-specific answers.

---

## 3. Validation across a second seed (n=500, seed=1)

To confirm the result is not a lucky test slice, we re-ran on a fresh 500-item sample with a
different seed *before* the full-test lock-in. The deltas held (and grew):

| Benchmark | Baseline | Ours | Δ |
|---|---|---|---|
| GSM8K | 62.2 | 74.0 | +11.8 |
| StrategyQA | 56.2 | 63.4 | +7.2 |
| MMLU (16-subj) | 48.2 | 61.4 | +13.2 |

---

## 4. Efficiency (PS06's "minimal latency overhead")

The reported model wins at **single-shot greedy** — one forward pass per question. As an ablation we
ran **self-consistency (vote@8)**: it lifts the high-variance *base* model far more than our trained
model, because RL made our model **reliable enough not to need test-time sampling**.

| Config | GSM8K Pass@1 | Tokens / question |
|---|---|---|
| **Ours, 1 sample (greedy)** | **74.0%** | ~280 |
| Base, 8 samples (vote@8) | 75.2% | ~1410 |

Our trained model reaches the base model's **8-sample** accuracy in a **single** pass — ~5× less
compute. So voting is *not* part of the deployed system; it is reported only to demonstrate the
reliability/efficiency the RL bought. (vote@8 deltas trained−base: GSM8K +5.2, StrategyQA +1.4,
MMLU −1.2 — i.e. voting compresses our lead, confirming single-shot is the right operating point.)

---

## 5. How the result was achieved (the decisive insight)

The win is a **reward-design** story — exactly the failure mode PS06 names ("SLMs have high
sensitivity to reward design"):

| Run | Reward matcher | LR / steps | GSM8K Δ |
|---|---|---|---|
| v1 | strict string compare | 1e-6 / 250 | **+1.0** (looked like failure) |
| **v2 (final)** | robust extraction (`match_answer`) | 1e-5 / 500 | **+12.6** |

v1's verifiable reward scored correct answers written as expressions (`<answer>1+3+…=8 hours</answer>`
for gold `8`) or prose as **wrong** — ~16% of genuinely-correct rollouts were rewarded 0, corrupting
GRPO's advantage estimates. The *same* bug deflated our eval (base GSM8K looked like 37% when it is
really ~64%). Fixing the answer extraction in **both** the reward and the eval (standard last-number /
boolean / option-letter extraction) plus a real learning rate is what unlocked +12.6.

---

## 6. Exact training config (the reported run)

```bash
python -m adaptivethink.rl.drgrpo_train \
  --model Qwen/Qwen2.5-1.5B-Instruct --datasets gsm8k,strategyqa \
  --out outputs/grpo-seed0-v2 --steps 500 --seed 0 \
  --loss dr_grpo --kl 0.0 --group-size 8 --lr 1e-5 \
  --max-prompt-length 512 --max-completion-length 640 --save-steps 100
```
Dr.GRPO (`loss_type=dr_grpo`, KL/β=0, `scale_rewards=False`), Unsloth 4-bit QLoRA, vLLM colocated
rollouts, group size 8, ~9,076 train rows (GSM8K + StrategyQA train splits). ~3.5 h on one 24 GB GPU.

## 7. Exact reproduce commands

```bash
./run.sh setup                         # pinned stack (vLLM 0.19.1 / torch 2.10 / TRL 0.24 / Unsloth)
source .venv/bin/activate && export PYTHONPATH="$PWD/src:$PYTHONPATH"

# baseline (no adapter) and trained (with adapter), identical harness, full test sets:
python eval/eval_kpi.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
  --datasets gsm8k,strategyqa,mmlu --n 0 --mmlu-all --out results/final_baseline.json
python eval/eval_kpi.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/grpo-seed0-v2 --datasets gsm8k,strategyqa,mmlu --n 0 --mmlu-all \
  --out results/final_trained.json
```
`--n 0` = full test split; `--mmlu-all` = standard 57-subject MMLU (omit for the 16-subject spread).
Per-item completions are saved in the output JSON so any score can be re-verified offline.

## 8. Integrity

- **No leakage:** training uses only *train* splits; every benchmark scored on its *test* split;
  MMLU and AQuA are never trained on.
- **No hardcoding:** no answer tables; predictions are live generations scored by standard extraction;
  baseline and trained scored by the identical function.
- **Full test sets**, greedy, single-shot; standard full-57 MMLU (no convenient subset).
- Reward/eval logic covered by unit tests (`tests/test_reward.py`).

## 9. Raw result files (in `results/`)

| File | What |
|---|---|
| `final_baseline.json` / `final_trained.json` | headline full-test numbers (16-subj MMLU) |
| `full57_baseline.json` / `full57_trained.json` | standard full 57-subject MMLU |
| `val_baseline.json` / `val_trained.json` | n=500 seed=1 validation |
| `vote_baseline.json` / `vote_trained.json` | self-consistency vote@8 ablation |
| `ood_baseline.json` / `ood_trained.json` | held-out AQuA generalization probe |
