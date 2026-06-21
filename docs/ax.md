# docs/ax.md — Agentic AI & Open-Weight Model Usage

> Required by the Samsung EnnovateX **AX** Hackathon. Describes (1) the open-weight models used and
> (2) how agentic AI was used to build this project — honestly, including what did *not* work.

---

## 1. Open-Weight Models Used

| Model | Role | Why |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | The reasoning SLM we train (and ship) | Strong open 1.5B reasoner, Apache-2.0, single-GPU-trainable, on-device-friendly. ChatML + standard LoRA targets mean 3B/7B are a one-flag `--model` switch. |

That is the **only** model in the pipeline. There is **no teacher model and no LLM API** anywhere in
training, reward, or inference — the learning signal (RLVR) is a rule-based exact-match against dataset
gold, and the difficulty/reward is computed from the base model's own rollouts. This was a deliberate
constraint: the model never sees an external model's answers, so the gains are its own.

---

## 2. Agentic AI in the development process

The system itself is a single RL-trained reasoner (not a multi-agent product). The **agentic AI is in
how we built it**: the RL pipeline, the evaluator, the debugging, and the analysis were driven through
**Claude Code (Anthropic Claude)** as an agentic engineering partner operating an explicit
observe → act → verify loop against a real training environment.

```
Observe   read logs / eval JSON / source           (file + shell tools)
  → Plan  hypothesize the bottleneck               (reasoning)
  → Act   edit code / launch train or eval on GPU  (edit + shell-over-SSH tools)
  → Verify  re-score, compare to held-out, re-run  (measurement before more compute)
```

This loop, not a one-shot prompt, is what produced the result.

### 2.1 The standout agentic moment — diagnosing the reward bug

Our first RL run improved GSM8K by only ~+1% and looked like a failure. Instead of blindly retraining,
the agent **re-scored the saved completions offline** and discovered the verifiable reward was scoring
correct-but-differently-formatted answers (e.g. `<answer>1+3+…=8 hours</answer>` for gold `8`) as
*wrong* — corrupting ~16% of the reward signal, and deflating our eval the same way. Fixing the answer
extraction in both the reward and the eval, plus a real learning rate, turned the "failed" run into
**+12.6% GSM8K**. The agentic contribution was the *diagnosis discipline* — measure, re-score, attribute
— not raw code generation.

### 2.2 Validation discipline (agent-enforced, before spending GPU)

The agent repeatedly **validated before training more**, which saved GPU budget and kept the result
honest:
- re-scored saved completions offline to find the true baseline before any retrain;
- added **held-out** probes (MMLU, AQuA — never trained on) to test for overfitting vs generalization;
- re-ran on a **second seed and the full test sets** to kill sampling-noise doubts;
- ran a **self-consistency (vote@8) ablation**, found it compresses our lead, and *dropped* it rather
  than ship a misleading number.

### 2.3 Honest reframing

An earlier project framing centered a novelty (a self-difficulty curriculum, "CCDD") and an adaptive
router that the **winning run did not actually use**. The agentic review caught this and **reframed the
submission around what was genuinely run** (Dr.GRPO + RLVR + the reward fix), demoting the unrun
components to clearly-labeled "implemented extensions, not part of the reported result." Genuineness
over claims.

---

## 3. Tool use / tool chaining

| Tool | Use in this project |
|---|---|
| **shell** (over SSH to a rented GPU) | launch/monitor Dr.GRPO training and vLLM evals; inspect `nvidia-smi`, logs, result JSON |
| **file read / edit / write** | author the trainer, reward, evaluator, demo, docs; apply surgical fixes |
| **web search** | confirm Qwen2.5-1.5B reference scores, GRPO/Dr.GRPO recipes, dependency-version compatibility |
| **offline re-score (Python)** | the key chain: *read result JSON → re-extract answers with corrected logic → recompute Pass@1* — diagnosing the reward bug without spending a single GPU-hour |
| **persistent memory files** | project / strategy / results notes carried across sessions (see §4) |

The decisive chain was **`read saved completions → re-score offline → compare to held-out → only then retrain`** — it converted an apparent failure into the winning result and prevented wasted training runs.

---

## 4. Memory / context handling

Agent sessions reset and long runs span hours, so we used an explicit, file-based memory the agent
read at the start of every session and updated as facts changed:
- **project** notes (KPI, constraints, hardware), **strategy** (the winning plan), and a live
  **run-results** log (each run's numbers, what worked, what was rejected and why).

This is effectively a small retrieval-augmented memory whose knowledge base is the project's own
notes — it is what let the work survive context compaction and SSH disconnects without losing the
thread (e.g. the reward-bug finding and the locked numbers persisted across sessions).

---

## 5. Human-in-the-loop, and what did / didn't work

**Worked well**
- *Agent as diagnostician.* The biggest win (the reward-extraction fix) came from the agent
  re-scoring saved data, not from generating more code.
- *Measure-before-compute.* Offline re-scoring and held-out probes caught issues cheaply and kept the
  result defensible (no leakage, standard extraction, full test sets).
- *Honest self-correction.* The agent dropped its own earlier over-claims (CCDD/router/voting) once
  the data didn't support them.

**Didn't work / lessons**
- *No direct access to the remote GPU.* The agent could not see the Vast box; the human relayed logs
  and ran commands — a genuine human-in-the-loop. Tighter integration (an MCP server exposing
  training metrics) would close this gap.
- *Long jobs + SSH drops.* Background runs needed `nohup`/`tmux`; block-buffered stdout made a healthy
  run look "stuck" until we forced `PYTHONUNBUFFERED=1` — an avoidable detour.
- *Initial over-engineering.* The agent first proposed elaborate machinery (curriculum + router +
  voting); the actual win was a small, correct reward fix. Lesson: diagnose the simple thing first.

---

## 6. Reproducibility of the agentic workflow

Every code change the agent made is in the repo and unit-tested (`tests/`); every result is
reproducible from the commands in [`results.md`](results.md) and [`../README.md`](../README.md) using
only the open-weight base model and the provided LoRA adapter — no API keys, no closed models.
