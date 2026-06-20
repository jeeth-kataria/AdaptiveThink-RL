# docs/ax.md — Agentic AI & Open-Weight Model Usage

> Required by Samsung EnnovateX AX Hackathon Phase 2 guidelines.
> Explains how we used agentic development tools and open-weight models to build AdaptiveThink.

---

## 1. Open-Weight Models Used

| Model | Role | Why chosen |
|---|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Reasoning SLM (trainee) | Best open 1.5B reasoning model; Qwen2.5-Math backbone is highly RL-responsive |
| `Qwen/Qwen2.5-0.5B-Instruct` | Verifier encoder base | Compact, strong text representations; fits in 2 GB VRAM |
| `deepseek-ai/DeepSeek-V3` (via API, inference only) | Teacher for difficulty labels | Best open-weight teacher available; used only at distillation time, not in final product |

All models in the final shipped artefact are fully open-weight (Apache-2.0 / MIT). DeepSeek-V3 is used only as a teacher signal generator during data preparation — it is not part of the deployed system.

---

## 2. Agentic AI Setup

AdaptiveThink is itself an **agentic reasoning system**: it contains an agent (the router) that observes a question, consults an external tool (the verifier), and decides which reasoning strategy to invoke. This mirrors the classic agent loop:

```
Observe (question)
  → Tool call (verifier: score difficulty d)
  → Plan (router: emit <think> or <no_think>)
  → Act (reasoner: generate answer with or without CoT)
  → Verify (reward: correctness + length efficiency)
```

### 2.1 Verifier as an external tool

The 400M difficulty verifier acts as a **tool** called by the routing policy. Unlike AdaptThink (which uses internal model confidence) or CODA (which uses group-rollout pass-rate), our verifier is a separately-trained, independently-callable module. This is architecturally equivalent to tool use in an agent framework — the policy does not have direct access to the verifier's weights during inference; it only receives the scalar score `d`.

### 2.2 Reward as a planning signal

The GRPO reward function encodes a **planning objective**: minimise tokens spent on easy questions, maximise correctness on hard ones. The `(1−d)` gating term is the mechanism by which the external tool's output shapes the agent's planning horizon.

---

## 3. Agentic Development Tools Used to Build This Project

### 3.1 Kiro CLI (Claude Sonnet 4.6)

We used **Kiro CLI** (an AI coding assistant powered by Claude Sonnet 4.6) as the primary agentic development tool throughout this project. Kiro was used for:

- **Planning:** generating `plan.md`, `execution.md`, `CLAUDE.md` — the full strategic and operational plan
- **Code generation:** all source files in `src/adaptivethink/` were written with Kiro assistance
- **Novelty checking:** Kiro searched arxiv for competing papers (AdaptThink, CODA, Qwen3, R2R, DIET, GRPO-LEAD) and identified which of our original claims were already published, forcing a pivot to a defensible system-level novelty claim
- **Bug finding:** Kiro ran the reward unit tests, found two bugs (`lambda_tok` too small, obey bonus making wrong answers positive), and fixed them before any training ran
- **Documentation:** this file, `WHERE_TO_RUN.md`, and all docs were generated with Kiro

**What worked well:**
- Kiro's ability to search the web for recent papers and immediately update the novelty claims in `plan.md` saved us from submitting a proposal that would have been shot down in 30 seconds
- The `CLAUDE.md` operating instructions pattern — writing rules for the AI assistant itself — meant that every session started with the same context, preventing drift
- Kiro caught the `num_workers=4` Colab deadlock bug and the wrong DeepInfra model ID before they caused wasted runs

**What did not work:**
- Kiro's context window resets between sessions. Without `CLAUDE.md` and `results/progress.md` as persistent state, each session would start blind. The solution was to mandate progress documentation every 50 steps.
- Kiro initially generated `lambda_tok=5e-4` which was too small to penalise long responses. The unit tests caught this, but it required an explicit test-driven workflow — Kiro alone would not have caught it without tests.
- Long multi-file generation in a single message caused timeouts. We learned to split work into smaller steps.

### 3.2 Reasoning & Planning Pipeline

The development workflow was itself a multi-step reasoning pipeline:

```
Step 1: Read proposal PDF → extract claims
Step 2: Web search for competing papers → identify dead claims
Step 3: Pivot novelty → update plan.md
Step 4: Generate code → run tests → fix bugs → iterate
Step 5: Generate docs → cross-check against submission guidelines
```

Each step was a separate Kiro session, with `CLAUDE.md` providing continuity.

### 3.3 Tool Use / Tool Chaining

Within Kiro sessions, the following tools were chained:

1. `web_search` → find competing papers
2. `web_fetch` → read arxiv abstracts for CODA, AdaptThink
3. `shell` → run pytest, check file structure
4. `write` / `strReplace` → create and patch source files
5. `read` → inspect existing code before modifying

The key insight: **web_search → web_fetch → strReplace** is a tool chain that updated our novelty claims in real time based on the latest literature. This is not possible with a static LLM.

### 3.4 Memory / Context Handling

Since Kiro has no persistent memory across sessions, we implemented a manual memory system:

- `CLAUDE.md §5 Decision Log` — append-only log of every non-trivial decision
- `results/progress.md` — current training state (step, metrics)
- `execution.md §Lessons` — surprises and fixes

Rule in `CLAUDE.md §4.5`: every AI session must read these three files before acting. This is equivalent to a **retrieval-augmented agent** where the knowledge base is the project's own documentation.

---

## 4. Training Pipeline as an Agentic Workflow

The three-stage training pipeline is itself an agentic workflow with feedback loops:

```
Stage 1: Teacher (DeepSeek-V3) labels difficulty
         ↓ tool: DeepInfra API
Stage 2: Verifier trained on teacher labels
         ↓ tool: Spearman ρ eval → accept/reject/retrain
Stage 3: GRPO router trained with verifier as live tool
         ↓ tool: verifier.score() called per batch
         ↓ reward signal shapes routing policy
         ↓ wandb monitoring → human-in-the-loop intervention if diverging
```

The human acts as an **orchestrator** watching wandb dashboards and intervening if `think_rate` collapses or KL explodes. The `execution.md` Day-6 decision tree (Cases A/B/C/D) is the explicit planning document for this orchestration.

---

## 5. What We Would Do Differently

- **MCP servers:** We did not use MCP servers in this project. In future, a wandb MCP server that lets the AI assistant directly query training metrics and trigger hyperparameter sweeps would close the human-in-the-loop gap.
- **Multi-agent orchestration:** The verifier training and GRPO training are currently sequential. A multi-agent setup where one agent monitors verifier quality and another monitors GRPO health, with a coordinator deciding when to proceed, would be more robust.
- **Automated novelty checking:** We manually triggered web searches. An agent that continuously monitors arxiv for new papers in the domain and flags conflicts with our claims would be valuable for any research project.
