# Winning PS06: A Ranked Strategy Guide for Enhancing SLM Reasoning with RL

## TL;DR

- **The single highest-probability path to winning is NOT the user's complex router; it is a clean, well-engineered "distillation cold-start → GRPO/Dr.GRPO with rule-based verifiable rewards + stability tricks" pipeline on a deliberately _non-saturated_ base model (Qwen2.5-3B-Instruct or Qwen2.5-1.5B-Instruct, NOT the math-distilled DeepSeek-R1-Distill-Qwen-1.5B), targeting GSM8K and StrategyQA as the two improvement benchmarks.** This reliably clears the "+5% over baseline" bar, is demo-able, and is robust within 2 weeks.
- **The user's "RL-trained router + tiny verifier + on-device" idea is genuinely interesting but its novelty claim is weak**: AdaptThink (arXiv:2505.13417), Thinkless (arXiv:2505.13473), and RouteLLM (Ong et al. 2024) already RL-train think/no-think gating, and System-1.5 (arXiv:2505.18962) and s1 (arXiv:2501.19393) already do adaptive compute. It is best positioned as a _secondary "wow" module_ (efficiency/Pareto + Samsung on-device mapping) bolted onto a solid accuracy core, not as the primary KPI engine.
- **The biggest hidden risk is the baseline ceiling.** DeepSeek-R1-Distill-Qwen-1.5B is math-specialized, so GSM8K is already very high (a +5% gain there is nearly impossible) while its non-math MMLU is borderline (~35–45%). Pick a base model with headroom, treat MMLU as a "maintain/secondary" metric (RL on math reasoning barely moves MMLU knowledge), and pick **GSM8K + StrategyQA** as your two improvement targets.

## Key Findings

1. **RL "sharpens" rather than "expands" reasoning at small scale.** Yue et al. (_Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?_, arXiv:2504.13837, Tsinghua LeapLab) show RLVR-trained models beat base models at pass@1 but base models match or exceed them at large pass@k — on AIME24, **over 95% of problems solved by the RL model could also be solved by the base model given enough attempts**, a result that held across six RL methods (PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO) with base models surpassing RL models at pass@256. RL boosts sampling efficiency of paths already in the base model and _narrows_ the reasoning boundary; distillation genuinely adds new reasoning patterns. **Implication: for a hackathon judged on pass@1/accuracy, RL is exactly the right tool, but a distillation cold-start is what raises the raw capability ceiling.**

2. **The algorithm that matters most for ≤7B is GRPO plus a few well-chosen fixes — not exotic methods.** Dr.GRPO (arXiv:2503.20783) removes GRPO's length and std-normalization biases; DAPO (arXiv:2503.14476) adds Clip-Higher, dynamic sampling, token-level loss and overlong reward shaping; GSPO (arXiv:2507.18071, Qwen) uses sequence-level importance ratios (mainly a win for MoE/large models). A direct SLM ablation (Alex Lavaee, _DAPO: A Case Study with Small Language Models_, Qwen2.5-0.5B) found that on tiny models **Dr.GRPO loss + overlong filtering + removing the KL penalty + disabling reward scaling all helped, while DAPO's Clip-Higher, token-level loss, soft overlong punishment, and dynamic sampling actually hurt.**

3. **Rule-based verifiable rewards (RLVR) beat process reward models (PRMs) at small scale.** DeepScaleR (1.5B) used a binary correct/incorrect reward with explicitly "**No partial rewards (such as PRMs) or intermediate feedback**" and went 28.8%→43.1% pass@1 on AIME 2024 (trained on ~40,000 problem-answer pairs). PRM surveys (arXiv:2510.08049, arXiv:2509.03403) note PRMs "frequently suffer from inaccuracies and are susceptible to reward hacking." **Use rule-based exact-match + a light format reward.**

4. **Data efficiency is shockingly high.** 1-shot RLVR (arXiv:2504.20571, U. Washington/Microsoft/USC) lifted Qwen2.5-Math-1.5B on MATH500 from **36.0%→73.6%** and lifted the average across six math benchmarks from 17.6%→35.7% with a _single_ training example — matching the 1.2k-example DeepScaleR subset (MATH500: 73.6%; avg: 35.9%), and the effect was reproduced on DeepSeek-R1-Distill-Qwen-1.5B and with both GRPO and PPO. **You do not need huge data or compute to show a +5% gain in 2 weeks.**

5. **Entropy collapse is the dominant SLM-RL failure mode, and it has cheap fixes.** Cui et al. (_The Entropy Mechanism of RL for Reasoning LLMs_, arXiv:2505.22617, PRIME-RL) establish R = −a·exp(H) + b (performance is bottlenecked by entropy exhaustion) and propose Clip-Cov/KL-Cov, which clip or KL-penalize high-covariance tokens. DAPO's Clip-Higher and a small entropy bonus are simpler mitigations. A 2026 study (arXiv:2604.06298) found that for sub-3B models _hard, unsolvable samples actively hurt_ (high-variance gradients) — filter to easy/medium difficulty.

6. **Single-GPU RL is feasible.** Unsloth advertises ~70% less VRAM and 2× faster GRPO (Qwen3-1.7B GRPO reportedly fits in ~5GB); multiple papers train 1.5B–3B GRPO on a single RTX 3080Ti/4070Ti/4090 with QLoRA. **The given 2×24GB hardware is comfortably sufficient.**

7. **The user's novelty claim is largely pre-empted.** AdaptThink (arXiv:2505.13417), Thinkless (arXiv:2505.13473, NeurIPS 2025, uses a Decoupled-GRPO/"DeGRPO" strategy with `<short>`/`<think>` control tokens), and RouteLLM all RL-train think/no-think or routing decisions on 1.5B-class models; System-1.5 (arXiv:2505.18962) and s1 budget forcing (arXiv:2501.19393) cover adaptive compute. The "tiny distilled verifier as router" angle (Weaver, Stanford Hazy Research / Scaling Intelligence Lab) is the freshest piece, but routing-on-difficulty itself is established.

## Details: Ranked Strategies

### Strategy 1 (RECOMMENDED PRIMARY) — Distillation cold-start + clean GRPO/Dr.GRPO with rule-based rewards on a non-saturated base

**What it is.** Pick **Qwen2.5-3B-Instruct** (or 1.5B-Instruct) as the base — deliberately NOT the math-saturated DeepSeek distill. Optionally do a short SFT "cold start" on a few thousand high-quality CoT traces (DeepSeek-R1 distilled traces / OpenThoughts / s1K-style) to install the reasoning format, then run GRPO with **Dr.GRPO loss** (constant normalization to kill length + std-normalization bias), a **binary correctness reward + small format reward** (`<think>`/`<answer>` tags), KL kept small or removed, plus an entropy/Clip-Higher safeguard. Train ~1–2k steps with QLoRA on Unsloth + TRL with colocated vLLM rollouts.

**Why it could win.**

- _KPIs:_ Baselines have headroom. Qwen2.5-3B-Instruct ≈ 84% GSM8K / 63.8% MMLU; Qwen2.5-1.5B-Instruct ≈ 68% GSM8K / 55% MMLU (independent eval, arXiv:2510.19208; base MMLU 60.9 in the Qwen2.5 Technical Report, arXiv:2412.15115). Both clear the ≥50% GSM8K / ≥45% MMLU floors with room for +5%. DeepScaleR proved GRPO yields large pass@1 gains on a 1.5B model.
- _Robust/low-risk:_ most-replicated recipe in the literature (DeepSeek-R1, SimpleRL-Zoo, DeepScaleR).
- _Demo-able:_ clean before/after bars, Pareto and reward/entropy curves.
- _Feasible:_ single-GPU, hours of training.

**Key risks.** Choosing a base already high on GSM8K (the 7B is ~85–87%, near ceiling) shrinks headroom — use 1.5B/3B and/or lean on StrategyQA. Entropy collapse if KL/clip mis-set; mitigate per Strategy 6.

**Evidence base.** DeepScaleR (HF: agentica-org/DeepScaleR-1.5B-Preview, Notion blog); Dr.GRPO arXiv:2503.20783; SimpleRL-Zoo arXiv:2503.18892 (format reward + difficulty control drive most gains); DeepSeek-R1 arXiv:2501.12948 (distillation > RL for small models); Lavaee SLM ablation (alexlavaee.me/projects/reasoning-slms).

### Strategy 2 — The user's "System-1.5 router + tiny verifier + on-device," REFRAMED as an efficiency/wow layer on top of Strategy 1

**What it is.** Keep the adaptive-compute idea (route easy questions to a short/no-think path, hard ones to full CoT) and the Weaver-style 400M verifier, but treat it as the _efficiency + deployment story_, with Strategy 1 supplying the accuracy gains. Use AdaptThink/Thinkless-style RL (reward = correctness − λ·tokens) to learn the gate.

**Why it could win.** Strong wow factor + concrete Samsung Galaxy on-device mapping (the EnnovateX/COD3INE template) + a compelling Pareto. **AdaptThink reduced the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 50.9% (GSM8K), 63.5% (MATH500) and 44.7% (AIME2024) while improving accuracy by +4.1%, +1.4% and +1.6% respectively** (≈53% fewer tokens, +2.4% accuracy on average). System-1.5 reported >20× inference speedup and 92.31% average token reduction on GSM8K with comparable accuracy. The distilled **Weaver 400M cross-encoder retains 98.7% of the full verifier-ensemble accuracy while cutting verification compute by up to 99.97%** (35.35 → 1.01 exaFLOPs per 100 samples; arXiv:2506.18203).

**Key risks.** (a) Novelty is contestable — judges who know AdaptThink/Thinkless/RouteLLM won't see it as "first"; be honest and frame it as a _combination + on-device_ contribution. (b) Execution: training a router head + verifier + reasoner in 2 weeks is a lot; the gate can collapse to always-think or always-skip (AdaptThink needed an importance-sampling balancing strategy to avoid this). (c) Routing mainly _saves tokens_, it doesn't directly raise accuracy — the "+5%" must come from the Strategy-1 core.

**Evidence base.** System-1.5 arXiv:2505.18962; AdaptThink arXiv:2505.13417; Thinkless arXiv:2505.13473 (github.com/VainF/Thinkless); RouteLLM (Ong et al. 2024); Weaver (hazyresearch.stanford.edu/blog/2025-06-18-weaver; arXiv:2506.18203); s1 arXiv:2501.19393.

### Strategy 3 — Minimal-data / 1-shot or LIMR-style RLVR as an efficiency flex

**What it is.** Demonstrate that with ~1–1.2k (or even one) carefully chosen verifiable examples you match large-data RL — a striking, cheap, reproducible result.

**Why it could win.** High insight/wow per compute; trivially reproducible; pairs perfectly as an ablation inside Strategy 1.

**Key risks.** On its own it's an ablation, not a full solution; gains are math-centric; "post-saturation generalization" can produce gibberish on the training example after ~1.4k steps (test accuracy stays strong, but inspect outputs).

**Evidence base.** 1-shot RLVR arXiv:2504.20571 (36.0%→73.6% MATH500; validated on DeepSeek-R1-Distill-Qwen-1.5B); LIMR.

### Strategy 4 — TTRL (test-time RL) as a secondary novelty ablation

**What it is.** Use majority-vote pseudo-labels at test time to RL-adapt on the eval distribution; add Clip-Cov/entropy mitigation and confidence-weighted voting.

**Why it could win.** Eye-catching ("learns with no labels"); **TTRL boosted Qwen2.5-Math-7B pass@1 ~211% on AIME 2024 using only unlabeled test data, with an average ~76% gain across AIME24/AMC/MATH-500/GPQA** (arXiv:2504.16084, NeurIPS 2025).

**Key risks.** Borderline "training on the test set" optics — judges may view it as benchmark gaming; relies on majority vote being right; collapses when the model is systematically wrong. **Use only as a clearly-labeled ablation, never the headline.**

**Evidence base.** TTRL arXiv:2504.16084; ETTRL arXiv:2508.11356.

### Strategy 5 — GSPO / DAPO / advanced variants

**What it is.** Use Qwen's GSPO (sequence-level optimization) or full DAPO instead of plain GRPO.

**Why it could win.** Latest-and-greatest narrative; GSPO underpins Qwen3.

**Key risks.** GSPO's proven benefit is MoE/large-model stability (it solves expert-activation volatility) — limited evidence of advantage on small _dense_ models; DAPO's extra tricks _hurt_ tiny models in the Lavaee ablation. Higher complexity, little KPI upside. **Cherry-pick only Dr.GRPO's bias fixes and DAPO's Clip-Higher; skip the rest.**

**Evidence base.** GSPO arXiv:2507.18071 (qwenlm.github.io/blog/gspo); DAPO arXiv:2503.14476; Lavaee ablation.

### Strategy 6 (CROSS-CUTTING) — Reward design + stability toolkit (use inside whichever strategy wins)

- **Reward:** binary exact-match correctness (RLVR) + small format reward (`<think>`/`<answer>`); avoid PRMs at this scale. SimpleRL-Zoo shows format reward + difficulty control drive most of the gains.
- **Length/efficiency:** Dr.GRPO loss (constant normalization) to avoid verbosity bias; optional soft length penalty for the on-device story.
- **Stability:** small or zero KL (DeepScaleR kept a KL term; Lavaee found removing it helped tiny models — test both), Clip-Higher and/or a small entropy bonus, Clip-Cov/KL-Cov if entropy collapses.
- **Curriculum:** filter out unsolvable-hard items for sub-3B models (arXiv:2604.06298); diversity-of-difficulty batching (interconnects.ai).

## Strategic / Meta Advice

**Base model + baseline selection (most important decision).** Do NOT use DeepSeek-R1-Distill-Qwen-1.5B as the base for the headline result: it is math-specialized, so GSM8K is already very high (a +5% gain is near-impossible) and its non-math MMLU is borderline (~35–45%, weakly sourced — verify on the Open LLM Leaderboard details dataset). Use **Qwen2.5-3B-Instruct** (≈84% GSM8K, ≈64% MMLU) or **Qwen2.5-1.5B-Instruct** (≈68% GSM8K, ≈55% MMLU) — both clear the floors with headroom for +5%. For comparison, Phi-3-Mini (3.8B) scores 82.5% GSM8K / 68.8% MMLU (arXiv:2404.14219), and Gemma-2-2B's pretrained GSM8K is only ~24–25% (a _poor_ base — would fail the GSM8K floor without heavy tuning). **Report the exact same eval harness** (lm-eval-harness), identical prompts and shot counts, for both baseline and trained model — note that published GSM8K varies 4-shot vs 8-shot-CoT, so fix it.

**Which two benchmarks to target.** Target **GSM8K** (RL on math directly moves it; clean rule-based reward) and **StrategyQA** (multi-hop yes/no; small ≤7B models sit ~60–73%, e.g., CoT baselines around 73.2%, leaving room for +5%; the original CoT-prompting prior SOTA was 69.4%). Treat **MMLU as the "maintain" metric**: it is mostly knowledge/multiple-choice, and RL on math reasoning does little for it (and can hurt) — your story is "improved on GSM8K + StrategyQA while maintaining MMLU above the floor."

**Maximize wow while minimizing risk.** Land the accuracy gains with Strategy 1 first (week 1). Then add ONE novelty module in week 2 — the adaptive-compute/router + Samsung on-device GGUF deployment — framed honestly as an _efficiency Pareto + deployment_ contribution, with 1-shot RLVR as a cheap "insight" ablation. This de-risks the hard KPI gate while still giving judges a memorable demo.

**Reproducibility/presentation (this wins hackathons).** Headline metric front and center (e.g., "+X% GSM8K, +Y% StrategyQA, MMLU maintained, ~50% fewer tokens"); an ablation table isolating each reward/stability component; a Pareto chart (accuracy vs tokens/latency); **multi-seed runs with confidence intervals** (RL is high-variance, and tiny eval sets swing on a single question); reward & entropy curves to prove training stability; one clean qualitative `<think>`/`<answer>` example; and on-device latency numbers from the quantized GGUF model.

**Honest verdict on the user's proposed approach.** The "learn WHEN to think" router + tiny verifier + on-device deployment is a _good demo and good story_, but as the _primary_ KPI engine it is high-risk: the novelty is contested by AdaptThink/Thinkless/RouteLLM, routing saves tokens rather than directly raising accuracy, and it couples three components to train in 2 weeks. **Recommendation: invert the plan.** Make clean distillation-cold-start + Dr.GRPO/RLVR the accuracy core (reliably clears +5%), and keep the router/verifier/on-device piece as the efficiency-and-deployment "wow" layer. This maximizes P(win) by satisfying the hard KPI gate first and adding novelty second.

## Recommendations (staged)

1. **Day 1–2:** Lock base model (Qwen2.5-3B-Instruct primary; 1.5B-Instruct fallback). Establish baselines on GSM8K, MMLU, StrategyQA with a fixed lm-eval-harness config. **Threshold to change plan:** if baseline GSM8K >80%, switch to 1.5B-Instruct for more headroom.
2. **Day 3–5:** Optional short SFT cold-start on ~1–4k DeepSeek-R1 CoT traces; then GRPO with Dr.GRPO loss + binary+format reward on GSM8K (Unsloth + TRL + vLLM, QLoRA, ~1–1.5k steps).
3. **Day 6–8:** Tune stability (KL on/off, Clip-Higher, entropy bonus); add StrategyQA-style reasoning data; run multi-seed evals. **Threshold:** if entropy collapses or reward gets hacked, enable Clip-Cov/KL-Cov and difficulty filtering.
4. **Day 9–11:** Add the novelty layer — AdaptThink/Thinkless-style think/no-think gate (reward = correctness − λ·tokens) and/or a Weaver 400M verifier router; quantize to GGUF Q4_K_M; measure on-device latency.
5. **Day 12–14:** Ablations (1-shot RLVR flex, optional TTRL), Pareto + CI charts, and a deck with the Samsung Galaxy mapping. **Threshold:** if the router degrades accuracy below the +5% bar, ship it as an optional inference mode and keep the accuracy model as the headline.

## Caveats

- Many cited gains (DeepScaleR, TTRL, 1-shot RLVR, AdaptThink) are reported on math/competition benchmarks (AIME/MATH500), not GSM8K/StrategyQA directly — gains should transfer but must be verified empirically.
- Exact MMLU for DeepSeek-R1-Distill-Qwen-1.5B is weakly sourced (~35–45% inferred); pull the precise number from the Open LLM Leaderboard details dataset before relying on it.
- RL results are high-variance; single-seed numbers are unreliable — always report multiple seeds with confidence intervals.
- The Lavaee ablation is on a 0.5B model from a single blog; treat "DAPO tricks hurt small models" as a strong hypothesis to verify on your chosen base, not gospel.
- TTRL and any test-time adaptation on eval data risk being seen as benchmark gaming; disclose methodology clearly.
- Some supporting sources are secondary (Medium, Substack, Emergent Mind); primary arXiv / Hugging Face / lab-blog numbers are cited where possible and should be preferred in your final writeup.
