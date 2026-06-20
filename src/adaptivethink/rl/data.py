"""Dataset loading + prep for Dr.GRPO RLVR training.

Maps GSM8K and StrategyQA (WITH gold labels) to a uniform schema:

    {"prompt": <str>, "answer": <gold str>, "dataset": <name>, "question": <str>}

Dataset ids / fields are taken VERBATIM from the DATA research:
  * GSM8K       : openai/gsm8k, config "main". Fields: question (str),
                  answer (CoT ending with '#### <number>'). Gold = post-'####'
                  number with thousands-commas stripped.
  * StrategyQA  : ChilleD/StrategyQA (primary; real train=1600 / test=687 splits,
                  both labeled). Fields: question (str), answer (bool). Gold mapped
                  to "yes"/"no". Fallback: wics/strategy-qa (single labeled "test"
                  split of 2290, self-split to avoid leakage).

The prompt reuses the repo's <think>/<answer> RLVR convention. We keep the SAME
template for training and (downstream) eval so the before/after delta is not
confounded (DATA research Meta Advice).

Difficulty filter (DATA research (d) / ALGO research): TRL has NO in-trainer
dynamic sampling, so difficulty filtering is an OFFLINE dataset preprocessing
step. We sample the FROZEN base model K times per item, compute the empirical
pass-rate p_hat with the repo's verifier, and KEEP items with 0 < p_hat < 1
(drop unsolvable p_hat==0 and trivial p_hat==1 — both give zero-advantage groups).
The expensive generation is injected as a callable so this module stays light and
testable; ``build_dataset`` returns a HF Dataset only when ``datasets`` is present.

Top-level imports are stdlib only; ``datasets`` is imported lazily.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional, Sequence

SUPPORTED_DATASETS = ("gsm8k", "strategyqa")

# HF dataset ids (DATA research concrete_spec).
GSM8K_ID = "openai/gsm8k"
GSM8K_CONFIG = "main"
STRATEGYQA_PRIMARY_ID = "ChilleD/StrategyQA"
STRATEGYQA_FALLBACK_ID = "wics/strategy-qa"

# RLVR instruction: chain-of-thought in <think>, final answer in <answer>.
SYSTEM_PROMPT = (
    "Reason step by step inside <think> and </think>, "
    "then give the final answer inside <answer> and </answer>."
)

# GSM8K gold extraction: the number after the '####' marker, commas stripped.
_GSM8K_GOLD_RE = re.compile(r"####\s*([\-0-9.,]+)")


# ── prompt construction ───────────────────────────────────────────────────────
#
# We reuse the repo's chat-template STYLE (Qwen ChatML: <|im_start|>role ...
# <|im_end|> with a trailing assistant turn — same shape as
# adaptivethink.router.prompt.make_prompt) but with the RLVR <think>/<answer>
# system prompt instead of the router's routing-token/\boxed{} SYSTEM, which
# conflicts with the format we are training. Keeping the ChatML shape identical
# means the special tokens still match the Qwen tokenizer.

def _build_prompt(question: str, instruction: str) -> str:
    """Build a Qwen-ChatML prompt with the RLVR system message + the question.

    Mirrors the repo's make_prompt template shape so tokenizer special tokens
    line up, but uses the RLVR system prompt. TRL's GRPOTrainer accepts a
    'prompt' column of plain strings.
    """
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── per-dataset mappers ───────────────────────────────────────────────────────

def gsm8k_to_row(example: dict, instruction: str = SYSTEM_PROMPT) -> dict:
    """Map an openai/gsm8k row to the uniform RLVR schema."""
    question = example["question"]
    raw = example["answer"]
    m = _GSM8K_GOLD_RE.search(raw)
    gold = m.group(1).replace(",", "").strip() if m else raw.split("####")[-1].strip()
    return {
        "prompt": _build_prompt(question, instruction),
        "answer": gold,
        "dataset": "gsm8k",
        "question": question,
    }


def strategyqa_to_row(example: dict, instruction: str = SYSTEM_PROMPT) -> dict:
    """Map a StrategyQA row (bool gold) to the uniform schema (yes/no gold)."""
    question = example["question"]
    ans = example["answer"]
    # ChilleD/StrategyQA & wics/strategy-qa expose a python bool; be defensive.
    if isinstance(ans, str):
        yes = ans.strip().lower() in ("true", "yes", "1")
    else:
        yes = bool(ans)
    instr = f"{instruction}\nAnswer yes or no."
    return {
        "prompt": _build_prompt(question, instr),
        "answer": "yes" if yes else "no",
        "dataset": "strategyqa",
        "question": question,
    }


# ── --datasets parsing ────────────────────────────────────────────────────────

def parse_datasets(spec: str) -> list[str]:
    """Parse a '--datasets gsm8k,strategyqa' spec into a validated name list."""
    names = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError("--datasets is empty; expected e.g. 'gsm8k,strategyqa'")
    unknown = [n for n in names if n not in SUPPORTED_DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s) {unknown}; supported: {list(SUPPORTED_DATASETS)}"
        )
    # De-dup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# ── single-dataset loaders (return list[dict] rows) ───────────────────────────

def _load_gsm8k_rows(split: str, instruction: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(GSM8K_ID, GSM8K_CONFIG, split=split)
    return [gsm8k_to_row(r, instruction) for r in ds]


def _load_strategyqa_rows(split: str, instruction: str) -> list[dict]:
    from datasets import load_dataset

    # Primary mirror has real train/test splits with labels; fall back to the
    # single labeled "test" split of wics/strategy-qa if the primary is absent.
    try:
        ds = load_dataset(STRATEGYQA_PRIMARY_ID, split=split)
    except Exception:  # noqa: BLE001 — any load/availability error -> fallback
        # wics/strategy-qa only ships a "test" split (the labeled dev set).
        ds = load_dataset(STRATEGYQA_FALLBACK_ID, split="test")
    return [strategyqa_to_row(r, instruction) for r in ds]


def load_rows(
    names: Sequence[str],
    split: str = "train",
    instruction: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Load + map the requested datasets into a single list of uniform rows."""
    rows: list[dict] = []
    for name in names:
        if name == "gsm8k":
            rows.extend(_load_gsm8k_rows(split, instruction))
        elif name == "strategyqa":
            rows.extend(_load_strategyqa_rows(split, instruction))
        else:  # pragma: no cover — parse_datasets already validated
            raise ValueError(f"Unsupported dataset: {name}")
    return rows


# ── difficulty filter (offline, base-model pass-rate) ─────────────────────────

# A base-generate callable: (prompt:str, k:int) -> list[str] of k sampled
# completions from the FROZEN base model. Injected by the trainer so this module
# carries no model deps.
BaseGenerateFn = Callable[[str, int], Sequence[str]]
# A verify callable: (completion_text:str, gold:str) -> bool.
VerifyFn = Callable[[str, str], bool]


def pass_rate(
    prompt: str,
    gold: str,
    base_generate: BaseGenerateFn,
    verify: VerifyFn,
    k: int = 8,
) -> float:
    """Empirical pass-rate p_hat = (#correct) / k for one item."""
    if k <= 0:
        raise ValueError("k must be >= 1 for the difficulty filter")
    samples = base_generate(prompt, k)
    n_ok = sum(1 for s in samples if verify(s, gold))
    return n_ok / float(k)


def keep_item(p_hat: float, drop_trivial: bool = True) -> bool:
    """Keep items in the learnable band; drop unsolvable (and optionally trivial).

    GRPO advantage is zero when every rollout in a group gets the same reward, so
    p_hat==0 (all-wrong) and p_hat==1 (all-right) groups carry no gradient. We
    always drop p_hat==0; ``drop_trivial`` also drops p_hat==1 (DAPO dynamic
    sampling logic, applied offline).
    """
    if drop_trivial:
        return 0.0 < p_hat < 1.0
    return p_hat > 0.0


def difficulty_filter_rows(
    rows: Sequence[dict],
    base_generate: BaseGenerateFn,
    verify: VerifyFn,
    k: int = 8,
    drop_trivial: bool = True,
) -> list[dict]:
    """Return rows whose base-model pass-rate is in the learnable band.

    This is the offline difficulty-filter hook. The trainer wires
    ``base_generate`` (frozen base sampling) and ``verify`` (the repo's reward
    matcher) and calls this BEFORE constructing the HF Dataset. Monitor the
    trainer's logged ``frac_reward_zero_std`` to confirm few dead groups remain.
    """
    kept: list[dict] = []
    for row in rows:
        p = pass_rate(row["prompt"], row["answer"], base_generate, verify, k=k)
        if keep_item(p, drop_trivial=drop_trivial):
            kept.append({**row, "p_solve": p})
    return kept


# ── top-level builder ─────────────────────────────────────────────────────────

def _shuffle_and_subset(
    rows: list[dict], seed: int, one_shot: bool, max_items: Optional[int]
) -> list[dict]:
    """Shuffle deterministically, then optionally take a tiny --one-shot subset.

    ``one_shot`` keeps a single item PER dataset (a smoke-sized subset for fast
    end-to-end pipeline checks); ``max_items`` caps total rows if set.
    """
    import random

    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    if one_shot:
        seen: set[str] = set()
        one: list[dict] = []
        for r in shuffled:
            ds = r.get("dataset", "?")
            if ds not in seen:
                seen.add(ds)
                one.append(r)
        return one

    if max_items is not None and max_items >= 0:
        return shuffled[:max_items]
    return shuffled


def build_dataset(
    names: Sequence[str],
    split: str = "train",
    seed: int = 0,
    one_shot: bool = False,
    max_items: Optional[int] = None,
    instruction: str = SYSTEM_PROMPT,
    base_generate: Optional[BaseGenerateFn] = None,
    verify: Optional[VerifyFn] = None,
    difficulty_k: int = 8,
    drop_trivial: bool = True,
):
    """Build the training Dataset.

    Steps: load+map -> (optional) offline difficulty filter -> shuffle/subset ->
    HF Dataset. When ``base_generate``+``verify`` are provided the difficulty
    filter runs; otherwise it is skipped (the filter is opt-in via the trainer's
    --difficulty-filter flag).

    Returns a ``datasets.Dataset`` (imported lazily). The returned dataset always
    has the columns: prompt, answer, dataset, question (+ p_solve if filtered).
    """
    from datasets import Dataset

    rows = load_rows(names, split=split, instruction=instruction)

    if base_generate is not None and verify is not None:
        rows = difficulty_filter_rows(
            rows, base_generate, verify, k=difficulty_k, drop_trivial=drop_trivial
        )

    rows = _shuffle_and_subset(rows, seed=seed, one_shot=one_shot, max_items=max_items)
    if not rows:
        raise ValueError(
            "build_dataset produced 0 rows — check --datasets/--split/filter settings"
        )
    return Dataset.from_list(rows)
