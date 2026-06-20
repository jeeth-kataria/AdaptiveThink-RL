"""RLVR reward functions for Dr.GRPO training (TRL GRPOTrainer).

Design (per ALGO + DATA research):
  * Correctness reward = BINARY exact-match (RLVR). We REUSE the repo's already
    robust answer logic — ``adaptivethink.router.reward.extract_answer`` and
    ``_answers_match`` (numeric / fraction / boolean / LaTeX tolerant). We do NOT
    reimplement matching. Correctness dominates (weight ~1.0).
  * Format reward = small nudge for a well-formed, properly-ordered
    ``<think>...</think><answer>...</answer>`` structure (weight ~0.1-0.2). Kept
    small so it can never be gamed over correctness.

TRL reward-fn contract (trl 0.24.0):
    def reward_func(completions, **kwargs) -> list[float]
The trainer passes ``prompts``, ``completions``, ``completion_ids``,
``trainer_state`` and EVERY non-'prompt' dataset column as kwargs (so the gold
label arrives as a kwarg). Completions are either:
  * standard datasets  -> list[str]
  * conversational      -> list[list[{"role","content"}]]  (use [0]["content"])
Return a plain python ``list[float]`` of length == len(completions). Per-item
``None`` is allowed (skips that item); we never return None here.

This module is dependency-free (stdlib + the repo's stdlib-only reward module),
so it imports cleanly without torch/trl/datasets.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Sequence

# Reuse the repo's robust answer logic — do NOT reimplement (interface contract).
from adaptivethink.router.reward import _answers_match, extract_answer

# The <answer>…</answer> payload. DOTALL so multi-line answers are captured.
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Strict, ordered, single-block structure: <think>…</think> then <answer>…</answer>.
# Anchored so trailing/leading junk fails the format check (small reward only).
_THINK_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL
)


# ── completion text extraction ────────────────────────────────────────────────

def _completion_text(c: Any) -> str:
    """Return the assistant text from a TRL completion.

    Handles both standard (str) and conversational (list[{role,content}])
    completion shapes, plus defensive fallbacks for dicts.
    """
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        if not c:
            return ""
        last = c[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    if isinstance(c, dict):
        return str(c.get("content", ""))
    return str(c)


def predicted_answer(text: str) -> str | None:
    """Return the model's predicted final answer string (or None).

    Two cases:
      * The trained ``<answer>…</answer>`` tag is present -> the answer IS the tag
        content (a bare value like '4' or 'yes'), so we return it directly. We do
        NOT route this through ``extract_answer``, which is built to dig a final
        answer out of a CoT trace and returns None for a bare token.
      * No tag -> fall back to the repo's ``extract_answer`` over the full
        completion so a \\boxed{}/labelled/'=' final answer can still be
        recovered. Correctness should not hinge on the wrapper being present.
    """
    m = _ANSWER_TAG_RE.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
        # Empty <answer></answer> — try to recover from the trace.
    return extract_answer(text)


# ── ground-truth resolution from TRL kwargs ───────────────────────────────────

# Accepted column names for the gold label, in priority order. The repo's data.py
# emits "answer"; the research's example uses "ground_truth"; we accept both.
_GT_KEYS = ("answer", "ground_truth", "gold", "solution", "label")


def _resolve_ground_truth(n: int, kwargs: dict) -> list[Any] | None:
    """Find the per-item gold labels among the dataset columns passed as kwargs."""
    for key in _GT_KEYS:
        val = kwargs.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)) and len(val) == n:
            return list(val)
        # A scalar broadcast (rare) — replicate.
        if not isinstance(val, (list, tuple)):
            return [val] * n
    return None


# ── reward functions (TRL signature) ──────────────────────────────────────────

def correctness_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Binary exact-match RLVR reward: 1.0 if the parsed answer matches gold.

    Gold labels are read from kwargs (``answer``/``ground_truth``/…). The
    prediction is the ``<answer>`` tag content when present, else the repo's
    ``extract_answer`` over the full completion. Matching is delegated to the
    repo's tolerant ``_answers_match`` (numeric/fraction/boolean aware).
    """
    texts = [_completion_text(c) for c in completions]
    gts = _resolve_ground_truth(len(texts), kwargs)
    if gts is None:
        # No gold available — cannot verify; contribute zero signal.
        return [0.0 for _ in texts]

    out: list[float] = []
    for text, gt in zip(texts, gts):
        if gt is None:
            out.append(0.0)
            continue
        pred = predicted_answer(text)
        ok = pred is not None and _answers_match(pred, str(gt))
        out.append(1.0 if ok else 0.0)
    return out


def format_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Small format reward: 1.0 for a well-formed, ordered <think>/<answer> block.

    Returns the *unweighted* indicator (0.0/1.0); the small weight is applied by
    GRPOConfig.reward_weights (e.g. 0.2). Keeping the magnitude at the call site
    rather than baking 0.1 in here lets the trainer's logged per-func reward stay
    interpretable.
    """
    out: list[float] = []
    for c in completions:
        text = _completion_text(c).strip()
        out.append(1.0 if _THINK_ANSWER_RE.match(text) else 0.0)
    return out


def combined_reward(
    completions: Sequence[Any],
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
    **kwargs: Any,
) -> list[float]:
    """Single summed reward = w_c * correctness + w_f * format.

    Provided for callers/harnesses that want ONE reward function instead of TRL's
    list-of-funcs + reward_weights mechanism. The Dr.GRPO trainer here uses the
    separate-functions path by default (so TRL logs each component), but this is a
    convenient, equivalent alternative for ad-hoc evaluation.
    """
    corr = correctness_reward(completions, **kwargs)
    fmt = format_reward(completions, **kwargs)
    return [correctness_weight * c + format_weight * f for c, f in zip(corr, fmt)]


def make_reward_funcs(use_format: bool = True) -> list[Callable[..., list[float]]]:
    """Return the list of reward funcs for GRPOTrainer(reward_funcs=...).

    Order matches GRPOConfig.reward_weights ([1.0, format_weight]). When
    ``use_format`` is False, only the correctness reward is returned (and
    reward_weights should then be [1.0]).
    """
    if use_format:
        return [correctness_reward, format_reward]
    return [correctness_reward]
