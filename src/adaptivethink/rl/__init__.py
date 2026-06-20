"""adaptivethink.rl — Dr.GRPO RLVR training for small dense reasoning models.

Strategy 1: a short SFT cold-start (installs the <think>/<answer> format) followed
by Dr.GRPO RL with a rule-based, verifiable reward (RLVR). The full Dr.GRPO bias
fix (constant length-normalization + no per-group std division) is configured via
TRL's GRPOConfig (loss_type='dr_grpo', scale_rewards=False), with the
small-model-safe DAPO trick (mask_truncated_completions=True) and no KL (beta=0.0).

Top-level imports are intentionally LIGHT: the heavy stack (torch, trl, datasets,
unsloth, transformers) is imported lazily inside functions so that
``import adaptivethink.rl`` (and `python -m py_compile`) succeeds in an
environment without those packages installed.

Public entry points:
  * ``python -m adaptivethink.rl.drgrpo_train``   — Dr.GRPO RL training
  * ``python -m adaptivethink.rl.sft_coldstart``  — SFT cold-start (owned elsewhere)

Re-exports (PEP 562 lazy) of the pure, dependency-free helpers:
  * ``correctness_reward`` / ``format_reward`` / ``combined_reward`` from rewards
  * ``build_dataset`` / ``parse_datasets`` from data
"""

__all__ = [
    "correctness_reward",
    "format_reward",
    "combined_reward",
    "make_reward_funcs",
    "build_dataset",
    "parse_datasets",
    "SYSTEM_PROMPT",
]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name in ("correctness_reward", "format_reward", "combined_reward",
                "make_reward_funcs"):
        from adaptivethink.rl import rewards
        return getattr(rewards, name)
    if name in ("build_dataset", "parse_datasets", "SYSTEM_PROMPT"):
        from adaptivethink.rl import data
        return getattr(data, name)
    raise AttributeError(f"module 'adaptivethink.rl' has no attribute {name!r}")
