"""AdaptiveThink — on-device System-1.5 SLM reasoning with an RL-trained router.

Re-exports are lazy (PEP 562) so importing a light submodule (e.g. the pure
reward/metric logic) does not pull in heavy deps such as `datasets` or `torch`.
"""

__all__ = ["build_verifier_pool", "build_verifier_eval", "compute_rewards", "make_prompt"]


def __getattr__(name):
    if name in ("build_verifier_pool", "build_verifier_eval"):
        from adaptivethink.data import loaders
        return getattr(loaders, name)
    if name == "compute_rewards":
        from adaptivethink.router.reward import compute_rewards
        return compute_rewards
    if name == "make_prompt":
        from adaptivethink.router.prompt import make_prompt
        return make_prompt
    raise AttributeError(f"module 'adaptivethink' has no attribute {name!r}")
