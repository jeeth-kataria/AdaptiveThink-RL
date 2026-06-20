"""Pure evaluation metrics — no heavy deps, so unit-testable anywhere.

Kept separate from eval/run_benchmarks.py (which imports `datasets`) so the
estimator and correctness check can be tested without the ML stack.
"""
import math

from adaptivethink.router.reward import extract_answer, _answers_match


def is_correct(completion: str, gt: str) -> bool:
    pred = extract_answer(completion)
    return pred is not None and _answers_match(pred, gt)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator (Chen et al. 2021). n samples, c correct."""
    if n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
