"""Pure TTRL pseudo-label logic — no torch, so unit-testable anywhere."""
from collections import Counter

from adaptivethink.router.reward import extract_answer, _answers_match


def majority_vote_reward(completions, lambda_tok=1e-3, conf_weight=True):
    """Reward each completion by agreement with the group's majority answer.

    conf_weight scales the reward by the majority's vote share (confidence), so
    low-agreement groups (likely wrong pseudo-label) contribute weaker signal.
    """
    answers = [extract_answer(c) for c in completions]
    valid = [a for a in answers if a is not None]
    if not valid:
        return [0.0] * len(completions)

    pseudo, votes = Counter(valid).most_common(1)[0]
    scale = (votes / len(completions)) if conf_weight else 1.0

    rewards = []
    for c, a in zip(completions, answers):
        agree = float(a is not None and _answers_match(a, pseudo))
        length_pen = lambda_tok * len(c.split())
        rewards.append(scale * agree - length_pen)
    return rewards
