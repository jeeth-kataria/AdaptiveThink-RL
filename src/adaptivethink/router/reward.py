"""
Reward function for GRPO router training.

Novelty vs AdaptThink/CODA: difficulty `d` is from an *external* distilled
verifier, not internal model confidence or group-rollout pass-rate.

Lever 1+2 (answer extraction + answer matching) is the single biggest accuracy
lever: a CoT trace ends with the real answer, so we extract the LAST boxed /
labelled answer (not the first scratch value), and we compare answers with
boolean mapping + numeric equivalence (so '72'=='72.0', '1/2'=='0.5',
'\\frac{1}{2}'=='0.5', '$1,000'=='1000', 'yes'=='True'). stdlib only.
"""
import math
import re
from fractions import Fraction

# All \boxed{...} occurrences; group is one level of brace content. Used with
# findall so we can pick the LAST (closest to the final answer).
ANSWER_RE = re.compile(r"\\boxed\{([^{}]+)\}")

# Labelled final-answer keyword (preferred over a bare '='). We locate every
# keyword occurrence and capture the value after the LAST one, so a later
# 'final answer: X' beats an earlier 'answer is Y'.
_LABEL_KW_RE = re.compile(
    r"(?:final\s+answer\s*(?:is\s*)?[:=]?\s*|answer\s+is\s*|answer\s*:\s*)",
    re.IGNORECASE,
)
# Value captured after a label keyword: up to a newline, sentence end, or the
# start of another label keyword.
_LABEL_VAL_RE = re.compile(
    r"([^\n.]+?)(?=\s*(?:final\s+answer|answer\s+is|answer\s*:)|[\n.]|$)",
    re.IGNORECASE,
)
# Bare '=' fallback, only used when no labelled form exists. Stops at '(' so a
# trailing annotation like 'x = -5 (rounded)' yields '-5', not '-5 (rounded)'.
_EQ_RE = re.compile(r"=\s*([^\n.=(]+)")

# Boolean / StrategyQA synonyms. Single-letter 't'/'f' are intentionally excluded
# — they collide with variable names / option labels in real datasets.
_TRUE_WORDS = {"yes", "true", "correct"}
_FALSE_WORDS = {"no", "false", "incorrect"}

# \frac{a}{b} or \dfrac{a}{b}
_FRAC_RE = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")


def _extract_boxed_balanced(text: str) -> list[str]:
    """Return the content of every \\boxed{...}, brace-balanced.

    Falls back gracefully on the simple regex behaviour for the common
    (non-nested) case, but correctly handles nested braces like
    ``\\boxed{\\frac{1}{2}}`` which the flat regex would truncate.
    """
    results = []
    needle = "\\boxed{"
    i = 0
    while True:
        start = text.find(needle, i)
        if start == -1:
            break
        j = start + len(needle)
        depth = 1
        buf = []
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth > 0:
                    buf.append(ch)
            else:
                buf.append(ch)
            j += 1
        if depth == 0:
            results.append("".join(buf).strip())
            i = j
        else:
            # Unbalanced; stop scanning to avoid an infinite loop.
            break
    return results


def extract_answer(text: str) -> str | None:
    """Extract the model's final answer from a (possibly CoT) completion.

    Strategy:
      1. Prefer the LAST \\boxed{...} (closest to the final answer).
      2. Else the LAST labelled form ('final answer', 'answer is', 'answer:').
      3. Else, only if no labelled form exists, the LAST bare '=' match.
      4. Strip trailing punctuation/whitespace; None if nothing found.
    """
    if not text:
        return None

    boxed = _extract_boxed_balanced(text)
    if not boxed:
        # Fall back to the flat regex in case the balanced scan missed
        # something unusual (kept for safety / backwards compatibility).
        boxed = [m.strip() for m in ANSWER_RE.findall(text)]
    if boxed:
        return _strip_trailing(boxed[-1])

    label_kws = list(_LABEL_KW_RE.finditer(text))
    if label_kws:
        tail = text[label_kws[-1].end():]
        m = _LABEL_VAL_RE.match(tail)
        if m and m.group(1).strip():
            return _strip_trailing(m.group(1))

    eqs = _EQ_RE.findall(text)
    if eqs:
        return _strip_trailing(eqs[-1])

    return None


def _strip_trailing(s: str) -> str | None:
    """Strip surrounding whitespace and trailing sentence punctuation."""
    s = s.strip().rstrip(".,;:!? ").strip()
    return s if s else None


def _norm(s: str) -> str:
    """Lowercase + strip $, commas, spaces, LaTeX wrappers, a trailing dot."""
    s = s.strip().lower()
    # Strip surrounding \( \) and $ $ math delimiters.
    s = s.replace("\\(", "").replace("\\)", "")
    # Strip \text{...} / \mathrm{...} wrappers (keep inner content).
    s = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", s)
    # Strip thin-space / negative-space LaTeX spacing macros.
    s = s.replace("\\!", "").replace("\\,", "")
    # Remove $, commas, spaces.
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    # Strip a single trailing '.'.
    if s.endswith("."):
        s = s[:-1]
    return s


def _as_bool(s: str) -> str | None:
    """Map a normalised string to 'true'/'false' if it is a boolean synonym."""
    if s in _TRUE_WORDS:
        return "true"
    if s in _FALSE_WORDS:
        return "false"
    return None


def _to_number(s: str):
    """Parse a normalised string as a number, or return None.

    Handles plain int/float, fractions 'a/b', LaTeX \\frac{a}{b} / \\dfrac{a}{b},
    leading '+', scientific notation. Guarded so non-numeric input never throws.
    """
    if not s:
        return None
    # Handle a leading sign once, so '-\frac{1}{2}' and '-1/2' parse correctly.
    neg = False
    if s.startswith("-"):
        neg, s = True, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    # LaTeX fraction.
    m = _FRAC_RE.fullmatch(s)
    if m:
        try:
            v = Fraction(m.group(1)) / Fraction(m.group(2))
            return -v if neg else v
        except (ValueError, ZeroDivisionError):
            return None
    # Plain fraction 'a/b'.
    if "/" in s:
        try:
            v = Fraction(s)
            return -v if neg else v
        except (ValueError, ZeroDivisionError):
            return None
    # Plain int / float / scientific notation.
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _answers_match(pred: str, gt: str) -> bool:
    """Compare predicted vs ground-truth answers robustly.

    Order: normalise -> boolean mapping -> numeric equivalence -> string equality.
    Every parse is guarded so a non-numeric answer never throws.
    """
    if pred is None or gt is None:
        return False

    np_, ng = _norm(pred), _norm(gt)

    # Boolean / StrategyQA mapping: only when BOTH sides are booleans.
    bp, bg = _as_bool(np_), _as_bool(ng)
    if bp is not None and bg is not None:
        return bp == bg

    # Percentages: strip a trailing '%' on both sides before numeric compare
    # (so '50%' vs '50' compares as numbers; we do NOT divide by 100).
    xp = np_[:-1] if np_.endswith("%") else np_
    xg = ng[:-1] if ng.endswith("%") else ng

    # Strip a single leading '+' for numeric parsing.
    num_p = _to_number(xp.lstrip("+") if xp.startswith("+") else xp)
    num_g = _to_number(xg.lstrip("+") if xg.startswith("+") else xg)
    if num_p is not None and num_g is not None:
        try:
            fp, fg = float(num_p), float(num_g)
            if math.isfinite(fp) and math.isfinite(fg):
                # Exact compare for integer-valued answers (avoids large-int
                # off-by-one false positives); tolerance only for non-integers.
                if fp == int(fp) and fg == int(fg):
                    return int(fp) == int(fg)
                return abs(fp - fg) <= 1e-6 * max(1.0, abs(fp), abs(fg))
            # inf / nan: let normalised string equality decide below.
        except (ValueError, OverflowError):
            pass  # fall through to string compare

    # Fallback: exact normalised string equality.
    return np_ == ng


def decision_from_response(response: str) -> str | None:
    s = response.strip()
    if s.startswith("<think>"):
        return "think"
    if s.startswith("<no_think>"):
        return "no_think"
    return None


def compute_rewards(
    responses: list[str],
    ground_truths: list[str],
    difficulties: list[float],
    token_counts: list[int] | None = None,
    lambda_tok: float = 3e-3,   # 5e-4 was too weak; 3e-3 penalises 300-token easy responses by ~0.9
    lambda_obey: float = 0.05,  # small enough that wrong+honoured stays negative
) -> list[float]:
    """
    Returns one scalar reward per response.
    token_counts: actual output token counts from the model (preferred over word count).
    """
    rewards = []
    for i, (resp, gt, d) in enumerate(zip(responses, ground_truths, difficulties)):
        pred = extract_answer(resp)
        correct = float(pred is not None and _answers_match(pred, gt))

        # Use actual token count if provided, else word count as proxy
        n_tok = token_counts[i] if token_counts else len(resp.split())
        length_penalty = lambda_tok * n_tok * (1.0 - float(d))

        decision = decision_from_response(resp)
        if decision == "think":
            honoured = float("</think>" in resp)
        elif decision == "no_think":
            honoured = float("</think>" not in resp)
        else:
            honoured = 0.0

        rewards.append(correct - length_penalty + lambda_obey * honoured * correct)
    return rewards
