"""Unit tests for reward function — run before any training: pytest tests/test_reward.py"""
import pytest
from adaptivethink.router.reward import compute_rewards, extract_answer, decision_from_response, _answers_match


def test_extract_boxed():
    assert extract_answer("so \\boxed{42} is correct") == "42"
    assert extract_answer("no answer here") is None


def test_extract_plain_fallback():
    assert extract_answer("The answer is 42.") == "42"


def test_answers_match_normalise():
    assert _answers_match("$42$", "42")
    assert _answers_match("1,000", "1000")
    assert not _answers_match("42", "43")


def test_answers_match_numeric_int_float():
    assert _answers_match("72", "72.0")
    assert _answers_match("72.0", "72")


def test_answers_match_fraction_vs_decimal():
    assert _answers_match("1/2", "0.5")
    assert _answers_match("0.5", "1/2")


def test_answers_match_latex_frac():
    assert _answers_match("\\frac{1}{2}", "0.5")
    assert _answers_match("\\dfrac{1}{2}", "0.5")
    assert _answers_match("\\frac{3}{4}", "0.75")


def test_answers_match_currency_comma():
    assert _answers_match("$1,000", "1000")
    assert _answers_match("$1,000.00", "1000")


def test_answers_match_boolean_yes_true():
    assert _answers_match("yes", "True")
    assert _answers_match("Yes", "true")
    assert _answers_match("correct", "True")


def test_answers_match_boolean_no_false():
    assert _answers_match("no", "False")
    assert _answers_match("No", "false")
    assert _answers_match("incorrect", "False")


def test_answers_match_boolean_mismatch():
    assert not _answers_match("yes", "False")
    assert not _answers_match("no", "True")


def test_answers_match_leading_plus_and_sci():
    assert _answers_match("+5", "5")
    assert _answers_match("1e3", "1000")


def test_answers_match_percentage_strip():
    # Both stripped of '%' then compared as numbers (no /100 division).
    assert _answers_match("50%", "50")
    assert _answers_match("50%", "50%")


def test_answers_match_non_numeric_string():
    assert _answers_match("Paris", "paris")
    assert not _answers_match("Paris", "London")


def test_answers_match_guards_no_throw():
    # Weird input must never raise — falls through to string compare.
    assert not _answers_match("1/0", "5")
    assert _answers_match("\\frac{1}{0}", "\\frac{1}{0}")


def test_extract_last_of_multiple_boxed():
    # In a CoT trace the first boxed is a scratch value; the last is the answer.
    text = "scratch \\boxed{7} then more work \\boxed{42}"
    assert extract_answer(text) == "42"


def test_extract_boxed_nested_braces():
    assert extract_answer("final \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_intermediate_eq_not_chosen():
    # An intermediate equation must NOT win over a later labelled answer.
    text = "We compute 2+2=4 along the way. The answer is 42"
    assert extract_answer(text) == "42"


def test_extract_last_label_wins():
    text = "answer is 7 first, but final answer: 42"
    assert extract_answer(text) == "42"


def test_extract_bare_eq_only_when_no_label():
    # No labelled form -> use the LAST '=' match.
    text = "x = 1 then y = 2 then z = 99"
    assert extract_answer(text) == "99"


def test_extract_strips_trailing_punct():
    assert extract_answer("The answer is 42.") == "42"
    assert extract_answer("answer: 42!") == "42"


# --- Regressions caught by adversarial review of the lever-1/2 diff ---

def test_extract_final_answer_is_phrase():
    # 'final answer is X' must yield 'X', not 'is X'.
    assert extract_answer("The final answer is 42.") == "42"
    assert extract_answer("So the final answer is Paris.") == "Paris"


def test_answers_match_negative_latex_fraction():
    assert _answers_match("-\\frac{1}{2}", "-0.5")
    assert _answers_match("-\\dfrac{1}{2}", "-0.5")
    assert _answers_match("-1/2", "-0.5")
    assert not _answers_match("-\\frac{1}{2}", "0.5")


def test_answers_match_single_letter_bool_no_collision():
    # 't'/'f' are NOT boolean synonyms (they collide with real answers/labels)…
    assert not _answers_match("no", "f")
    assert not _answers_match("yes", "t")
    # …but genuine word synonyms still match.
    assert _answers_match("incorrect", "no")
    assert _answers_match("correct", "yes")


def test_answers_match_inf_nan_self_equal():
    assert _answers_match("inf", "inf")
    assert _answers_match("nan", "nan")


def test_answers_match_large_int_off_by_one():
    # Relative tolerance must not call consecutive large integers equal.
    assert not _answers_match("999999", "1000000")
    assert not _answers_match("1000001", "1000002")
    assert _answers_match("1000000", "1000000")


def test_extract_bare_eq_strips_parenthetical():
    assert extract_answer("x = -5 (rounded)") == "-5"


def test_decision():
    assert decision_from_response("<think> reasoning...") == "think"
    assert decision_from_response("<no_think> 42") == "no_think"
    assert decision_from_response("random") is None


def test_correct_short_easy():
    r = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.1])[0]
    assert r > 0.5, f"got {r}"


def test_correct_long_easy_penalised():
    long_resp = "<think> " + "word " * 300 + "</think> \\boxed{42}"
    r = compute_rewards([long_resp], ["42"], [0.1])[0]
    assert r < 0.5, f"long easy should be penalised, got {r}"


def test_correct_long_hard_not_penalised():
    long_resp = "<think> " + "word " * 300 + "</think> \\boxed{42}"
    r = compute_rewards([long_resp], ["42"], [0.95])[0]
    assert r > 0.5, f"long hard should not be penalised, got {r}"


def test_wrong_answer_negative():
    r = compute_rewards(["<no_think> \\boxed{99}"], ["42"], [0.5])[0]
    assert r < 0, f"wrong answer should give negative reward, got {r}"


def test_missing_routing_token_penalised():
    r_no_token = compute_rewards(["\\boxed{42}"], ["42"], [0.5])[0]
    r_with_token = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.5])[0]
    assert r_no_token < r_with_token


def test_token_counts_override():
    # When actual token counts provided, use them instead of word count
    r_word = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.1])[0]
    r_tok  = compute_rewards(["<no_think> \\boxed{42}"], ["42"], [0.1], token_counts=[500])[0]
    assert r_tok < r_word  # more tokens → more penalty
