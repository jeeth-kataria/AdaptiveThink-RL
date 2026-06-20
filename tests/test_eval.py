"""Unit tests for eval + TTRL pure logic (no model load): pytest tests/test_eval.py"""
import sys
from pathlib import Path

from adaptivethink.metrics import pass_at_k, is_correct as _is_correct
from adaptivethink.ttrl.vote import majority_vote_reward

# eval/ is a script dir, not a package — add it so the import-light helpers
# (no torch/datasets at module scope) are reachable. load_benchmark is imported
# lazily inside eval_benchmark, so importing run_benchmarks stays dependency-free.
_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from run_benchmarks import valid_k_list, parse_seeds  # noqa: E402
from plots import kpi_table  # noqa: E402


def test_pass_at_k_all_correct():
    assert pass_at_k(n=8, c=8, k=1) == 1.0
    assert pass_at_k(n=8, c=8, k=8) == 1.0


def test_pass_at_k_none_correct():
    assert pass_at_k(n=8, c=0, k=1) == 0.0


def test_pass_at_k_monotonic_in_k():
    # more attempts -> higher chance of at least one correct
    assert pass_at_k(8, 2, 8) >= pass_at_k(8, 2, 1)


def test_pass_at_k_fewer_wrong_than_k():
    # if n-c < k it's guaranteed a hit
    assert pass_at_k(n=4, c=2, k=3) == 1.0


def test_is_correct_boxed():
    assert _is_correct("blah \\boxed{42}", "42")
    assert not _is_correct("blah \\boxed{7}", "42")


def test_majority_vote_reward_rewards_consensus():
    comps = ["\\boxed{42}", "\\boxed{42}", "\\boxed{42}", "\\boxed{7}"]
    r = majority_vote_reward(comps, lambda_tok=0.0)
    # the three agreeing with majority (42) score higher than the dissenter
    assert r[0] > r[3]
    assert r[0] == r[1] == r[2]


def test_majority_vote_confidence_scaling():
    high_conf = majority_vote_reward(["\\boxed{1}"] * 4, lambda_tok=0.0)
    low_conf = majority_vote_reward(
        ["\\boxed{1}", "\\boxed{1}", "\\boxed{2}", "\\boxed{3}"], lambda_tok=0.0)
    # full agreement -> reward 1.0; split -> scaled down
    assert high_conf[0] == 1.0
    assert low_conf[0] < 1.0


def test_majority_vote_no_valid_answers():
    assert majority_vote_reward(["no answer", "nope"]) == [0.0, 0.0]


# --- Pass@k k>n_samples reporting-integrity guard ---

def test_valid_k_list_drops_k_above_n_samples():
    kept, dropped = valid_k_list([1, 8, 64], n_samples=8)
    assert kept == [1, 8]
    assert dropped == [64]


def test_valid_k_list_dedups_and_drops_nonpositive():
    # Non-positive k are surfaced in `dropped` (not silently swallowed) so a
    # caller learns they were invalid.
    kept, dropped = valid_k_list([1, 1, 8, 0, -3, 64, 128], n_samples=8)
    assert kept == [1, 8]
    assert dropped == [-3, 0, 64, 128]


def test_valid_k_list_all_valid_drops_nothing():
    kept, dropped = valid_k_list([1, 4, 8], n_samples=8)
    assert kept == [1, 4, 8]
    assert dropped == []


def test_valid_k_list_prevents_spurious_pass_at_k():
    # The hazard: Pass@64 on 8 samples hits the n-c<k branch and reports 1.0.
    n_samples, c = 8, 0  # zero correct -> Pass@anything should be 0
    assert pass_at_k(n_samples, c, 64) == 1.0  # the buggy/spurious value
    kept, _ = valid_k_list([64], n_samples)
    # guard removes it, so the harness never reports the spurious 100%
    assert 64 not in kept


# --- Seed parsing (honest Pass@1 averaging) ---

def test_parse_seeds_defaults_to_single_seed():
    assert parse_seeds("", 0) == [0]
    assert parse_seeds("  ", 5) == [5]


def test_parse_seeds_comma_list():
    assert parse_seeds("0,1,2", 0) == [0, 1, 2]


# --- KPI table denominator + router selection ---

def _bench(p1):
    return {"pass@1": p1}


def test_kpi_table_denominator_full_three():
    baseline = {"benchmarks": {k: _bench(0.1) for k in ("gsm8k", "mmlu", "strategyqa")}}
    router = {"tag": "router", "benchmarks": {
        "gsm8k": _bench(0.60), "mmlu": _bench(0.50), "strategyqa": _bench(0.10)}}
    out = kpi_table(baseline, [router])
    # 2 of 3 KPIs met -> denominator 2, PASS
    assert "2/2" in out
    assert "(PASS)" in out


def test_kpi_table_denominator_subset_present():
    # Only gsm8k comparable in both -> denominator must not assume 3
    baseline = {"benchmarks": {"gsm8k": _bench(0.10)}}
    router = {"tag": "router", "benchmarks": {"gsm8k": _bench(0.60)}}
    out = kpi_table(baseline, [router])
    assert "1/1" in out
    assert "(PASS)" in out


def test_kpi_table_warns_when_no_router_tag(capsys):
    baseline = {"benchmarks": {"gsm8k": _bench(0.10)}}
    untagged = {"tag": "always_think", "benchmarks": {"gsm8k": _bench(0.60)}}
    kpi_table(baseline, [untagged])
    assert "no run tagged 'router'" in capsys.readouterr().out


def test_kpi_table_two_present_one_met_is_not_pass():
    # Regression: with exactly 2 KPI benchmarks comparable and only 1 met, the
    # verdict must be NOT YET (required=ceil(2*2/3)=2), not a false PASS.
    baseline = {"benchmarks": {"gsm8k": _bench(0.50), "mmlu": _bench(0.45)}}
    router = {"tag": "router", "benchmarks": {
        "gsm8k": _bench(0.56), "mmlu": _bench(0.45)}}  # only gsm8k meets delta+target
    out = kpi_table(baseline, [router])
    assert "(PASS)" not in out
    assert "1/2" in out
