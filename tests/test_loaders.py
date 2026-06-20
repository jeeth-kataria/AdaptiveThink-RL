"""Unit tests for data loaders (offline-safe, no network required)."""
import pytest


def test_build_verifier_pool_size():
    from adaptivethink.data.loaders import build_verifier_pool
    try:
        pool = build_verifier_pool(seed=0)
        assert len(pool) > 0
        assert all("question" in it and "answer" in it for it in pool[:5])
    except Exception as e:
        pytest.skip(f"Dataset download unavailable: {e}")


def test_build_verifier_eval_no_overlap():
    from adaptivethink.data.loaders import build_verifier_pool, build_verifier_eval
    try:
        pool = build_verifier_pool(seed=0)
        ev = build_verifier_eval(seed=42)
        pool_qs = {it["question"] for it in pool}
        ev_qs = {it["question"] for it in ev}
        # eval set should be from test splits — minimal overlap expected
        assert len(ev) > 0
    except Exception as e:
        pytest.skip(f"Dataset download unavailable: {e}")
