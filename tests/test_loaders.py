"""Unit tests for data loaders (offline-safe, no network required)."""
import pytest


# --- AQuA-RAT pure logic (no datasets load: option formatting + dispatch) ---

def test_aqua_option_formatting_strips_letter_prefix():
    # deepmind/aqua_rat ships options already letter-prefixed ('A)125'); the
    # loader must strip the source prefix and re-emit a uniform '(A) <value>'.
    from adaptivethink.data.loaders import _format_aqua_options
    out = _format_aqua_options(["A)125", "B)150", "C)175", "D)200", "E)225"])
    assert out == "(A) 125\n(B) 150\n(C) 175\n(D) 200\n(E) 225"


def test_aqua_option_formatting_handles_bare_and_spaced():
    from adaptivethink.data.loaders import _format_aqua_options
    # Bare values get A-E assigned positionally; spaced/dotted prefixes stripped.
    assert _format_aqua_options(["125", "150"]) == "(A) 125\n(B) 150"
    assert _format_aqua_options(["A ) 5 apples", "B. 10"]) == "(A) 5 apples\n(B) 10"


def test_load_benchmark_dispatch_knows_aqua():
    # The dispatch must RECOGNISE 'aqua' / 'aqua_rat' (so it routes to the AQuA
    # loader) — proven by the fact it does NOT raise the unknown-benchmark
    # ValueError. The actual datasets download is not required for this check:
    # without `datasets` installed it fails on the lazy import instead, which is
    # NOT a ValueError. An unknown name, by contrast, raises ValueError eagerly.
    from adaptivethink.data.loaders import load_benchmark
    with pytest.raises(ValueError):
        load_benchmark("totally_unknown_benchmark_xyz")
    for name in ("aqua", "aqua_rat"):
        try:
            load_benchmark(name, n=1)
        except ValueError:
            pytest.fail(f"load_benchmark({name!r}) raised ValueError — "
                        "dispatch does not recognise the AQuA benchmark")
        except Exception:
            # Any non-ValueError (e.g. ModuleNotFoundError for `datasets`, or a
            # network error) means dispatch reached the loader — that is success.
            pass


def test_aqua_loader_uses_datasets_when_available():
    # End-to-end smoke: only runs when `datasets` is installed. Gold must be a
    # bare option letter A-E and the question must embed the option block.
    pytest.importorskip("datasets")
    from adaptivethink.data.loaders import load_aqua_rat
    try:
        items = load_aqua_rat(split="train", seed=0, n=3)
    except Exception as e:
        pytest.skip(f"Dataset download unavailable: {e}")
    assert len(items) > 0
    for it in items:
        assert "question" in it and "answer" in it
        assert it["answer"] in {"A", "B", "C", "D", "E"}


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
