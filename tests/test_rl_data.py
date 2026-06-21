"""Offline-safe unit tests for adaptivethink.rl.data (no network required).

Heavy loaders are monkeypatched so the pool/mapper logic is exercised without
downloading any HF dataset. These tests also pin the interface contract:
StrategyQA gold is 'True'/'False', AQuA gold is a bare option letter, and
build_grpo_pool emits {prompt, answer, source} in a 45/30/25 mix.
"""
from collections import Counter

import adaptivethink.rl.data as d


# --- parse_datasets: aliases + validation -------------------------------------

def test_parse_datasets_supports_aqua_and_aliases():
    assert d.parse_datasets("gsm8k,strategyqa,aqua") == ["gsm8k", "strategyqa", "aqua"]
    assert d.parse_datasets("aqua_rat") == ["aqua"]
    assert d.parse_datasets("aquarat") == ["aqua"]
    assert d.parse_datasets("gsm8k, aqua_rat , gsm8k") == ["gsm8k", "aqua"]


def test_parse_datasets_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        d.parse_datasets("gsm8k,not_a_dataset")


# --- mappers: schema + gold normalisation -------------------------------------

def test_strategyqa_gold_is_true_false():
    assert d.strategyqa_to_row({"question": "q", "answer": True})["answer"] == "True"
    assert d.strategyqa_to_row({"question": "q", "answer": False})["answer"] == "False"
    assert d.strategyqa_to_row({"question": "q", "answer": "yes"})["answer"] == "True"
    assert d.strategyqa_to_row({"question": "q", "answer": "no"})["answer"] == "False"


def test_aqua_to_row_letter_gold_and_lettered_options():
    ex = {
        "question": "How many?",
        "options": ["A)125", "B)150", "C)175", "D)200", "E)225"],
        "correct": "C",
    }
    row = d.aqua_to_row(ex)
    assert row["answer"] == "C"
    assert row["dataset"] == "aqua"
    assert "(A) 125" in row["question"] and "(E) 225" in row["question"]
    assert "A)125" not in row["question"]
    # prompt uses the repo ChatML <think>/<answer> style
    assert row["prompt"].startswith("<|im_start|>system")
    assert row["prompt"].endswith("<|im_start|>assistant\n")


def test_aqua_option_formatter_strips_varied_prefixes():
    assert d._format_aqua_options(["125", "150"]) == "(A) 125\n(B) 150"
    assert d._format_aqua_options(["A ) 5 apples", "B. 10"]) == "(A) 5 apples\n(B) 10"


# --- build_grpo_pool: mix, schema, determinism (monkeypatched loaders) --------

def _patch_loaders(monkeypatch):
    monkeypatch.setattr(
        d, "_load_gsm8k_rows",
        lambda split, instr: [
            {"prompt": f"g{i}", "answer": str(i), "dataset": "gsm8k", "question": f"qg{i}"}
            for i in range(5000)
        ],
    )
    monkeypatch.setattr(
        d, "_load_strategyqa_rows",
        lambda split, instr: [
            {"prompt": f"s{i}", "answer": "True", "dataset": "strategyqa", "question": f"qs{i}"}
            for i in range(1603)
        ],
    )
    monkeypatch.setattr(
        d, "_load_aqua_rows",
        lambda split, instr: [
            {"prompt": f"a{i}", "answer": "A", "dataset": "aqua", "question": f"qa{i}"}
            for i in range(2000)
        ],
    )


def test_build_grpo_pool_default_mix(monkeypatch):
    _patch_loaders(monkeypatch)
    pool = d.build_grpo_pool()
    assert len(pool) == 3000 + 1603 + 1500
    counts = Counter(r["source"] for r in pool)
    assert counts == {"gsm8k": 3000, "strategyqa": 1603, "aqua": 1500}


def test_build_grpo_pool_schema_and_shuffle(monkeypatch):
    _patch_loaders(monkeypatch)
    pool = d.build_grpo_pool()
    assert all(set(r.keys()) == {"prompt", "answer", "source"} for r in pool)
    # shuffled, not grouped by source
    assert [r["source"] for r in pool[:12]] != ["gsm8k"] * 12


def test_build_grpo_pool_is_deterministic(monkeypatch):
    _patch_loaders(monkeypatch)
    a = d.build_grpo_pool(seed=7)
    b = d.build_grpo_pool(seed=7)
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]


def test_build_grpo_pool_caps_to_available(monkeypatch):
    _patch_loaders(monkeypatch)
    pool = d.build_grpo_pool(n_gsm8k=10, n_strategyqa=20, n_aqua=999_999)
    counts = Counter(r["source"] for r in pool)
    assert counts["gsm8k"] == 10 and counts["strategyqa"] == 20
    assert counts["aqua"] == 2000  # only 2000 available -> take all


# --- lazy package export ------------------------------------------------------

def test_build_grpo_pool_lazy_export():
    from adaptivethink.rl import build_grpo_pool
    assert build_grpo_pool is d.build_grpo_pool
