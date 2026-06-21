"""CCDD — Competence-Calibrated Difficulty Self-Distillation (API-free).

This module is the **teacher-free** replacement for ``data/teacher_labels.py``.
Instead of asking an external LLM to *guess* how hard a question is, we measure
how hard it is *for the model we are actually training* — i.e. difficulty is
distilled from the model's own competence, with **no teacher API** anywhere.

CCDD in one line
----------------
    difficulty(q) = 1 - solve_rate(q)
    solve_rate(q) = (# of K rollouts whose final answer matches gold) / K

The base model is sampled ``K`` times per question (temperature > 0 so the
rollouts differ). Each rollout's final answer is extracted and checked against
the gold answer using the **exact same verifier the GRPO reward uses**
(``adaptivethink.router.reward.extract_answer`` + ``_answers_match`` — numeric /
boolean / letter / fraction tolerant). The empirical solve-rate is the model's
*competence* on that item; one minus it is the *self-distilled difficulty*.

Why this is useful (two downstream consumers, both API-free)
------------------------------------------------------------
1. **Curriculum filter for Dr.GRPO.** GRPO advantage is zero whenever every
   rollout in a group earns the same reward, so items the base model *always*
   solves (solve_rate == 1.0 -> difficulty 0.0) or *never* solves
   (solve_rate == 0.0 -> difficulty 1.0) contribute no gradient. Dropping those
   two extremes keeps only the learnable band — this mirrors the offline
   difficulty filter already wired in ``rl/data.py`` / ``rl/drgrpo_train.py``.
2. **Training the tiny difficulty verifier.** The rows written here use the SAME
   schema ``verifier/train.py`` consumes ({question, answer, difficulty}, plus
   solve_rate/source extras it simply ignores), so
   ``scripts/03_train_verifier.sh`` can train the verifier on these *self-labels*
   with no teacher. That verifier gates think / no_think at inference time.

Output schema (JSONL, one row per kept question)::

    {"question": str, "answer": str, "difficulty": float,
     "solve_rate": float, "source": str}

``difficulty`` and ``question``/``answer`` match the columns ``verifier/train.py``
reads; ``solve_rate`` and ``source`` are extra diagnostics.

Implementation notes
--------------------
* Heavy deps (vllm / torch / transformers / datasets) are imported INSIDE
  functions so ``python3 -m py_compile`` works on a machine without them.
* Rollouts PREFER vLLM (``from vllm import LLM, SamplingParams``) for throughput
  and fall back to HF ``transformers`` ``.generate`` when vLLM is unavailable.
* The question pool comes from ``rl.data`` (the same loaders GRPO trains on), so
  self-difficulty is computed over exactly the train pool — never on any test /
  mmlu split.

CLI::

    python -m adaptivethink.rl.self_difficulty \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --datasets gsm8k,strategyqa,aqua \
        --k 8 --n 2000 --out data/self_difficulty.jsonl \
        --temperature 0.9 --max-new-tokens 512 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

# Interface-contract default model. 3B / 7B are one-flag switches.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASETS = "gsm8k,strategyqa,aqua"
DEFAULT_OUT = "data/self_difficulty.jsonl"


# ── argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m adaptivethink.rl.self_difficulty",
        description=(
            "CCDD: compute self-distilled (teacher-free) difficulty by sampling "
            "the base model K times per question and scoring with the GRPO "
            "verifier. solve_rate = #correct/K; difficulty = 1 - solve_rate."
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id (default {DEFAULT_MODEL}). 3B/7B are "
                        "one-flag switches.")
    p.add_argument("--datasets", default=DEFAULT_DATASETS,
                   help="Comma-separated dataset names "
                        f"(default '{DEFAULT_DATASETS}').")
    p.add_argument("--k", type=int, default=8,
                   help="Rollouts per question (K). solve_rate denom.")
    p.add_argument("--n", type=int, default=2000,
                   help="Cap on #questions (24h budget). Negative = no cap.")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"Output JSONL path (default {DEFAULT_OUT}).")
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Sampling temperature for rollouts (>0 for diversity).")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max new tokens per rollout.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for pool shuffle/subset and sampling.")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Nucleus top-p for rollouts.")
    p.add_argument("--no-vllm", dest="use_vllm", action="store_false",
                   default=True,
                   help="Force the transformers fallback instead of vLLM.")
    return p


# ── pool loading (reuses rl.data, the GRPO train loaders) ──────────────────────

def _row_question(row: dict) -> Optional[str]:
    """The natural-language question text from a pool row (verifier input)."""
    q = row.get("question")
    return q if isinstance(q, str) and q.strip() else None


def _row_prompt(row: dict) -> Optional[str]:
    """The full chat-formatted prompt to feed the model for sampling."""
    p = row.get("prompt")
    if isinstance(p, str) and p.strip():
        return p
    # Defensive fallback: some pool variants may only carry 'question'.
    return _row_question(row)


def _row_source(row: dict) -> str:
    """Dataset/source tag; rl.data uses 'dataset', other variants use 'source'."""
    return str(row.get("source") or row.get("dataset") or "unknown")


def load_pool(datasets_spec: str, n: int, seed: int) -> list[dict]:
    """Load the GRPO train pool as a list of dict rows, capped to ``n`` items.

    Prefers ``rl.data.build_grpo_pool`` (the interface-contract pool builder).
    Falls back to ``rl.data.build_dataset`` / ``rl.data.load_rows`` so this module
    works against the current repo state too. ``datasets`` is imported lazily by
    those builders; we never touch any test / mmlu split here.
    """
    from . import data as rl_data

    rows: Optional[list[dict]] = None

    # 1) Preferred: load full rows (these carry 'question') then shuffle/subset.
    #    build_grpo_pool emits only {prompt,answer,source} (no 'question'), so we
    #    use load_rows here — CCDD needs the raw question to sample the base model.
    if hasattr(rl_data, "load_rows"):
        names = _parse_datasets(rl_data, datasets_spec)
        rows = _shuffle_subset(list(rl_data.load_rows(names)), seed=seed, n=n)

    # 2) Fallback A: build_dataset returns an HF Dataset (has .to_list()).
    if rows is None and hasattr(rl_data, "build_dataset"):
        names = _parse_datasets(rl_data, datasets_spec)
        max_items = None if n is None or n < 0 else n
        ds = rl_data.build_dataset(names, seed=seed, max_items=max_items)
        rows = _coerce_to_rows(ds)

    # 3) Fallback B: plain row loader + manual shuffle/subset.
    if rows is None and hasattr(rl_data, "load_rows"):
        names = _parse_datasets(rl_data, datasets_spec)
        all_rows = list(rl_data.load_rows(names))
        rows = _shuffle_subset(all_rows, seed=seed, n=n)

    if rows is None:
        raise RuntimeError(
            "rl.data exposes none of build_grpo_pool/build_dataset/load_rows; "
            "cannot load the question pool."
        )

    # Final safety cap (build_grpo_pool may or may not honour n).
    if n is not None and n >= 0:
        rows = rows[:n]
    if not rows:
        raise ValueError(
            "Loaded 0 pool rows — check --datasets / --n / --seed."
        )
    return rows


def _parse_datasets(rl_data, spec: str) -> list[str]:
    """Use rl.data.parse_datasets if present, else a permissive split."""
    parser = getattr(rl_data, "parse_datasets", None)
    if callable(parser):
        return list(parser(spec))
    return [s.strip().lower() for s in spec.split(",") if s.strip()]


def _coerce_to_rows(pool: Any) -> Optional[list[dict]]:
    """Coerce a pool (HF Dataset / list of dict) into a list[dict]."""
    if pool is None:
        return None
    if isinstance(pool, list):
        return [dict(r) for r in pool]
    # HF datasets.Dataset exposes to_list().
    to_list = getattr(pool, "to_list", None)
    if callable(to_list):
        return [dict(r) for r in to_list()]
    # Generic iterable of mappings.
    try:
        return [dict(r) for r in pool]
    except TypeError:
        return None


def _shuffle_subset(rows: list[dict], seed: int, n: int) -> list[dict]:
    """Deterministic shuffle then cap to ``n`` (negative/None ⇒ no cap)."""
    import random

    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    if n is not None and n >= 0:
        return shuffled[:n]
    return shuffled


# ── scoring (REUSE the GRPO verifier — do NOT reimplement matching) ───────────

def score_solve_rate(samples: Sequence[str], gold: str) -> float:
    """Fraction of ``samples`` whose extracted answer matches ``gold``.

    Reuses adaptivethink.router.reward.extract_answer + _answers_match exactly
    (the same matcher the GRPO reward uses), so self-difficulty is calibrated to
    the metric the model is actually trained against.
    """
    from adaptivethink.router.reward import _answers_match, extract_answer

    if not samples:
        return 0.0
    n_ok = 0
    for s in samples:
        pred = extract_answer(s)
        if pred is not None and _answers_match(pred, gold):
            n_ok += 1
    return n_ok / float(len(samples))


def difficulty_from_solve_rate(solve_rate: float) -> float:
    """CCDD core: difficulty is one minus the model's own solve-rate."""
    return 1.0 - solve_rate


def is_learnable(solve_rate: float) -> bool:
    """Curriculum filter: drop trivial (1.0) and unsolvable (0.0) items."""
    return 0.0 < solve_rate < 1.0


# ── rollout samplers (vLLM preferred, transformers fallback) ──────────────────

def _sample_vllm(
    prompts: Sequence[str],
    model: str,
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> list[list[str]]:
    """Batch-sample K completions per prompt with vLLM. Returns [N][k] texts."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, seed=seed)
    sampling = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    outputs = llm.generate(list(prompts), sampling)
    results: list[list[str]] = []
    for out in outputs:
        results.append([comp.text for comp in out.outputs])
    return results


def _sample_transformers(
    prompts: Sequence[str],
    model: str,
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> list[list[str]]:
    """HF fallback: K sampled completions per prompt via model.generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = AutoModelForCausalLM.from_pretrained(model).to(device)
    lm.eval()

    results: list[list[str]] = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = lm.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=k,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        results.append(
            [tok.decode(o[prompt_len:], skip_special_tokens=True) for o in out]
        )
    return results


def sample_rollouts(
    prompts: Sequence[str],
    model: str,
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    use_vllm: bool = True,
) -> list[list[str]]:
    """Sample K completions per prompt, preferring vLLM, falling back to HF."""
    if use_vllm:
        try:
            return _sample_vllm(
                prompts, model, k, temperature, top_p, max_new_tokens, seed
            )
        except Exception as e:  # noqa: BLE001 — vLLM missing / GPU/init failure
            print(f"[ccdd] vLLM unavailable ({e!r}); falling back to transformers.")
    return _sample_transformers(
        prompts, model, k, temperature, top_p, max_new_tokens, seed
    )


# ── orchestration ─────────────────────────────────────────────────────────────

def compute_self_difficulty(
    rows: Sequence[dict],
    model: str,
    k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    use_vllm: bool = True,
) -> list[dict]:
    """Run rollouts over ``rows`` and return self-difficulty label rows.

    Each output row: {question, answer, difficulty, solve_rate, source}.
    Rows missing a usable question or gold answer are skipped.
    """
    usable: list[dict] = []
    prompts: list[str] = []
    for row in rows:
        question = _row_question(row)
        gold = row.get("answer")
        prompt = _row_prompt(row)
        if question is None or prompt is None or gold is None:
            continue
        usable.append(row)
        prompts.append(prompt)

    if not usable:
        return []

    all_samples = sample_rollouts(
        prompts, model, k, temperature, top_p, max_new_tokens, seed,
        use_vllm=use_vllm,
    )

    labels: list[dict] = []
    for row, samples in zip(usable, all_samples):
        gold = str(row["answer"])
        solve_rate = score_solve_rate(samples, gold)
        labels.append({
            "question": _row_question(row),
            "answer": gold,
            "difficulty": difficulty_from_solve_rate(solve_rate),
            "solve_rate": solve_rate,
            "source": _row_source(row),
        })
    return labels


def write_jsonl(rows: Sequence[dict], out_path: str) -> None:
    """Write label rows to ``out_path`` as JSONL, creating parent dirs."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run(args: argparse.Namespace) -> list[dict]:
    """End-to-end: load pool -> sample -> score -> write JSONL."""
    print(
        f"[ccdd] model={args.model} datasets={args.datasets} "
        f"K={args.k} n={args.n} temp={args.temperature} seed={args.seed}"
    )
    rows = load_pool(args.datasets, n=args.n, seed=args.seed)
    print(f"[ccdd] loaded {len(rows)} pool questions.")

    labels = compute_self_difficulty(
        rows,
        model=args.model,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        use_vllm=args.use_vllm,
    )

    write_jsonl(labels, args.out)

    learnable = sum(1 for r in labels if is_learnable(r["solve_rate"]))
    if labels:
        mean_d = sum(r["difficulty"] for r in labels) / len(labels)
    else:
        mean_d = 0.0
    print(
        f"[ccdd] wrote {len(labels)} self-difficulty rows -> {args.out} | "
        f"mean difficulty={mean_d:.3f} | learnable (0<solve<1)={learnable}"
    )
    return labels


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
