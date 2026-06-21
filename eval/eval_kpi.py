"""
KPI evaluator — TRAIN-CONSISTENT scoring, vLLM (fast) or HF (fallback) backend.

Fixes the below-random MMLU/StrategyQA artifact: the old eval prompted with the
router '\\boxed{}' convention and parsed '\\boxed{}', while the model is TRAINED
on the RLVR '<think>/<answer>' convention — and MMLU/StrategyQA were never told
the required answer format (letter / True-False). This evaluator uses the EXACT
same prompt + answer extraction as Dr.GRPO training, so the trained model is
scored in-distribution and the baseline is scored the same way (apples-to-apples).

  * Prompts: adaptivethink.rl.data mappers for gsm8k/strategyqa/aqua (identical to
    training); a letter-answer prompt for mmlu.
  * Prediction: the <answer>...</answer> content (rl.rewards.predicted_answer),
    falling back to the repo's extract_answer if the tag is absent.
  * Matching: the repo's tolerant _answers_match (numeric/fraction/boolean),
    plus a letter normaliser for mmlu.

Backends:
  * vllm (default): batched, continuous-scheduling generation — saturates the GPU
    and a single repetition-loop item never blocks the rest. ~5-10x faster.
    Greedy = SamplingParams(temperature=0). LoRA via --adapter (vLLM LoRARequest).
  * hf: one-prompt-at-a-time transformers (+ merged PEFT adapter). Exact, simple,
    but slow (single-sequence decode underuses the GPU). Use only if vLLM errors.

Run it twice with identical flags — base model = baseline, base+adapter = trained
— and the per-benchmark delta is the honest +X%.

  python eval/eval_kpi.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
      --datasets gsm8k,strategyqa,mmlu --n 200 --out results/kpi_baseline.json
  python eval/eval_kpi.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
      --adapter outputs/grpo-seed0 --datasets gsm8k,strategyqa,mmlu --n 200 \
      --out results/kpi_trained.json
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Stdlib-only repo modules (safe to import without torch); the model stack is
# imported lazily inside main().
from adaptivethink.rl.rewards import predicted_answer
from adaptivethink.router.reward import _answers_match, match_answer

_LETTER_RE = re.compile(r"[A-Ea-e]")

# Standard broad MMLU: a balanced 16-subject spread across all four MMLU
# categories (STEM / humanities / social sciences / other). load_mmlu's own
# default is 3 math-only subjects, which is an unrepresentative, unusually hard
# slice — the KPI's "MMLU >=45%" refers to general MMLU, so we sample broadly.
_MMLU_SUBJECTS = [
    # STEM
    "high_school_mathematics", "college_computer_science",
    "high_school_physics", "high_school_biology",
    # Humanities
    "philosophy", "world_religions",
    "high_school_european_history", "moral_scenarios",
    # Social sciences
    "high_school_psychology", "sociology",
    "high_school_geography", "high_school_government_and_politics",
    # Other
    "clinical_knowledge", "management", "marketing", "miscellaneous",
]

# Qwen ChatML turn terminator — stop generation here so a base model that never
# emits </answer> can't ramble to max_tokens (the <answer> block, if any, is
# already produced before this token, so extraction still works).
_STOP = ["<|im_end|>"]


# MMLU option-letter patterns, most explicit first. cais/mmlu has 4 options (A-D).
_MMLU_PATTERNS = [
    re.compile(r"(?:answer|option|correct)\s*(?:is|:)?\s*\(?([A-Da-d])\b", re.I),
    re.compile(r"\(([A-Da-d])\)"),
    re.compile(r"\b([A-D])\b"),
]


def _norm_mmlu(pred: str | None) -> str | None:
    """Extract the MMLU option letter (A-D) robustly.

    Prefers an explicit 'answer is X' / '(X)' / isolated capital A-D (taking the
    LAST mention — the stated conclusion), falling back to the first stray a-e
    only as a last resort. Avoids the classic bug of matching the 'e' in 'The...'
    as the answer. Validated to reproduce the offline re-score (MMLU base 53.5 /
    trained 65.0)."""
    if pred is None:
        return None
    text = str(pred)
    for pat in _MMLU_PATTERNS:
        m = pat.findall(text)
        if m:
            return m[-1].upper()
    m = _LETTER_RE.search(text)
    return m.group(0).upper() if m else text.strip()


def _build_items(name: str, split: str, n: int, seed: int,
                 mmlu_subjects: list | None = None) -> list[dict]:
    """Build {prompt, answer, question} rows using the TRAINING prompt format.

    n falsy (0/None) => use the FULL split (no cap) — used for the lock-in eval.
    """
    name = name.lower().strip()
    if name in ("gsm8k", "strategyqa", "aqua", "aqua_rat", "aquarat"):
        from adaptivethink.rl import data as rl_data
        rows = rl_data.load_rows([name], split=split)  # load_rows applies aliases
        if n and len(rows) > n:
            random.Random(seed).shuffle(rows)
            rows = rows[:n]
        return rows
    if name == "mmlu":
        from adaptivethink.data.loaders import load_mmlu
        from adaptivethink.rl.data import _build_prompt, SYSTEM_PROMPT
        instr = (SYSTEM_PROMPT
                 + "\nAnswer with the letter (A, B, C, or D) of the correct option.")
        subjects = mmlu_subjects or _MMLU_SUBJECTS
        return [{"prompt": _build_prompt(it["question"], instr),
                 "answer": it["answer"], "question": it["question"]}
                for it in load_mmlu(subjects=subjects, seed=seed, n=n)]
    raise ValueError(f"unknown benchmark: {name}")


def _is_correct(name: str, pred: str | None, gold: str) -> bool:
    # MMLU: reduce to the option letter first (matches the validated re-score).
    if name == "mmlu":
        return _answers_match(_norm_mmlu(pred) or "", gold)
    # gsm8k / strategyqa / aqua: tolerant compare + standard per-type extraction.
    return match_answer(pred, gold)


def _canon(name: str, pred: str | None) -> str | None:
    """Canonical answer key for majority voting (per-benchmark).

    Reduces a sampled completion's answer to a comparable token so identical
    answers group together: option letter (mmlu/aqua), True/False (strategyqa),
    or the last number (gsm8k). None when nothing parseable.
    """
    if pred is None:
        return None
    pred = str(pred)
    if name == "mmlu":
        return _norm_mmlu(pred)
    if name in ("aqua", "aqua_rat", "aquarat"):
        m = re.findall(r"\b([A-Ea-e])\b", pred)  # word-boundary, take last (the conclusion)
        return m[-1].upper() if m else None
    if name == "strategyqa":
        toks = re.findall(r"\b(true|false|yes|no)\b", pred.lower())
        if toks:
            return "True" if toks[-1] in ("true", "yes") else "False"
        return None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", pred.replace(",", ""))
    return nums[-1] if nums else None


def _evaluate(name: str, rows: list[dict], gen_iter) -> tuple[int, int, list[dict]]:
    """Score (text, n_tok) pairs against gold. gen_iter may be a lazy generator
    (HF: yields one generation at a time -> live progress) or a precomputed list
    (vLLM: already batched). Either way scoring is identical."""
    n_correct, tok_total, per_item = 0, 0, []
    total = len(rows)
    for i, (r, (text, n_tok)) in enumerate(zip(rows, gen_iter)):
        pred = predicted_answer(text)
        ok = bool(_is_correct(name, pred, str(r["answer"])))
        n_correct += int(ok)
        tok_total += n_tok
        per_item.append({"question": r["question"][:200], "gold": r["answer"],
                         "pred": pred, "correct": ok, "n_tokens": n_tok,
                         "completion": text[:2000]})  # saved so we can re-score offline
        if (i + 1) % 25 == 0:
            print(f"    [{name}] {i + 1}/{total}  "
                  f"running Pass@1={n_correct / (i + 1):.3f}", flush=True)
    return n_correct, tok_total, per_item


def _hf_gen(backend, rows: list[dict], max_new_tokens: int):
    """Lazy generator: one greedy HF generation per row (live progress)."""
    for r in rows:
        yield backend.generate(r["prompt"], max_new_tokens, 0.0, greedy=True)


def _vllm_gen(llm, rows, sampling_params, lora_request, name, vote) -> list[tuple[str, int]]:
    """Batched vLLM generation in input order.

    vote==1: greedy single-shot, returns each completion verbatim.
    vote>1 : self-consistency — sample `vote` completions per prompt, majority-vote
    the canonical answer, and return it wrapped as '<answer>X</answer>' so the
    scorer is unchanged. n_tok is the TOTAL across samples (honest latency cost).
    """
    prompts = [r["prompt"] for r in rows]
    outs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    if vote <= 1:
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]
    results = []
    for o in outs:
        keys, ntok = [], 0
        for comp in o.outputs:
            ntok += len(comp.token_ids)
            k = _canon(name, predicted_answer(comp.text))
            if k is not None:
                keys.append(k)
        winner = Counter(keys).most_common(1)[0][0] if keys else ""
        results.append((f"<answer>{winner}</answer>", ntok))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for baseline)")
    p.add_argument("--datasets", default="gsm8k,strategyqa,mmlu")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=200, help="items per benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-mem-util", type=float, default=0.85, help="vLLM only")
    p.add_argument("--max-lora-rank", type=int, default=64, help="vLLM only")
    p.add_argument("--vote", type=int, default=1,
                   help="self-consistency samples per item (1=greedy single-shot; vLLM only)")
    p.add_argument("--mmlu-all", action="store_true",
                   help="use the full 57-subject cais/mmlu 'all' config (else the 16-subject spread)")
    p.add_argument("--out", default="results/kpi_eval.json")
    args = p.parse_args()

    tag = "trained" if args.adapter else "baseline"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"[eval_kpi] {tag}: backend={args.backend} model={args.model} "
          f"adapter={args.adapter} datasets={datasets} n={args.n} "
          f"split={args.split} max_new={args.max_new_tokens}", flush=True)

    # ── backend setup: each exposes make_gen(rows) -> iterable of (text, n_tok) ──
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm_kwargs = dict(model=args.model, dtype="bfloat16",
                          gpu_memory_utilization=args.gpu_mem_util,
                          max_model_len=2048, enforce_eager=True)
        lora_request = None
        if args.adapter:
            llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
        llm = LLM(**llm_kwargs)
        if args.adapter:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("grpo", 1, args.adapter)
        vote = max(1, args.vote)
        if vote > 1:
            sampling_params = SamplingParams(n=vote, temperature=0.7, top_p=0.95,
                                             max_tokens=args.max_new_tokens, stop=_STOP)
        else:
            sampling_params = SamplingParams(temperature=0.0,
                                             max_tokens=args.max_new_tokens, stop=_STOP)

        def make_gen(rows, name):
            return _vllm_gen(llm, rows, sampling_params, lora_request, name, vote)
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from adaptivethink.inference.pipeline import HFBackend
        backend = HFBackend(args.model, args.adapter, device)

        def make_gen(rows, name):  # voting is vLLM-only; HF stays greedy single-shot
            return _hf_gen(backend, rows, args.max_new_tokens)

    # ── run the benchmarks ──────────────────────────────────────────────────────
    results = {"tag": tag, "backend": args.backend, "model": args.model,
               "adapter": args.adapter, "split": args.split, "n": args.n,
               "vote": args.vote, "benchmarks": {}}
    mmlu_subjects = ["all"] if args.mmlu_all else _MMLU_SUBJECTS
    for name in datasets:
        rows = _build_items(name, args.split, args.n, args.seed, mmlu_subjects)
        mode = f"vote@{args.vote}" if args.vote > 1 else args.backend
        print(f"  [{name}] generating {len(rows)} items ({mode})...", flush=True)
        n_correct, tok_total, per_item = _evaluate(name, rows, make_gen(rows, name))
        m = len(rows)
        results["benchmarks"][name] = {
            "n": m,
            "pass@1": round(n_correct / m, 4) if m else 0.0,
            "avg_tokens": round(tok_total / m, 1) if m else 0.0,
            "per_item": per_item,
        }
        print(f"  {name}: Pass@1={results['benchmarks'][name]['pass@1']:.3f} "
              f"({n_correct}/{m})  avg_tok={results['benchmarks'][name]['avg_tokens']:.0f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval_kpi] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
