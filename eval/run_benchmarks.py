"""
Evaluation harness (build step 6).

Measures Pass@1 (and optionally Pass@k) plus efficiency stats (avg tokens,
latency, think-rate) for any (model, adapter, route_mode) combination on
GSM8K / MMLU / StrategyQA / AIME24.

KPI workflow:
  1. Baseline  : base reasoner, no adapter, full CoT.
       python eval/run_benchmarks.py --benchmark all --route-mode always_think \
              --tag baseline --out results/baseline.json
  2. Router    : trained adapter, adaptive routing.
       python eval/run_benchmarks.py --benchmark all --adapter outputs/router-seed0 \
              --verifier-ckpt outputs/verifier-400m/best.pt --route-mode model \
              --tag router --out results/router.json
  3. Delta is computed by eval/plots.py (must be >= +5% on >= 2 benchmarks).

Pass@k uses the unbiased Chen et al. estimator over --n-samples sampled
completions per question. Any requested k > n_samples is dropped (the
estimator would otherwise report a guaranteed-hit 100%), and Pass@1 is
averaged across --seeds for an honest mean/std.
"""
import argparse
import json
import math
from pathlib import Path

# NOTE: load_benchmark (which imports `datasets`) is imported lazily inside
# eval_benchmark so the pure helpers below stay importable without the ML stack.
from adaptivethink.metrics import is_correct as _is_correct, pass_at_k
from adaptivethink.router.reward import extract_answer

BENCHMARKS = ["gsm8k", "mmlu", "strategyqa", "aime24"]


def valid_k_list(k_list, n_samples):
    """Drop any k > n_samples (the estimator's n-c<k branch would report a
    guaranteed 100%). Returns (kept_sorted, dropped_sorted) — pure, no I/O."""
    kept = sorted({k for k in k_list if 0 < k <= n_samples})
    dropped = sorted({k for k in k_list if k > n_samples or k <= 0})
    return kept, dropped


def parse_seeds(seeds_arg, default_seed):
    """Parse a comma list of seeds; fall back to [default_seed] for back-compat."""
    if not seeds_arg:
        return [default_seed]
    seeds = [int(x) for x in seeds_arg.split(",") if x.strip()]
    return seeds or [default_seed]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    """Sample standard deviation (Bessel-corrected, /(N-1)) — the unbiased
    estimator for the small seed counts (2-3) used here."""
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def eval_benchmark(pipeline, name, n, seed, k_list, n_samples):
    from adaptivethink.data.loaders import load_benchmark

    # Defensive clamp: never let an invalid k reach the estimator even if the
    # caller forgot to filter (the headline filtering happens in main()). Warn
    # so programmatic callers that bypass main() still learn k was dropped.
    k_list, _dropped = valid_k_list(k_list, n_samples)
    if _dropped:
        import warnings
        warnings.warn(
            f"eval_benchmark: dropping k={_dropped} (k>n_samples={n_samples} or k<=0); "
            "Pass@k for those would be a spurious guaranteed-hit.")
    items = load_benchmark(name, seed=seed, n=n)
    per_item, n_correct, n_think, tok_total, lat_total = [], 0, 0, 0.0, 0.0
    pass_k_counts = {k: 0.0 for k in k_list}

    for it in items:
        q, gt = it["question"], it["answer"]
        # Pass@1 with a single (greedy) sample drives the headline accuracy + stats.
        res = pipeline.answer(q, greedy=True)
        correct = _is_correct(res.completion, gt)
        n_correct += int(correct)
        n_think += int(res.decision == "think")
        tok_total += res.n_tokens
        lat_total += res.latency_s

        if k_list:
            samples = [pipeline.answer(q, greedy=False) for _ in range(n_samples)]
            c = sum(_is_correct(s.completion, gt) for s in samples)
            for k in k_list:
                pass_k_counts[k] += pass_at_k(n_samples, c, k)

        per_item.append({
            "question": q[:200], "gt": gt, "pred": extract_answer(res.completion),
            "correct": correct, "decision": res.decision,
            "difficulty": round(res.difficulty, 3), "n_tokens": res.n_tokens,
            "latency_s": round(res.latency_s, 3),
        })

    m = len(items)
    return {
        "benchmark": name,
        "n": m,
        "pass@1": round(n_correct / m, 4) if m else 0.0,
        "pass@k": {str(k): round(v / m, 4) for k, v in pass_k_counts.items()} if k_list else {},
        "think_rate": round(n_think / m, 4) if m else 0.0,
        "avg_tokens": round(tok_total / m, 1) if m else 0.0,
        "avg_latency_s": round(lat_total / m, 4) if m else 0.0,
        "per_item": per_item,
    }


def eval_benchmark_seeds(pipeline, name, n, seeds, k_list, n_samples):
    """Run eval_benchmark once per seed, return a single per-benchmark dict.

    pass@1 stays the mean across seeds (backward-compatible JSON shape) and we
    add pass@1_std + pass@1_seeds for honest variance. When --n is None the
    sets are identical so std is 0.0; we still report it. Rich fields
    (per_item, pass@k, stats) come from the first seed for stable artifacts.
    """
    runs = [eval_benchmark(pipeline, name, n, s, k_list, n_samples) for s in seeds]
    p1 = [r["pass@1"] for r in runs]
    out = dict(runs[0])  # copy: keep per_item/pass@k/stats from the first seed
    out["pass@1"] = round(_mean(p1), 4)
    out["pass@1_std"] = round(_std(p1), 4)
    out["pass@1_seeds"] = {str(s): r["pass@1"] for s, r in zip(seeds, runs)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="PEFT router adapter dir")
    p.add_argument("--verifier-ckpt", default=None)
    p.add_argument("--gguf", default=None, help="GGUF Q4_K_M file (on-device backend)")
    p.add_argument("--route-mode", default="model",
                   choices=["model", "threshold", "always_think", "never_think"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--benchmark", default="all",
                   choices=BENCHMARKS + ["all", "math500"])
    p.add_argument("--n", type=int, default=None, help="limit items per benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="",
                   help='comma list, e.g. "0,1,2"; defaults to just --seed. '
                        "Each seed reshuffles the --n subsample so Pass@1 "
                        "variance is captured (mean+std reported).")
    p.add_argument("--pass-k", default="", help='e.g. "1,8,64" — enables sampling')
    p.add_argument("--n-samples", type=int, default=8, help="samples for Pass@k")
    p.add_argument("--max-think-tokens", type=int, default=1024)
    p.add_argument("--max-answer-tokens", type=int, default=256)
    p.add_argument("--no-budget-force", action="store_true")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default="results/eval.json")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from adaptivethink.inference.pipeline import build_pipeline
    pipeline = build_pipeline(
        model_name=args.model_name,
        adapter_path=args.adapter,
        verifier_ckpt=args.verifier_ckpt,
        gguf_path=args.gguf,
        device=device,
        route_mode=args.route_mode,
        threshold=args.threshold,
        max_think_tokens=args.max_think_tokens,
        max_answer_tokens=args.max_answer_tokens,
        budget_force=not args.no_budget_force,
    )

    requested_k = [int(x) for x in args.pass_k.split(",") if x.strip()] if args.pass_k else []
    k_list, dropped_k = valid_k_list(requested_k, args.n_samples)
    if dropped_k:
        print(f"[eval] WARNING: dropping Pass@k for k > n_samples ({args.n_samples}): "
              f"k={dropped_k}. These would report a spurious 100% (estimator's "
              f"n-c<k branch). Increase --n-samples to evaluate them.")

    seeds = parse_seeds(args.seeds, args.seed)
    targets = BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    results = {"tag": args.tag, "route_mode": args.route_mode,
               "adapter": args.adapter, "gguf": args.gguf,
               "seeds": seeds, "benchmarks": {}}
    for name in targets:
        print(f"[eval] {name} (seeds={seeds}) ...")
        r = eval_benchmark_seeds(pipeline, name, args.n, seeds, k_list, args.n_samples)
        results["benchmarks"][name] = r
        print(f"  {name}: Pass@1={r['pass@1']:.3f}+-{r['pass@1_std']:.3f}  "
              f"think={r['think_rate']:.2f}  tok={r['avg_tokens']:.0f}  "
              f"lat={r['avg_latency_s']:.2f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
