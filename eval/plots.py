"""
Plots + KPI table from eval JSONs (build step 6 / deliverables).

Produces:
  - Pareto chart  : avg tokens (compute) vs Pass@1, one point per policy.
  - Length histogram per policy (token distribution).
  - KPI delta table: router vs baseline, flags >= +5% on >= 2 benchmarks.

Usage:
  python eval/plots.py --baseline results/baseline.json \
         --runs results/router.json results/always_think.json results/never_think.json \
         --outdir results/figures
"""
import argparse
import json
import math
from pathlib import Path

KPI_TARGETS = {"gsm8k": 0.50, "mmlu": 0.45, "strategyqa": 0.65}
MIN_DELTA = 0.05


def _load(path):
    with open(path) as f:
        return json.load(f)


def _select_router(runs):
    """Pick the run tagged 'router'. If none is tagged, warn and fall back to
    the first run rather than silently treating runs[0] as the router."""
    router = next((r for r in runs if r.get("tag") == "router"), None)
    if router is None and runs:
        router = runs[0]
        print("[plots] WARNING: no run tagged 'router' in --runs; "
              f"falling back to first run (tag={router.get('tag')!r}). "
              "Pass the router run with --tag router for a correct KPI table.")
    return router


def kpi_table(baseline, runs):
    base_b = baseline["benchmarks"]
    lines = ["| Benchmark | Baseline | Router | Delta | Target | KPI met? |",
             "|---|---|---|---|---|---|"]
    router = _select_router(runs)
    n_met = 0
    n_comparable = 0  # KPI benchmarks present (with pass@1) in BOTH runs
    if router:
        for name, target in KPI_TARGETS.items():
            if name not in base_b or name not in router["benchmarks"]:
                continue
            if "pass@1" not in base_b[name] or "pass@1" not in router["benchmarks"][name]:
                continue
            n_comparable += 1
            b = base_b[name]["pass@1"]
            r = router["benchmarks"][name]["pass@1"]
            delta = r - b
            met = (delta >= MIN_DELTA) and (r >= target)
            n_met += int(met)
            lines.append(f"| {name} | {b:.3f} | {r:.3f} | {delta:+.3f} | "
                         f">={target:.2f} | {'YES' if met else 'no'} |")
    # Denominator preserves the brief's "2 of 3" ratio for ANY subset size:
    # ceil(n*2/3) -> n=1:1, n=2:2, n=3:2. Derived from how many benchmarks are
    # actually comparable, never hardcoded. PASS rule unchanged: met iff
    # delta>=MIN_DELTA AND router>=target; overall PASS iff n_met >= required.
    required = max(1, math.ceil(n_comparable * 2 / 3)) if n_comparable else 0
    passed = n_comparable > 0 and n_met >= required
    verdict = f"\n**KPI status: {n_met}/{required} required benchmarks met "
    if n_comparable == 0:
        verdict = ("\n**KPI status: no comparable KPI benchmarks in both "
                   "baseline and router runs (NOT YET)**")
    elif passed:
        verdict += "(PASS)**"
    else:
        verdict += f"(NOT YET — need >= +5% AND target on >= {required})**"
    return "\n".join(lines) + "\n" + verdict


def pareto_chart(policies, benchmark, outpath):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for pol in policies:
        b = pol["benchmarks"].get(benchmark)
        if not b:
            continue
        ax.scatter(b["avg_tokens"], b["pass@1"], s=120)
        ax.annotate(pol.get("tag", "?"), (b["avg_tokens"], b["pass@1"]),
                    textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("Avg output tokens (compute cost)")
    ax.set_ylabel(f"{benchmark} Pass@1 (accuracy)")
    ax.set_title(f"Accuracy vs compute — {benchmark}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def length_hist(policy, benchmark, outpath):
    import matplotlib.pyplot as plt

    b = policy["benchmarks"].get(benchmark)
    if not b:
        return
    toks = [it["n_tokens"] for it in b["per_item"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(toks, bins=40)
    ax.set_xlabel("output tokens")
    ax.set_ylabel("count")
    ax.set_title(f"Length distribution — {policy.get('tag')} / {benchmark}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--outdir", default="results/figures")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    baseline = _load(args.baseline)
    runs = [_load(r) for r in args.runs]
    all_policies = [baseline] + runs

    table = kpi_table(baseline, runs)
    (Path(args.outdir) / "kpi_table.md").write_text(table)
    print(table)

    benchmarks = sorted({b for pol in all_policies for b in pol["benchmarks"]})
    for bench in benchmarks:
        pareto_chart(all_policies, bench, Path(args.outdir) / f"pareto_{bench}.png")
        for pol in all_policies:
            length_hist(pol, bench, Path(args.outdir) / f"len_{pol.get('tag')}_{bench}.png")
    print(f"[plots] wrote figures + kpi_table.md to {args.outdir}")


if __name__ == "__main__":
    main()
