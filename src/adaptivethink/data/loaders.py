"""Data loaders for GSM8K, MATH-500, StrategyQA, AQuA-RAT, MMLU.

``load_dataset`` is imported lazily inside each loader so that importing this
module (and ``python -m py_compile``) succeeds without ``datasets`` installed.
"""


def load_gsm8k(split="train", seed=0, n=None):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip()} for r in ds]


def load_math500(seed=0, n=None):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": r["problem"], "answer": r["answer"]} for r in ds]


def _bool_to_gold(value) -> str:
    """Map a StrategyQA gold (bool or stringy bool) to 'True'/'False'.

    The reward matcher (router.reward._answers_match) treats 'True'/'False' and
    'yes'/'no' as equivalent boolean synonyms, so 'True'/'False' is a safe gold.
    """
    if isinstance(value, str):
        is_true = value.strip().lower() in ("true", "yes", "1")
    else:
        is_true = bool(value)
    return "True" if is_true else "False"


def load_strategyqa(seed=0, n=None):
    """Held-out StrategyQA test set via wics/strategy-qa (its only labeled split).

    Use this for EVAL/sanity. For TRAIN labels use ``load_strategyqa_train``,
    which reads the ChilleD/StrategyQA train split.
    """
    from datasets import load_dataset

    ds = load_dataset("wics/strategy-qa", split="test")
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": r["question"], "answer": _bool_to_gold(r["answer"])} for r in ds]


def load_strategyqa_train(split="train", seed=0, n=None):
    """StrategyQA TRAIN labels via ChilleD/StrategyQA (real labeled train split).

    Gold boolean is mapped to the string 'True'/'False'. Kept separate from
    ``load_strategyqa`` (wics/strategy-qa test) so train/eval never leak.
    """
    from datasets import load_dataset

    ds = load_dataset("ChilleD/StrategyQA", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [{"question": r["question"], "answer": _bool_to_gold(r["answer"])} for r in ds]


def _format_aqua_options(options) -> str:
    """Render AQuA-RAT options as clean '(A) value' lines.

    deepmind/aqua_rat ships options as a list of strings already prefixed with
    their letter, e.g. ['A)125', 'B)150', ...]. We strip any leading
    'A)' / 'A.' / 'A:' style prefix and re-emit a uniform '(A) <value>' line so
    the prompt is consistent regardless of the source punctuation.
    """
    import re

    lines = []
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        text = str(opt).strip()
        # Strip a leading option letter + separator (e.g. 'A)', 'A.', 'A:', '(A)').
        text = re.sub(r"^\(?[A-Ea-e]\)?\s*[).:\-]?\s*", "", text).strip()
        lines.append(f"({letter}) {text}")
    return "\n".join(lines)


def load_aqua_rat(split="train", seed=0, n=None):
    """AQuA-RAT (deepmind/aqua_rat, config 'raw') -> {question, answer}.

    question = problem text + newline + lettered options '(A) .. (E)'.
    answer   = the correct option letter ('A'-'E').
    """
    from datasets import load_dataset

    ds = load_dataset("deepmind/aqua_rat", "raw", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    items = []
    for r in ds:
        question = f"{r['question']}\n{_format_aqua_options(r['options'])}"
        items.append({"question": question, "answer": str(r["correct"]).strip()})
    return items


def load_mmlu(subjects=None, seed=0, n=None):
    from datasets import load_dataset

    subjects = subjects or ["high_school_mathematics", "college_mathematics", "abstract_algebra"]
    items = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        for r in ds:
            choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(r["choices"]))
            items.append({
                "question": f"{r['question']}\n{choices}",
                "answer": chr(65 + r["answer"]),
            })
    import random; rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n] if n else items


def load_aime24(seed=0, n=None):
    """AIME 2024 — 30 advanced-math problems. Integer answers 0-999."""
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"question": r["Problem"], "answer": str(r["Answer"]).strip()} for r in ds]


def load_benchmark(name, split="test", seed=0, n=None):
    """Dispatch by benchmark name. Used by the eval harness."""
    name = name.lower()
    if name == "gsm8k":
        return load_gsm8k(split=split, seed=seed, n=n)
    if name == "math500":
        return load_math500(seed=seed, n=n)
    if name == "strategyqa":
        return load_strategyqa(seed=seed, n=n)
    if name in ("aqua", "aqua_rat"):
        return load_aqua_rat(split=split, seed=seed, n=n)
    if name == "mmlu":
        return load_mmlu(seed=seed, n=n)
    if name == "aime24":
        return load_aime24(seed=seed, n=n)
    raise ValueError(f"Unknown benchmark: {name}")


def build_verifier_pool(seed=0):
    """12k items for verifier distillation."""
    items = (
        load_gsm8k("train", seed, 6000)
        + load_math500(seed, 3000)
        + load_strategyqa(seed, 2000)
        + load_mmlu(seed=seed, n=1000)
    )
    import random; random.Random(seed).shuffle(items)
    return items


def build_verifier_eval(seed=42):
    """500-item held-out set, never used for training."""
    items = (
        load_gsm8k("test", seed, 125)
        + load_math500(seed, 125)
        + load_strategyqa(seed, 125)
        + load_mmlu(seed=seed, n=125)
    )
    import random; random.Random(seed).shuffle(items)
    return items


def _dump(name, split, out):
    """Write load_benchmark(name, split) rows as {question, answer} jsonl."""
    import json, pathlib
    rows = load_benchmark(name, split=split)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
    print(f"Dumped {len(rows)} {name}/{split} rows -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dump", nargs=3, metavar=("NAME", "SPLIT", "OUT"),
                   help="Write load_benchmark(NAME, SPLIT) rows as {question, answer} jsonl to OUT")
    args = p.parse_args()
    if args.dump:
        _dump(args.dump[0], args.dump[1], args.dump[2])
    else:
        p.error("nothing to do: pass --dump NAME SPLIT OUT")
