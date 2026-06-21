"""AdaptiveThink-RL demo — base vs RL-trained, on a Mac (no CUDA/vLLM needed).

Shows the RL model reasoning in <think>…</think><answer>…</answer> and getting questions
right where the base model fumbles. Verdicts (correct/wrong) use the repo's OWN answer
matcher, so what you see on screen is scored exactly like the KPI eval.

The questions below are REAL test-set items where, in our full evaluation, the base model
was wrong and our trained model was right (237 such GSM8K items, 108 StrategyQA). They
illustrate the aggregate gain (+12.6 GSM8K / +8.0 MMLU / +5.4 StrategyQA); the full numbers
are the proof (see docs/results.md).

Install (clean venv, Python 3.11/3.12/3.13 — NOT 3.14):
    /opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
    pip install torch transformers peft accelerate

Run:
    python demo.py                 # showcase: BASE vs TRAINED, with correct/wrong + tally
    python demo.py --trained-only  # just the RL model — fastest, cleanest for a video clip
    python demo.py -q "If 3 pens cost $6, how much do 7 pens cost?"
    python demo.py --interactive   # type your own questions
"""
import argparse
import gc
import os
import sys
import time

# Make the repo's matcher importable so on-screen verdicts == KPI scoring.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from adaptivethink.rl.rewards import predicted_answer
    from adaptivethink.router.reward import match_answer
except Exception:                                   # demo still runs without verdicts
    predicted_answer = match_answer = None

# Identical to the training/eval prompt (adaptivethink.rl.data._build_prompt).
SYSTEM = ("Reason step by step inside <think> and </think>, "
          "then give the final answer inside <answer> and </answer>.")

# Real test-set items where base was WRONG and our trained model was RIGHT (from the eval).
SHOWCASE = [
    ("Claire makes a 3 egg omelet every morning for breakfast. How many dozens of "
     "eggs will she eat in 4 weeks?", "7"),
    ("Terry eats 2 yogurts a day. They are currently on sale at 4 yogurts for $5.00. "
     "How much does he spend on yogurt over 30 days?", "75"),
    ("Lloyd has an egg farm. His chickens produce 252 eggs per day and he sells them "
     "for $2 per dozen. How much does Lloyd make on eggs per week?", "294"),
    ("Answer True or False. Would an Olympic athlete be tired out after running a mile?",
     "False"),
]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def build_prompt(question: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def load(model_id: str, adapter: str | None, device: str, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return tok, model.to(device).eval()


def generate(tok, model, device, question: str, max_new_tokens: int) -> tuple[str, float]:
    enc = tok(build_prompt(question), return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return text, time.time() - t0


def is_correct(text: str, gold: str | None):
    if gold is None or match_answer is None or predicted_answer is None:
        return None
    return bool(match_answer(predicted_answer(text), gold))


def run_all(model_id, adapter, device, dtype, questions, max_new, label):
    """Load ONE model, generate for every question, then free it — avoids holding two
    1.5B models on MPS at once (which thrashes memory and crawls)."""
    print(f"\n[loading {label}…]", flush=True)
    tok, model = load(model_id, adapter, device, dtype)
    out = {}
    for i, (q, gold) in enumerate(questions, 1):
        text, secs = generate(tok, model, device, q, max_new)
        out[q] = (text, secs, is_correct(text, gold))
        print(f"  · {label}: {i}/{len(questions)} done", flush=True)
    del model, tok
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return out


def block(label, res):
    text, _secs, ok = res
    tag = "" if ok is None else ("  ✅ CORRECT" if ok else "  ❌ wrong")
    return f"\n----- {label}{tag} -----\n{text}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/grpo-seed0-v2")
    p.add_argument("--trained-only", action="store_true", help="skip the base model (fastest)")
    p.add_argument("-q", "--question", default=None, help="ask one question and exit")
    p.add_argument("--interactive", action="store_true", help="type your own questions")
    p.add_argument("--max-new-tokens", type=int, default=320)
    args = p.parse_args()

    device, dtype = pick_device()
    print(f"[demo] device={device} dtype={dtype} | first run downloads the ~3GB base "
          f"from HuggingFace", flush=True)

    # interactive: one model, loop
    if args.interactive:
        tok, model = load(args.model, args.adapter or None, device, dtype)
        print("\n[interactive] type a question (Ctrl-C / Ctrl-D to quit):")
        try:
            while True:
                q = input("\nQ> ").strip()
                if q:
                    text, _secs = generate(tok, model, device, q, args.max_new_tokens)
                    print(f"\n{text}")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
        return

    questions = [(args.question, None)] if args.question else SHOWCASE

    # generate one model at a time (base first, then trained) — no MPS thrashing
    base_out = None if args.trained_only else run_all(
        args.model, None, device, dtype, questions, args.max_new_tokens, "BASE model")
    trained_out = run_all(
        args.model, args.adapter or None, device, dtype, questions, args.max_new_tokens, "TRAINED model")

    bs = ts = 0
    for q, gold in questions:
        print("\n" + "=" * 80 + f"\nQ: {q}" + (f"\n(expected: {gold})" if gold else "") + "\n" + "=" * 80)
        if base_out is not None:
            print(block("BASE (no RL)", base_out[q]))
            bs += int(bool(base_out[q][2]))
        print(block("TRAINED (RL)", trained_out[q]))
        ts += int(bool(trained_out[q][2]))

    if not args.question:
        n = len(questions)
        print("\n" + "=" * 80)
        print(f"SCORE  base {bs}/{n}   →   trained {ts}/{n}" if base_out is not None
              else f"SCORE  trained {ts}/{n}")
        print("=" * 80)


if __name__ == "__main__":
    main()
