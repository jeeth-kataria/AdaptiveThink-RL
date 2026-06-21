"""AdaptiveThink-RL demo — base vs RL-trained, side by side, on a Mac (no CUDA/vLLM needed).

Shows the RL model reasoning in <think>…</think><answer>…</answer> and getting questions
right where the base model fumbles. Verdicts (correct/wrong) use the repo's OWN answer
matcher, so what you see on screen is scored exactly like the KPI eval.

Install (clean venv, 4 Mac-friendly deps):
    python3 -m venv .demo && source .demo/bin/activate
    pip install torch transformers peft accelerate

Run:
    python demo.py                 # curated showcase: BASE vs TRAINED, with correct/wrong + tally
    python demo.py --trained-only  # just the RL model (cleaner for a quick clip)
    python demo.py -q "If 3 pens cost $6, how much do 7 pens cost?"
    python demo.py --interactive   # type your own questions
"""
import argparse
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

# Curated so the contrast is visible (gold known -> we can show correct/wrong).
# Real GSM8K items + one StrategyQA-style commonsense question.
SHOWCASE = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
     "babysitting. How much did she earn?", "10"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts "
     "in total does it take?", "3"),
    ("Answer True or False. Would a pear sink in water?", "False"),
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


def verdict(text: str, gold: str | None) -> str:
    """correct/wrong tag using the repo matcher (same as the KPI eval)."""
    if gold is None or match_answer is None or predicted_answer is None:
        return ""
    ok = match_answer(predicted_answer(text), gold)
    return "  ✅ CORRECT" if ok else "  ❌ wrong"


def show(label, tok, model, device, q, gold, max_new):
    text, secs = generate(tok, model, device, q, max_new)
    print(f"\n----- {label} ({secs:.1f}s){verdict(text, gold)} -----\n{text}")
    return verdict(text, gold).strip().startswith("✅") if gold else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/grpo-seed0-v2")
    p.add_argument("--trained-only", action="store_true", help="skip the base model")
    p.add_argument("-q", "--question", default=None, help="ask one question and exit")
    p.add_argument("--interactive", action="store_true", help="type questions in a loop")
    p.add_argument("--max-new-tokens", type=int, default=384)
    args = p.parse_args()

    device, dtype = pick_device()
    print(f"[demo] device={device} dtype={dtype} | loading models "
          f"(first run downloads the ~3GB base from HuggingFace)...", flush=True)
    tok, trained = load(args.model, args.adapter or None, device, dtype)
    tok_b, base = ((None, None) if args.trained_only else load(args.model, None, device, dtype))

    # one-off question
    if args.question:
        print("\n" + "=" * 80 + f"\nQ: {args.question}\n" + "=" * 80)
        if base is not None:
            show("BASE (no RL)", tok_b, base, device, args.question, None, args.max_new_tokens)
        show("TRAINED (RL)", tok, trained, device, args.question, None, args.max_new_tokens)
        return

    # curated showcase with correct/wrong + tally
    base_score = trained_score = 0
    for q, gold in SHOWCASE:
        print("\n" + "=" * 80 + f"\nQ: {q}\n(expected: {gold})\n" + "=" * 80)
        if base is not None:
            base_score += int(bool(show("BASE (no RL)", tok_b, base, device, q, gold, args.max_new_tokens)))
        trained_score += int(bool(show("TRAINED (RL)", tok, trained, device, q, gold, args.max_new_tokens)))
    n = len(SHOWCASE)
    print("\n" + "=" * 80)
    if base is not None:
        print(f"SCORE  base {base_score}/{n}   →   trained {trained_score}/{n}")
    else:
        print(f"SCORE  trained {trained_score}/{n}")
    print("=" * 80)

    if args.interactive:
        print("\n[interactive] type a question (Ctrl-C / Ctrl-D to quit):")
        try:
            while True:
                q = input("\nQ> ").strip()
                if q:
                    show("TRAINED (RL)", tok, trained, device, q, None, args.max_new_tokens)
        except (KeyboardInterrupt, EOFError):
            print("\nbye")


if __name__ == "__main__":
    main()
