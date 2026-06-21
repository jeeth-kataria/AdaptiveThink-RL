"""AdaptiveThink-RL demo — runs on a Mac (Apple Silicon / CPU), no CUDA/vLLM needed.

Loads Qwen2.5-1.5B-Instruct (base) + our RL LoRA adapter and answers questions in the
trained <think>/<answer> reasoning format. Use --compare to show base vs trained side
by side (great for the demo video — the base often fumbles a problem the RL model nails).

Install (a clean venv is fine — only 3 deps, all Mac-friendly):
    python3 -m venv .demo && source .demo/bin/activate
    pip install torch transformers peft accelerate

Run:
    python demo.py                                   # trained model, sample questions + interactive
    python demo.py --compare                         # base vs trained, side by side
    python demo.py -q "If 3 pens cost $6, how much do 7 pens cost?"
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# IDENTICAL to the training/eval prompt (adaptivethink.rl.data._build_prompt) so the
# model sees exactly what it was trained on.
SYSTEM = ("Reason step by step inside <think> and </think>, "
          "then give the final answer inside <answer> and </answer>.")

SAMPLES = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did she sell altogether in April and May?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
    "bolts in total does it take?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
    "babysitting. How much did she earn?",
    "Answer True or False. Would a pear sink in water?",
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
        model = model.merge_and_unload()          # bake the RL adapter into the weights
    return tok, model.to(device).eval()


def generate(tok, model, device, question: str, max_new_tokens: int) -> str:
    enc = tok(build_prompt(question), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/grpo-seed0-v2",
                   help="RL LoRA adapter dir (set '' to run the plain base model)")
    p.add_argument("--compare", action="store_true",
                   help="show base (no RL) vs trained, side by side")
    p.add_argument("-q", "--question", default=None, help="ask one question and exit")
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    device, dtype = pick_device()
    print(f"[demo] device={device} dtype={dtype} | loading models (first run downloads "
          f"the ~3GB base from HuggingFace)...", flush=True)

    tok, trained = load(args.model, args.adapter or None, device, dtype)
    tok_b, base = (load(args.model, None, device, dtype) if args.compare else (None, None))

    questions = [args.question] if args.question else SAMPLES
    for q in questions:
        print("\n" + "=" * 78 + f"\nQ: {q}\n" + "=" * 78)
        if base is not None:
            print("\n----- BASE (no RL) -----\n" + generate(tok_b, base, device, q, args.max_new_tokens))
            print("\n----- TRAINED (RL) -----\n" + generate(tok, trained, device, q, args.max_new_tokens))
        else:
            print(generate(tok, trained, device, q, args.max_new_tokens))

    if args.question is None:
        print("\n[interactive] type a question (Ctrl-C / Ctrl-D to quit):")
        try:
            while True:
                q = input("\nQ> ").strip()
                if q:
                    print(generate(tok, trained, device, q, args.max_new_tokens))
        except (KeyboardInterrupt, EOFError):
            print("\nbye")


if __name__ == "__main__":
    main()
