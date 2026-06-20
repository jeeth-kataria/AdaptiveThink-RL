"""Dr.GRPO RLVR trainer for small dense reasoning models (TRL GRPOTrainer).

Implements Strategy 1's RL stage: Dr.GRPO (constant length-normalization + no
per-group std division) with the small-model-safe DAPO trick (overlong
filtering), no KL, and a rule-based RLVR reward (correctness + small format).

Config knobs map VERBATIM to TRL 0.24.0 GRPOConfig (ALGO research):
  * loss_type="dr_grpo"            — constant 1/(L*G) denom (kills length bias)
  * scale_rewards=False            — no per-group std div (kills difficulty bias)
  * beta=<--kl>  (default 0.0)     — KL off => no ref model loaded (saves memory)
  * epsilon / epsilon_high         — symmetric by default; --clip-higher sets 0.28
  * top_entropy_quantile=1.0       — entropy masking OFF (harmful for tiny models)
  * mask_truncated_completions=True — DAPO overlong filtering (HELPS tiny models)
  * num_generations=<--group-size> — G; effective batch must be divisible by it
  * max_completion_length=L        — Dr.GRPO's normalization constant + trunc cap

NOTE (ALGO research pitfall): TRL's default loss_type is now "dapo" (token-level,
which the cited ablation flags as harmful for sub-3B models). We ALWAYS set
loss_type explicitly. There is NO additive entropy bonus in TRL — ``--entropy``
maps to ``top_entropy_quantile`` (a token-mask), and for tiny models we keep it
at 1.0 unless the user lowers it.

Heavy imports (torch, trl, unsloth, transformers, peft, datasets) are inside
functions so the module imports / compiles without them installed.

Run:
  python -m adaptivethink.rl.drgrpo_train \
      --model Qwen/Qwen2.5-3B-Instruct --datasets gsm8k,strategyqa \
      --out outputs/grpo-seed0 --steps 1500 --seed 0 \
      --loss dr_grpo --kl 0.0 --no-clip-higher --entropy 1.0 \
      --difficulty-filter --no-one-shot --max-seq-len 2048 --group-size 8
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# Default + fallback models (interface contract).
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# VRAM threshold (GB) for colocated vLLM rollouts (contract: >= 22 GB).
VLLM_VRAM_GB = 22.0
# LoRA target modules for Qwen2.5 dense models.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── argparse (EXACT per interface contract) ───────────────────────────────────

def _add_bool_pair(parser: argparse.ArgumentParser, name: str, default: bool,
                   help_text: str) -> None:
    """Add a --flag / --no-flag boolean pair with an explicit default."""
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true",
                        default=default, help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false",
                        help=f"Disable --{name}.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m adaptivethink.rl.drgrpo_train",
        description="Dr.GRPO RLVR training for small dense reasoning models.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id (default {DEFAULT_MODEL}; "
                        f"fallback {FALLBACK_MODEL}).")
    p.add_argument("--datasets", default="gsm8k,strategyqa",
                   help="Comma-separated dataset names (gsm8k,strategyqa).")
    p.add_argument("--out", required=True,
                   help="Output dir for the LoRA adapter + checkpoints "
                        "(e.g. outputs/grpo-seed0).")
    p.add_argument("--steps", type=int, default=1500, help="Max training steps.")
    p.add_argument("--seed", type=int, default=0, help="Random seed (multi-seed).")
    p.add_argument("--loss", choices=["grpo", "dr_grpo"], default="dr_grpo",
                   help="Loss normalization variant (Dr.GRPO recommended).")
    p.add_argument("--kl", type=float, default=0.0,
                   help="KL beta (0.0 = no ref model / no KL; modern default).")
    _add_bool_pair(p, "clip-higher", default=False,
                   help_text="DAPO Clip-Higher (epsilon_high=0.28). Off for tiny "
                             "models; also inert at on-policy num_iterations=1.")
    p.add_argument("--entropy", type=float, default=1.0,
                   help="top_entropy_quantile (token-mask; 1.0=all tokens). "
                        "TRL has no additive entropy bonus.")
    _add_bool_pair(p, "difficulty-filter", default=False,
                   help_text="Offline difficulty filter: drop base-model "
                             "unsolvable/trivial items before RL.")
    _add_bool_pair(p, "one-shot", default=False,
                   help_text="Tiny one-item-per-dataset subset (smoke pipeline).")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Total seq len; split into prompt + completion caps.")
    p.add_argument("--group-size", type=int, default=8,
                   help="num_generations G per prompt.")
    # Auxiliary (not in the core contract string but needed for a runnable train).
    p.add_argument("--lr", type=float, default=1e-6,
                   help="Learning rate (GRPOConfig default for RL).")
    p.add_argument("--batch", type=int, default=8,
                   help="per_device_train_batch_size.")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="gradient_accumulation_steps.")
    p.add_argument("--difficulty-k", type=int, default=8,
                   help="Samples per item for the difficulty filter.")
    p.add_argument("--max-train-items", type=int, default=None,
                   help="Optional cap on training rows (after filter/shuffle).")
    p.add_argument("--save-steps", type=int, default=50,
                   help="Checkpoint cadence.")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    return p


# ── GPU / model loading ───────────────────────────────────────────────────────

def _gpu_gb() -> float:
    """Detected VRAM in GB (0.0 if no CUDA)."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _load_model(model_id: str, max_seq_len: int, lora_r: int, lora_alpha: int):
    """Load a 4-bit QLoRA model. Unsloth FastLanguageModel first, PEFT+bnb fallback.

    Returns (model, tokenizer).
    """
    import torch

    try:
        from unsloth import FastLanguageModel  # type: ignore

        print(f"[model] Unsloth 4-bit QLoRA: {model_id}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_len,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            target_modules=LORA_TARGETS,
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer
    except Exception as exc:  # noqa: BLE001 — unsloth absent OR runtime issue
        print(f"[model] Unsloth unavailable ({exc!r}); "
              f"falling back to transformers + PEFT + bitsandbytes.")

    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        task_type=TaskType.CAUSAL_LM, target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


# ── difficulty-filter wiring (frozen base sampling + repo verifier) ───────────

def _make_base_generate(model, tokenizer, max_new_tokens: int):
    """Build a (prompt, k) -> [k completions] sampler over the frozen base model.

    Used only for the offline difficulty filter. Sampling is at temperature ~0.9
    (DATA research) to estimate the base pass-rate honestly.
    """
    import torch

    def base_generate(prompt: str, k: int):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=k,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
                for o in out]

    return base_generate


def _make_verify():
    """Return a (completion_text, gold) -> bool verifier reusing the repo logic.

    Uses the SAME prediction + matching path as the training reward
    (``rewards.predicted_answer`` + the repo's ``_answers_match``) so the
    difficulty filter's notion of 'solved' is identical to the reward signal.
    """
    from adaptivethink.rl.rewards import predicted_answer
    from adaptivethink.router.reward import _answers_match

    def verify(text: str, gold: str) -> bool:
        pred = predicted_answer(text)
        return pred is not None and _answers_match(pred, str(gold))

    return verify


# ── checkpoint / resume ───────────────────────────────────────────────────────

def _find_resume_checkpoint(out_dir: str):
    """Latest checkpoint-N dir under out_dir, or None."""
    ckpts = glob.glob(os.path.join(out_dir, "checkpoint-*"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))


# ── GRPOConfig assembly ───────────────────────────────────────────────────────

def _build_grpo_config(args, max_prompt_len: int, max_completion_len: int,
                       use_vllm: bool, vllm_gpu_util: float):
    """Assemble a Dr.GRPO-tuned GRPOConfig from parsed args (ALGO research)."""
    from trl import GRPOConfig

    # Reward weights aligned with reward_funcs order [correctness, format].
    reward_weights = [1.0, 0.2]

    # Clip-Higher: epsilon_high=0.28 only when explicitly requested; else symmetric.
    epsilon_high = 0.28 if args.clip_higher else None

    cfg_kwargs = dict(
        output_dir=args.out,
        max_steps=args.steps,
        seed=args.seed,

        # (a) Dr.GRPO: constant normalization + drop per-group std.
        loss_type=args.loss,                 # "dr_grpo" | "grpo"
        scale_rewards=False,                 # == "none" (Dr.GRPO)
        max_completion_length=max_completion_len,   # L in the constant 1/(L*G)
        max_prompt_length=max_prompt_len,

        # (c) KL toggle (0.0 => no ref model loaded).
        beta=args.kl,

        # (b) Clip — symmetric unless --clip-higher.
        epsilon=0.2,
        epsilon_high=epsilon_high,

        # (d) Entropy token-mask (1.0 = all tokens; no additive bonus exists).
        top_entropy_quantile=args.entropy,

        # (e) DAPO overlong filtering (HELPS small models).
        mask_truncated_completions=True,

        # (f) Group + on-policy.
        num_generations=args.group_size,
        num_iterations=1,                    # mu=1 -> on-policy (clip is a no-op)
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        importance_sampling_level="token",

        # Reward composition.
        reward_weights=reward_weights,

        # Training hygiene.
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        temperature=0.9,
        top_p=0.95,
        logging_steps=10,
        log_completions=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
        use_vllm=use_vllm,
    )
    if use_vllm:
        # Colocate vLLM in the training process; modest GPU split for sub-3B.
        cfg_kwargs["vllm_mode"] = "colocate"
        cfg_kwargs["vllm_gpu_memory_utilization"] = vllm_gpu_util
    return GRPOConfig(**cfg_kwargs)


# ── train ─────────────────────────────────────────────────────────────────────

def train(args) -> None:
    import random

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from adaptivethink.rl import data as rl_data
    from adaptivethink.rl.rewards import make_reward_funcs

    names = rl_data.parse_datasets(args.datasets)

    gb = _gpu_gb()
    use_vllm = gb >= VLLM_VRAM_GB
    print(f"[config] VRAM={gb:.1f} GB | vLLM(colocate)={use_vllm} | "
          f"loss={args.loss} | kl(beta)={args.kl} | "
          f"clip_higher={args.clip_higher} | entropy_q={args.entropy} | "
          f"group_size={args.group_size} | seed={args.seed}")

    # Seq-len budget: split into prompt + completion caps.
    max_prompt_len = max(256, args.max_seq_len // 4)
    max_completion_len = args.max_seq_len - max_prompt_len

    # wandb (offline if no key) — optional.
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb  # type: ignore

            wandb.init(project="adaptivethink-rl",
                       name=f"drgrpo_seed{args.seed}", config=vars(args))
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] init skipped: {exc!r}")

    model, tokenizer = _load_model(
        args.model, args.max_seq_len, args.lora_r, args.lora_alpha
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional offline difficulty filter (frozen base sampling + repo verifier).
    base_generate = None
    verify = None
    if args.difficulty_filter:
        print(f"[filter] Difficulty filter ON (k={args.difficulty_k}): "
              "dropping base-unsolvable/trivial items.")
        base_generate = _make_base_generate(model, tokenizer,
                                            max_new_tokens=max_completion_len)
        verify = _make_verify()

    dataset = rl_data.build_dataset(
        names,
        split="train",
        seed=args.seed,
        one_shot=args.one_shot,
        max_items=args.max_train_items,
        base_generate=base_generate,
        verify=verify,
        difficulty_k=args.difficulty_k,
        drop_trivial=True,
    )
    print(f"[data] {len(dataset)} training rows from {names}")

    grpo_cfg = _build_grpo_config(
        args, max_prompt_len, max_completion_len,
        use_vllm=use_vllm, vllm_gpu_util=0.45,
    )

    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_funcs(use_format=True),
        args=grpo_cfg,
        train_dataset=dataset,
    )

    resume = _find_resume_checkpoint(args.out)
    if resume:
        print(f"[resume] Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[done] Saved LoRA adapter to {args.out}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
