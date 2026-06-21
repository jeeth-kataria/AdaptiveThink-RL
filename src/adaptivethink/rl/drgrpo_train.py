"""Dr.GRPO RLVR trainer for small dense reasoning models (TRL GRPOTrainer).

Implements Strategy 1's RL stage: Dr.GRPO (constant length-normalization + no
per-group std division) with the small-model-safe DAPO trick (overlong
filtering), no KL, and a rule-based RLVR reward (correctness + small format).
No teacher API anywhere — the reward is pure exact-match and the curriculum
signal (CCDD) comes from the base model's OWN solve-rate.

Config knobs map VERBATIM to TRL GRPOConfig (ALGO research):
  * loss_type="dr_grpo"            — constant 1/(L*G) denom (kills length bias)
  * scale_rewards=False            — no per-group std div (kills difficulty bias)
  * beta=<--kl>  (default 0.0)     — KL off => no ref model loaded (saves memory)
  * epsilon / epsilon_high         — symmetric by default; --clip-higher sets 0.28
  * top_entropy_quantile=1.0       — entropy masking OFF (harmful for tiny models)
  * mask_truncated_completions=True — DAPO overlong filtering (HELPS tiny models)
  * num_generations=<--group-size> — G; effective batch must be divisible by it
  * max_completion_length=L        — Dr.GRPO's normalization constant + trunc cap

CCDD curriculum (the API-free novelty): a separate self-difficulty pass writes
data/self_difficulty.jsonl with rows {question, answer, difficulty}, where
difficulty = 1 - (base model's empirical solve-rate over K rollouts per item),
scored with the SAME verifier the reward uses. When --self-difficulty-file is
given, we DROP pool items whose solve_rate is 0.0 (unsolvable) or 1.0 (trivial)
— both yield zero-advantage GRPO groups. When absent, we train on the full pool.

NOTE (ALGO research pitfall): TRL's default loss_type is "dapo" (token-level,
which the cited ablation flags as harmful for sub-3B models). We ALWAYS set
loss_type explicitly. There is NO additive entropy bonus in TRL — ``--entropy``
maps to ``top_entropy_quantile`` (a token-mask), and for tiny models we keep it
at 1.0 unless the user lowers it.

Heavy imports (torch, trl, unsloth, transformers, peft, datasets) are inside
functions so the module imports / compiles without them installed.

Run:
  python -m adaptivethink.rl.drgrpo_train \
      --model Qwen/Qwen2.5-1.5B-Instruct --datasets gsm8k,strategyqa,aqua \
      --out outputs/grpo-seed0 --steps 400 --seed 0 \
      --loss dr_grpo --kl 0.0 --no-clip-higher --entropy 1.0 \
      --self-difficulty-file data/self_difficulty.jsonl \
      --group-size 8 --max-prompt-length 512 --max-completion-length 1024
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# Default model (interface contract). 3B/7B are one-flag --model switches; all
# Qwen2.5 share ChatML + the same 7 LoRA modules. Do NOT hardcode DeepSeek.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Default training datasets (the three IMPROVE targets).
DEFAULT_DATASETS = "gsm8k,strategyqa,aqua"

# Default CCDD self-difficulty file (curriculum signal; same schema the verifier
# consumes). Empty string => curriculum filter OFF (train on full pool).
DEFAULT_SELF_DIFFICULTY_FILE = "data/self_difficulty.jsonl"

# VRAM threshold (GB) for colocated vLLM rollouts (contract: >= 22 GB).
VLLM_VRAM_GB = 22.0
# LoRA target modules — shared by ALL Qwen2.5 dense models (1.5B/3B/7B).
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
                        "3B/7B are one-flag switches).")
    p.add_argument("--datasets", default=DEFAULT_DATASETS,
                   help=f"Comma-separated dataset names (default "
                        f"'{DEFAULT_DATASETS}').")
    p.add_argument("--out", required=True,
                   help="Output dir for the LoRA adapter + checkpoints "
                        "(e.g. outputs/grpo-seed0).")
    p.add_argument("--steps", type=int, default=400, help="Max training steps.")
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
    # CCDD curriculum: self-difficulty file (the API-free novelty). When given,
    # drop pool items whose solve_rate is 0.0 (unsolvable) or 1.0 (trivial).
    p.add_argument("--self-difficulty-file", dest="self_difficulty_file",
                   default=DEFAULT_SELF_DIFFICULTY_FILE,
                   help="data/self_difficulty.jsonl with rows "
                        "{question, answer, difficulty}. When present, applies "
                        "the CCDD curriculum filter (drop solve_rate 0.0/1.0). "
                        "Pass '' to disable and train on the full pool.")
    _add_bool_pair(p, "one-shot", default=False,
                   help_text="Tiny one-item-per-dataset subset (smoke pipeline).")
    p.add_argument("--max-prompt-length", dest="max_prompt_length",
                   type=int, default=512,
                   help="GRPOConfig max_prompt_length (Dr.GRPO 24h config).")
    p.add_argument("--max-completion-length", dest="max_completion_length",
                   type=int, default=1024,
                   help="GRPOConfig max_completion_length; also the Dr.GRPO 1/L "
                        "normalization constant + truncation cap.")
    p.add_argument("--group-size", type=int, default=8,
                   help="num_generations G per prompt.")
    # Auxiliary (not in the core contract string but needed for a runnable train).
    p.add_argument("--lr", type=float, default=1e-6,
                   help="Learning rate (Dr.GRPO 24h config).")
    p.add_argument("--batch", type=int, default=8,
                   help="per_device_train_batch_size.")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="gradient_accumulation_steps.")
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
            # Colocated vLLM rollout engine, Unsloth-managed (the canonical GRPO
            # path). enforce_eager=True disables vLLM's torch.compile, which
            # crashes on vLLM 0.19.1 + torch 2.10 ('SymInt' not subscriptable in
            # the rotary-embedding inductor graph). Eager is slightly slower but
            # rollouts still use fast PagedAttention.
            fast_inference=True,
            max_lora_rank=lora_r,
            gpu_memory_utilization=0.45,
            enforce_eager=True,
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


# ── CCDD self-difficulty resolution ───────────────────────────────────────────

def _resolve_self_difficulty_file(args) -> str | None:
    """Return the self-difficulty path to use, or None (curriculum OFF).

    The CCDD filter is applied only when the file is explicitly configured AND
    exists on disk. An unset/empty flag, or a missing file, falls back to the
    full pool (no curriculum), with a clear log line either way.
    """
    path = (args.self_difficulty_file or "").strip()
    if not path:
        print("[ccdd] No --self-difficulty-file; training on the FULL pool.")
        return None
    if not os.path.isfile(path):
        print(f"[ccdd] self-difficulty file not found ({path!r}); "
              "training on the FULL pool (run the self-difficulty pass first).")
        return None
    print(f"[ccdd] Curriculum filter ON via {path}: "
          "dropping items with solve_rate 0.0 (unsolvable) or 1.0 (trivial).")
    return path


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
          f"model={args.model} | datasets={names} | "
          f"loss={args.loss} | kl(beta)={args.kl} | "
          f"clip_higher={args.clip_higher} | entropy_q={args.entropy} | "
          f"group_size={args.group_size} | "
          f"max_prompt={args.max_prompt_length} | "
          f"max_completion={args.max_completion_length} | seed={args.seed}")

    max_prompt_len = args.max_prompt_length
    max_completion_len = args.max_completion_length
    # The model context must hold prompt + completion.
    max_seq_len = max_prompt_len + max_completion_len

    # wandb (offline if no key) — optional.
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb  # type: ignore

            wandb.init(project="adaptivethink-rl",
                       name=f"drgrpo_seed{args.seed}", config=vars(args))
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] init skipped: {exc!r}")

    model, tokenizer = _load_model(
        args.model, max_seq_len, args.lora_r, args.lora_alpha
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # CCDD curriculum: drop solve_rate 0.0/1.0 items via the self-difficulty file.
    self_difficulty_file = _resolve_self_difficulty_file(args)

    dataset = rl_data.build_dataset(
        names,
        split="train",
        seed=args.seed,
        one_shot=args.one_shot,
        max_items=args.max_train_items,
        self_difficulty_file=self_difficulty_file,
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
