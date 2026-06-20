"""
GRPO router training — robust version.
Auto-detects GPU VRAM and switches between:
  - RTX 4090 (24 GB): vLLM colocated, group_size=8, seq_len=2048, steps=1500
  - T4 (16 GB):       no vLLM, group_size=4, seq_len=1024, steps=800
Resumes from last checkpoint automatically.
"""
import json, os, argparse, random, glob
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# ── GPU profile ──────────────────────────────────────────────────────────────

def _gpu_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _auto_config(args) -> dict:
    """Return training kwargs tuned for available VRAM."""
    gb = _gpu_gb()
    print(f"[config] Detected {gb:.1f} GB VRAM")
    if gb >= 22:                          # RTX 4090 / A100
        return dict(
            use_vllm=True,
            vllm_gpu_memory_utilization=0.45,
            group_size=args.group_size or 8,
            max_seq_len=args.max_seq_len or 2048,
            steps=args.steps or 1500,
            batch=args.batch or 1,
            grad_accum=4,
        )
    elif gb >= 14:                        # T4 / V100-16G
        print("[config] T4 mode: disabling vLLM, reducing group_size/seq_len")
        return dict(
            use_vllm=False,
            vllm_gpu_memory_utilization=0.0,
            group_size=args.group_size or 4,
            max_seq_len=args.max_seq_len or 1024,
            steps=args.steps or 800,
            batch=args.batch or 1,
            grad_accum=8,
        )
    else:
        raise RuntimeError(f"Only {gb:.1f} GB VRAM — need at least 14 GB")


# ── Data ─────────────────────────────────────────────────────────────────────

def _load_data(path: str, verifier_model, verifier_tok, device: str):
    from datasets import Dataset
    from adaptivethink.router.prompt import make_prompt

    items = [json.loads(l) for l in open(path)]
    print(f"[data] Scoring {len(items)} items with verifier...")
    difficulties = verifier_model.score(
        [it["question"] for it in items], verifier_tok, device=device
    )
    rows = [
        {"prompt": make_prompt(it["question"]), "answer": it["answer"], "difficulty": d}
        for it, d in zip(items, difficulties)
    ]
    return Dataset.from_list(rows)


# ── Reward ────────────────────────────────────────────────────────────────────

def _make_reward_fn(lambda_tok: float, lambda_obey: float, tokenizer=None):
    from adaptivethink.router.reward import compute_rewards

    def reward_fn(completions, prompts=None, **kwargs):
        # TRL passes dataset columns as kwargs
        answers = kwargs.get("answer", [""] * len(completions))
        difficulties = kwargs.get("difficulty", [0.5] * len(completions))
        # Use the model's real tokenizer for an exact output-token count
        # (the brief's reward is in tokens, not words).
        token_counts = None
        if tokenizer is not None:
            token_counts = [len(tokenizer(c, add_special_tokens=False).input_ids)
                            for c in completions]
        return compute_rewards(completions, list(answers), list(difficulties),
                               token_counts=token_counts,
                               lambda_tok=lambda_tok, lambda_obey=lambda_obey)
    return reward_fn


# ── Resume helper ─────────────────────────────────────────────────────────────

def _find_resume_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(glob.glob(f"{output_dir}/checkpoint-*"),
                   key=lambda p: int(p.split("-")[-1]))
    return ckpts[-1] if ckpts else None


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed); torch.manual_seed(args.seed)

    cfg = _auto_config(args)
    print(f"[config] {cfg}")

    # Wandb — offline mode if no key
    if os.environ.get("WANDB_API_KEY"):
        wandb.init(project="adaptivethink", name=f"grpo_seed{args.seed}", config={**vars(args), **cfg})
    else:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="adaptivethink", name=f"grpo_seed{args.seed}", mode="offline")

    device = "cuda"

    # Load verifier
    from adaptivethink.verifier.model import load_verifier
    verifier, vtok = load_verifier(args.verifier_ckpt, device)

    # Load model
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_NAME, max_seq_length=cfg["max_seq_len"], load_in_4bit=True, dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=32, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
        )
    except ImportError:
        print("[warn] Unsloth not available, falling back to PEFT+BitsAndBytes")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
        lora_cfg = LoraConfig(r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lora_cfg)

    dataset = _load_data(args.data, verifier, vtok, device)

    from trl import GRPOConfig, GRPOTrainer

    resume = _find_resume_checkpoint(args.output_dir)
    if resume:
        print(f"[resume] Resuming from {resume}")

    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=cfg["steps"],
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=args.lr,
        num_generations=cfg["group_size"],
        max_prompt_length=cfg["max_seq_len"] // 2,
        max_completion_length=cfg["max_seq_len"] // 2,
        temperature=0.7,
        top_p=0.95,
        beta=args.kl_beta,
        save_steps=50,
        save_total_limit=3,
        push_to_hub=bool(os.environ.get("HF_TOKEN")),
        hub_model_id=f"statezero/router-1p5b-seed{args.seed}",
        hub_strategy="every_save",
        report_to="wandb",
        seed=args.seed,
        use_vllm=cfg["use_vllm"],
        **({"vllm_gpu_memory_utilization": cfg["vllm_gpu_memory_utilization"]}
           if cfg["use_vllm"] else {}),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,   # TRL >=0.24 dropped the `tokenizer=` kwarg
        reward_funcs=_make_reward_fn(args.lambda_tok, args.lambda_obey, tokenizer),
        args=grpo_cfg,
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=resume)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] Saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gsm8k_train_labelled.jsonl")
    p.add_argument("--verifier-ckpt", default="outputs/verifier-400m/best.pt")
    p.add_argument("--output-dir", default="outputs/router-seed0")
    p.add_argument("--steps", type=int, default=0)       # 0 = auto
    p.add_argument("--batch", type=int, default=0)        # 0 = auto
    p.add_argument("--group-size", type=int, default=0)   # 0 = auto
    p.add_argument("--max-seq-len", type=int, default=0)  # 0 = auto
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=5e-3)
    p.add_argument("--lambda-tok", type=float, default=3e-3)
    p.add_argument("--lambda-obey", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())
