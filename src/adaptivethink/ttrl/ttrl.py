"""
Test-Time RL add-on (optional Idea A) — arXiv:2504.16084.

On *unlabeled* data the model samples a group of completions, takes a
confidence-weighted majority vote of the extracted answers as a pseudo-label,
and rewards agreement with that pseudo-label. GRPO then sharpens toward the
self-consistent answer with no ground truth.

Entropy mitigation (Clip-Cov spirit, arXiv:2505.22617): a small entropy bonus
plus a low KL-beta keeps the sampling distribution from collapsing onto a single
mode — the dominant TTRL failure mode for SLMs.

Assumes per_device_train_batch_size=1 so each reward call sees one prompt's
group of `num_generations` completions.
"""
import argparse
import os

import torch
from dotenv import load_dotenv

from adaptivethink.ttrl.vote import majority_vote_reward

load_dotenv()

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def _make_ttrl_reward(lambda_tok, conf_weight):
    def reward_fn(completions, prompts=None, **kwargs):
        return majority_vote_reward(completions, lambda_tok=lambda_tok, conf_weight=conf_weight)
    return reward_fn


def _load_unlabeled(n, seed):
    from datasets import Dataset
    from adaptivethink.data.loaders import load_mmlu
    from adaptivethink.router.prompt import make_prompt

    items = load_mmlu(seed=seed, n=n)
    return Dataset.from_list([{"prompt": make_prompt(it["question"])} for it in items])


def train(args):
    torch.manual_seed(args.seed)

    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(project="adaptivethink", name=f"ttrl_seed{args.seed}", config=vars(args))
    else:
        os.environ["WANDB_MODE"] = "offline"

    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            MODEL_NAME, max_seq_length=args.max_seq_len, load_in_4bit=True, dtype=None)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        else:
            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=32, lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map="auto")
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

    dataset = _load_unlabeled(args.n, args.seed)

    from trl import GRPOConfig, GRPOTrainer
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=1,          # one prompt per group (required)
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_generations=args.group_size,
        max_prompt_length=args.max_seq_len // 2,
        max_completion_length=args.max_seq_len // 2,
        temperature=0.8,                         # higher temp -> vote diversity
        top_p=0.95,
        beta=args.kl_beta,                       # low KL: don't anchor too hard
        save_steps=50,
        save_total_limit=3,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        seed=args.seed,
        use_vllm=torch.cuda.get_device_properties(0).total_memory / 1e9 >= 22
        if torch.cuda.is_available() else False,
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tok,   # TRL >=0.24 dropped the `tokenizer=` kwarg
        reward_funcs=_make_ttrl_reward(args.lambda_tok, not args.no_conf_weight),
        args=cfg, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[ttrl] saved -> {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="warm-start from router adapter")
    p.add_argument("--output-dir", default="outputs/ttrl-seed0")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=1e-3)
    p.add_argument("--lambda-tok", type=float, default=1e-3)
    p.add_argument("--no-conf-weight", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())
