"""SFT cold-start for Strategy 1 — install the <think>/<answer> reasoning format.

A SHORT supervised fine-tune (a few thousand DeepSeek-R1 CoT traces) that teaches a
sub-3B base model the ``<think>...</think><answer>...</answer>`` output convention
BEFORE Dr.GRPO RL.  This is the Day 3-5 cold-start of the staged plan: it does not
need to raise accuracy on its own, it just makes the format reliable so the GRPO
format reward is not fighting the model from step 0.

Entry point (per the project interface contract)::

    python -m adaptivethink.rl.sft_coldstart \
        --model Qwen/Qwen2.5-3B-Instruct \
        --traces simplescaling/s1K-1.1 \
        --out outputs/sft \
        --steps 500 \
        --max-seq-len 4096

Backend: Unsloth + TRL ``SFTTrainer`` with QLoRA (4-bit) when Unsloth is available;
otherwise a transformers + peft + bitsandbytes fallback path with the same
``SFTTrainer``.  A LoRA adapter (and tokenizer) is saved to ``--out``.

DESIGN NOTES
------------
* Top-level imports are LIGHT on purpose.  Every heavy dependency (torch, datasets,
  trl, unsloth, transformers, peft) is imported INSIDE a function, so this module
  ``import``s and ``py_compile``s in an environment without the RL stack installed.
* IMPORTANT runtime ordering: ``import unsloth`` must happen BEFORE trl/transformers
  so Unsloth's patches apply; the loader below honours that.
* The ``<think>/<answer>`` format and system prompt are kept IDENTICAL to the RL
  trainer's so the format installed here matches what GRPO (and eval) expect.  We
  reuse ``adaptivethink.rl.data.SYSTEM_PROMPT`` when that sibling module is present,
  else fall back to a local copy of the same string.
* Answer matching is NOT reimplemented: when ``--validate-answers`` is on we reuse
  ``extract_answer`` + ``_answers_match`` from ``adaptivethink.router.reward`` to drop
  traces whose final answer does not match the dataset gold (keeps the cold-start
  clean without a second, divergent matcher).
"""
from __future__ import annotations

import argparse
import re

# ---------------------------------------------------------------------------
# Format / prompt constants (kept in sync with the Dr.GRPO trainer + eval).
# ---------------------------------------------------------------------------

# Local fallback copy of the system prompt.  At runtime we prefer the shared
# constant from adaptivethink.rl.data so SFT and RL never drift; this literal is
# used only if that sibling module is not importable.
_SYSTEM_PROMPT_FALLBACK = (
    "Reason step by step inside <think></think>, "
    "then give the final answer inside <answer></answer>."
)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

# Recognised trace datasets and which fields carry the reasoning body / final
# answer / question.  All values verified against the DATA RESEARCH spec.
#   s1K-1.1            -> DeepSeek-R1 deepseek_thinking_trajectory / deepseek_attempt
#   OpenR1-Math-220k   -> R1 'solution' trace + 'answer'; also ships chat 'messages'
#   OpenThoughts-114k  -> default config is ShareGPT chat in 'conversations'
_TRACE_SCHEMAS = {
    "simplescaling/s1K-1.1": dict(
        question="question",
        think="deepseek_thinking_trajectory",
        answer="deepseek_attempt",
        gold="solution",
    ),
    "simplescaling/s1K": dict(  # legacy (Gemini traces) — supported but discouraged
        question="question",
        think="thinking_trajectories",
        answer="attempt",
        gold="solution",
    ),
    "open-r1/OpenR1-Math-220k": dict(
        question="problem",
        think="solution",
        answer="answer",
        gold="answer",
        chat_field="messages",
    ),
    "open-thoughts/OpenThoughts-114k": dict(
        chat_field="conversations",
    ),
}

# Default trace dataset for the short cold-start (right-sized at ~1k rows, R1 traces).
DEFAULT_TRACES = "simplescaling/s1K-1.1"


def get_system_prompt() -> str:
    """Return the shared system prompt, falling back to the local copy.

    Imported lazily so a missing sibling module never breaks ``import``/compile.
    """
    try:
        from adaptivethink.rl import data as _rl_data  # type: ignore

        prompt = getattr(_rl_data, "SYSTEM_PROMPT", None)
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    except Exception:  # pragma: no cover - sibling module optional at SFT time
        pass
    return _SYSTEM_PROMPT_FALLBACK


# ---------------------------------------------------------------------------
# Trace -> {prompt, target text} mapping (pure, dependency-free, unit-testable).
# ---------------------------------------------------------------------------


def _coerce_text(value) -> str:
    """Best-effort flatten of a trace field to a single string.

    DeepSeek-R1 trace fields are usually strings, but some mirrors ship a list of
    segments; join those.  ``None`` becomes ``""``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_coerce_text(v) for v in value)
    return str(value)


def build_sft_text(question: str, think: str, answer: str, system_prompt: str) -> str:
    """Compose one SFT training example in the chat-templated body form.

    The returned string is the *assistant target* wrapped in the project's
    <think>/<answer> convention.  The trainer applies the chat template around it;
    here we only own the reasoning-format wrapping so it is exact + testable.
    """
    think_body = _coerce_text(think).strip()
    answer_body = _coerce_text(answer).strip()
    # Defensive: if the trace already carries the tags, do not double-wrap.
    if think_body.startswith(THINK_OPEN):
        think_block = think_body
    else:
        think_block = f"{THINK_OPEN}{think_body}{THINK_CLOSE}"
    if answer_body.startswith(ANSWER_OPEN):
        answer_block = answer_body
    else:
        answer_block = f"{ANSWER_OPEN}{answer_body}{ANSWER_CLOSE}"
    return f"{think_block}\n{answer_block}"


def _chat_to_text(conversation) -> str | None:
    """Extract the assistant turn from a ShareGPT/OpenAI-style conversation list.

    Returns the assistant content already containing the model's reasoning, or
    ``None`` if no assistant turn is present.  Used for chat-formatted datasets
    (OpenThoughts-114k 'conversations', OpenR1 'messages').
    """
    if not isinstance(conversation, (list, tuple)):
        return None
    for turn in reversed(conversation):
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from")
        if role in ("assistant", "gpt"):
            content = turn.get("content")
            if content is None:
                content = turn.get("value")
            return _coerce_text(content)
    return None


def _resolve_schema(traces: str) -> dict:
    """Return the field schema for a known dataset id, else an empty dict.

    Matches on a suffix so local paths like ``/data/s1K-1.1`` still resolve.
    """
    if traces in _TRACE_SCHEMAS:
        return _TRACE_SCHEMAS[traces]
    for known, schema in _TRACE_SCHEMAS.items():
        short = known.split("/")[-1]
        if traces.endswith(known) or traces.endswith(short):
            return schema
    return {}


# ---------------------------------------------------------------------------
# Dataset construction (lazy datasets import).
# ---------------------------------------------------------------------------


def build_sft_dataset(
    traces: str,
    *,
    config: str | None,
    split: str,
    max_examples: int,
    validate_answers: bool,
    seed: int,
):
    """Load a trace dataset and map it to a single ``text`` column for SFT.

    Returns a ``datasets.Dataset`` with one column, ``text`` — the full prompt +
    assistant target rendered through the model chat template happens later in
    ``train`` (SFTTrainer handles templating); here we attach the raw
    {system, user, assistant} content as a ``messages`` column so the trainer can
    apply the chat template uniformly across datasets.
    """
    from datasets import load_dataset  # heavy import, kept local

    system_prompt = get_system_prompt()
    schema = _resolve_schema(traces)

    load_kwargs = {}
    if config:
        load_kwargs["name"] = config
    try:
        ds = load_dataset(traces, split=split, **load_kwargs)
    except Exception as exc:  # surface a clear, actionable error
        raise RuntimeError(
            f"Failed to load trace dataset {traces!r} (config={config!r}, "
            f"split={split!r}): {exc}. Known ids: {sorted(_TRACE_SCHEMAS)}"
        ) from exc

    # Optional cap + shuffle so a 220k set can be subsampled to a short cold-start.
    if max_examples and len(ds) > max_examples:
        ds = ds.shuffle(seed=seed).select(range(max_examples))

    chat_field = schema.get("chat_field")
    q_field = schema.get("question")
    think_field = schema.get("think")
    ans_field = schema.get("answer")
    gold_field = schema.get("gold")

    matcher = _get_answer_matcher() if validate_answers else None
    columns = ds.column_names

    def _to_messages(ex):
        # 1) Prefer a ready chat field if the dataset ships one AND we have no
        #    explicit think/answer fields to rebuild the tagged format from.
        assistant = None
        user = None
        if chat_field and chat_field in ex and not (think_field and ans_field):
            assistant = _chat_to_text(ex[chat_field])
            # For pure chat sets, recover the user turn too if present.
            for turn in ex[chat_field] or []:
                if isinstance(turn, dict) and (turn.get("role") in ("user", "human")
                                               or turn.get("from") in ("user", "human")):
                    user = _coerce_text(turn.get("content") or turn.get("value"))
                    break
        # 2) Otherwise rebuild from think/answer/question fields into the tags.
        if assistant is None and q_field and think_field and ans_field:
            user = _coerce_text(ex.get(q_field, ""))
            assistant = build_sft_text(
                user, ex.get(think_field, ""), ex.get(ans_field, ""), system_prompt
            )
        if not assistant:
            return {"messages": None}
        if not user:
            user = ""
        # Optional answer validation: drop traces whose final answer disagrees
        # with the dataset gold (reuses router.reward matcher — no reimpl).
        if matcher is not None and gold_field and gold_field in ex:
            gold = _coerce_text(ex.get(gold_field, ""))
            if gold and not matcher(assistant, gold):
                return {"messages": None}
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }

    mapped = ds.map(_to_messages, remove_columns=columns)
    mapped = mapped.filter(lambda ex: ex["messages"] is not None)
    if len(mapped) == 0:
        raise RuntimeError(
            f"No usable SFT examples built from {traces!r}. Check the field schema "
            f"or pass --traces with a known id ({sorted(_TRACE_SCHEMAS)})."
        )
    print(f"[sft] Built {len(mapped)} SFT examples from {traces} (split={split}).")
    return mapped


def _get_answer_matcher():
    """Return a ``fn(completion_text, gold) -> bool`` reusing router.reward.

    Imported lazily; reuses the robust ``extract_answer`` + ``_answers_match`` so we
    never maintain a second answer-matching implementation.
    """
    from adaptivethink.router.reward import extract_answer, _answers_match

    answer_re = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def _match(completion_text: str, gold: str) -> bool:
        m = answer_re.search(completion_text)
        candidate = m.group(1).strip() if m else completion_text
        pred = extract_answer(candidate) or extract_answer(completion_text) or candidate
        return bool(pred is not None and _answers_match(pred, gold))

    return _match


# ---------------------------------------------------------------------------
# Model loading (Unsloth QLoRA, with transformers+peft fallback).
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(model_name: str, max_seq_len: int, lora_rank: int):
    """Load a 4-bit QLoRA model + tokenizer.

    Tries Unsloth first (``import unsloth`` BEFORE trl/transformers, per the deps
    research import-order gotcha); falls back to transformers + peft + bnb.
    Returns ``(model, tokenizer, backend)`` where backend is "unsloth"|"peft".
    """
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    try:
        import unsloth  # noqa: F401  (must precede trl/transformers imports)
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0,
            target_modules=target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=0,
        )
        print(f"[sft] Loaded {model_name} via Unsloth (4-bit QLoRA, r={lora_rank}).")
        return model, tokenizer, "unsloth"
    except ImportError:
        print("[sft][warn] Unsloth unavailable — falling back to transformers+peft+bnb.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    print(f"[sft] Loaded {model_name} via transformers+peft (4-bit QLoRA, r={lora_rank}).")
    return model, tokenizer, "peft"


# ---------------------------------------------------------------------------
# Training driver.
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the SFT cold-start and save the adapter to ``args.out``."""
    import os
    from pathlib import Path

    import torch

    torch.manual_seed(args.seed)

    model, tokenizer, backend = _load_model_and_tokenizer(
        args.model, args.max_seq_len, args.lora_rank
    )

    dataset = build_sft_dataset(
        args.traces,
        config=args.config,
        split=args.split,
        max_examples=args.max_examples,
        validate_answers=args.validate_answers,
        seed=args.seed,
    )

    # Render chat messages -> a single 'text' column via the model chat template,
    # so SFTTrainer trains on the full templated sequence (prompt + target).
    def _apply_template(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(_apply_template, remove_columns=["messages"])

    from trl import SFTConfig, SFTTrainer

    sft_cfg = SFTConfig(
        output_dir=args.out,
        max_steps=args.steps if args.steps > 0 else -1,
        num_train_epochs=args.epochs if args.steps <= 0 else 1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=args.max_seq_len,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        seed=args.seed,
        report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_cfg,
        train_dataset=dataset,
    )

    resume = _find_resume_checkpoint(args.out)
    if resume:
        print(f"[sft][resume] Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[sft][done] Saved cold-start adapter ({backend}) to {args.out}")


def _find_resume_checkpoint(output_dir: str) -> str | None:
    """Return the highest-numbered ``checkpoint-*`` under ``output_dir`` (resume)."""
    import glob

    ckpts = glob.glob(f"{output_dir}/checkpoint-*")
    if not ckpts:
        return None
    try:
        return sorted(ckpts, key=lambda p: int(p.split("-")[-1]))[-1]
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser (matches the interface contract)."""
    p = argparse.ArgumentParser(
        prog="python -m adaptivethink.rl.sft_coldstart",
        description="SFT cold-start: install the <think>/<answer> format before Dr.GRPO.",
    )
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model id (fallback: Qwen/Qwen2.5-1.5B-Instruct).",
    )
    p.add_argument(
        "--traces", default=DEFAULT_TRACES,
        help=("HF dataset id or local path of CoT traces. Known: "
              "simplescaling/s1K-1.1 (default), open-r1/OpenR1-Math-220k, "
              "open-thoughts/OpenThoughts-114k."),
    )
    p.add_argument("--out", default="outputs/sft", help="Adapter output directory.")
    p.add_argument(
        "--steps", type=int, default=500,
        help="Max training steps (a SHORT cold-start; <=0 uses --epochs instead).",
    )
    p.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length.")
    # Secondary knobs (sensible defaults; not part of the core contract).
    p.add_argument("--config", default=None,
                   help="Dataset config/subset name (e.g. 'default' for OpenR1/OpenThoughts).")
    p.add_argument("--split", default="train", help="Dataset split to load.")
    p.add_argument("--max-examples", type=int, default=4000,
                   help="Cap traces for a short cold-start (DATA RESEARCH: ~1-4k).")
    p.add_argument("--epochs", type=float, default=1.0,
                   help="Epochs when --steps<=0.")
    p.add_argument("--batch", type=int, default=1, help="Per-device train batch size.")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate (QLoRA SFT).")
    p.add_argument("--lora-rank", type=int, default=32, help="LoRA rank r.")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--validate-answers", dest="validate_answers", action="store_true",
        help="Drop traces whose <answer> mismatches the dataset gold "
             "(reuses router.reward.extract_answer + _answers_match).",
    )
    p.add_argument("--no-validate-answers", dest="validate_answers",
                   action="store_false")
    p.set_defaults(validate_answers=False)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
