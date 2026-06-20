#!/usr/bin/env bash
# Day 1: 50-step GRPO smoke test (no verifier, simple correctness reward)
set -e
python - <<'EOF'
import torch, random
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import re, sys

random.seed(0); torch.manual_seed(0)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ANSWER_RE = re.compile(r"\\boxed\{([^}]+)\}")

# 50 toy items
items = [{"prompt": f"<|im_start|>user\nWhat is {i} + {i}?<|im_end|>\n<|im_start|>assistant\n",
          "answer": str(2*i)} for i in range(1, 51)]
ds = Dataset.from_list(items)

def reward_fn(completions, prompts, **kwargs):
    answers = kwargs["answer"]
    rewards = []
    for c, a in zip(completions, answers):
        m = ANSWER_RE.search(c)
        rewards.append(1.0 if m and m.group(1).strip() == a else 0.0)
    return rewards

model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=512, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj","v_proj"], use_gradient_checkpointing="unsloth")

cfg = GRPOConfig(output_dir="outputs/smoke", max_steps=50,
    per_device_train_batch_size=1, num_generations=4,
    max_prompt_length=256, max_completion_length=256,
    temperature=0.7, beta=5e-3, report_to="none",
    use_vllm=True, vllm_gpu_memory_utilization=0.40)

trainer = GRPOTrainer(model=model, tokenizer=tok, reward_funcs=reward_fn,
    args=cfg, train_dataset=ds)
trainer.train()
print("SMOKE TEST PASSED")
EOF
