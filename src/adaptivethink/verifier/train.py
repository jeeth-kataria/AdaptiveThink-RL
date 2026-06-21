"""Train the 400M difficulty verifier — robust version with per-epoch checkpointing."""
import json, argparse, os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from scipy.stats import spearmanr
import wandb

from .model import DifficultyVerifier

# num_workers=0 is safe on Colab; >0 can deadlock in forked processes
NUM_WORKERS = 0


class DifficultyDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 512):
        self.items = [json.loads(l) for l in open(path)]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        enc = self.tok(item["question"], truncation=True, max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "label": torch.tensor(float(item["difficulty"]), dtype=torch.float32),
        }


def _collate(batch):
    return {
        "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0),
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "label": torch.stack([b["label"] for b in batch]),
    }


def _eval_rho(model, loader, device) -> float:
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds.extend(torch.sigmoid(logits).cpu().tolist())
            gts.extend(batch["label"].tolist())
    return float(spearmanr(preds, gts).statistic)


def train(args):
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(project="adaptivethink", name="verifier", config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[verifier] Training on {device}")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = DifficultyVerifier().to(device)

    # Resume if checkpoint exists
    ckpt_path = Path(args.out)
    resume_path = ckpt_path.parent / "last.pt"
    start_epoch = 0
    if resume_path.exists():
        print(f"[verifier] Resuming from {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1

    train_ds = DifficultyDataset(args.train, tok)
    eval_ds  = DifficultyDataset(args.eval,  tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=_collate, num_workers=NUM_WORKERS)
    eval_dl  = DataLoader(eval_ds,  batch_size=64, collate_fn=_collate,
                          num_workers=NUM_WORKERS)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * (args.epochs - start_epoch)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * total_steps)), total_steps)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    best_rho = -1.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_dl:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(ids, mask)
            loss = mse_fn(torch.sigmoid(logits), labels) + 0.2 * bce_fn(logits, (labels > 0.5).float())
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            if use_wandb:
                wandb.log({"train/loss": loss.item()})

        rho = _eval_rho(model, eval_dl, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Spearman ρ = {rho:.4f}")
        if use_wandb:
            wandb.log({"eval/spearman_rho": rho, "epoch": epoch + 1})

        # Save last (for resume) and best
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, resume_path)
        if rho > best_rho:
            best_rho = rho
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best (ρ={rho:.4f}) saved to {ckpt_path}")

    print(f"[verifier] Done. Best Spearman ρ = {best_rho:.4f}")
    if best_rho < 0.5:
        print("WARNING: ρ < 0.5 — verifier quality is poor. Re-check the "
              "self-difficulty labels (mean difficulty, learnable fraction).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",  default="data/self_difficulty.jsonl")
    p.add_argument("--eval",   default="data/verifier_eval_labelled.jsonl")
    p.add_argument("--out",    default="outputs/verifier-400m/best.pt")
    p.add_argument("--epochs", type=int,   default=3)
    p.add_argument("--batch",  type=int,   default=32)
    p.add_argument("--lr",     type=float, default=2e-5)
    train(p.parse_args())
