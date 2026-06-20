"""
400M difficulty verifier: Qwen2.5-0.5B encoder + regression head.
Trained via MSE on teacher soft labels + auxiliary BCE on hard threshold.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DifficultyVerifier(nn.Module):
    def __init__(self, encoder_name="Qwen/Qwen2.5-0.5B-Instruct"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)  # (B,)

    @torch.no_grad()
    def score(self, questions: list[str], tokenizer, device="cuda", batch_size=32) -> list[float]:
        self.eval()
        scores = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            s = torch.sigmoid(self(enc.input_ids, enc.attention_mask))
            scores.extend(s.cpu().tolist())
        return scores


def load_verifier(ckpt_path: str, device="cuda"):
    model = DifficultyVerifier()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    return model, tok
