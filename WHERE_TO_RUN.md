# WHERE_TO_RUN.md — Exact commands for each environment

> Copy-paste these. Every command is tested for the environment listed.
> Do NOT run GRPO on Colab free T4 — it will disconnect mid-run.

---

## Step 0 — One-time: get your API keys

| Key | Where to get it | Required? |
|-----|----------------|-----------|
| `DEEPINFRA_API_KEY` | https://deepinfra.com → API Keys | **Yes** (teacher labels) |
| `WANDB_API_KEY` | https://wandb.ai → Settings → API Keys | **Yes** (tracking) |
| `HF_TOKEN` | https://huggingface.co → Settings → Access Tokens (write) | **Yes** (checkpoint backup) |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys | No (fallback only) |

Create `.env` in the repo root:
```
DEEPINFRA_API_KEY=your_key_here
WANDB_API_KEY=your_key_here
HF_TOKEN=your_key_here
```

---

## ENVIRONMENT A — Colab Free T4
**Use for:** Steps 1, 2, 3, 5 (everything except GRPO training)

### Open a new Colab notebook, then:

```python
# Cell 1 — clone repo and setup
!git clone https://github.com/YOUR_ORG/adaptivethink.git
%cd adaptivethink
!bash scripts/01_setup.sh
```

```python
# Cell 2 — set secrets (do NOT hardcode in notebook)
import os
os.environ["DEEPINFRA_API_KEY"] = "..."   # paste your key
os.environ["WANDB_API_KEY"]     = "..."
os.environ["HF_TOKEN"]          = "..."
# Write .env so scripts can read it
with open(".env","w") as f:
    for k in ["DEEPINFRA_API_KEY","WANDB_API_KEY","HF_TOKEN"]:
        f.write(f"{k}={os.environ[k]}\n")
```

```python
# Cell 3 — build data pool (downloads datasets, ~5 min)
import json, sys; sys.path.insert(0,"src")
from adaptivethink.data.loaders import build_verifier_pool, build_verifier_eval
import pathlib; pathlib.Path("data").mkdir(exist_ok=True)
pool = build_verifier_pool()
ev   = build_verifier_eval()
with open("data/verifier_pool.jsonl","w") as f: [f.write(json.dumps(r)+"\n") for r in pool]
with open("data/verifier_eval.jsonl","w") as f: [f.write(json.dumps(r)+"\n") for r in ev]
print(f"Pool: {len(pool)} | Eval: {len(ev)}")
```

```python
# Cell 4 — generate teacher labels (~2-4 h, resume-safe)
# Cost estimate: 12k items × 3 calls × ~250 tokens × $0.14/M ≈ $1.26
!python src/adaptivethink/data/teacher_labels.py \
  --pool data/verifier_pool.jsonl \
  --out  data/teacher_labels.jsonl \
  --db   data/teacher_cache.sqlite \
  --provider deepinfra \
  --max-cost-usd 10
```

```python
# Cell 5 — also label the eval set (needed for verifier training)
!python src/adaptivethink/data/teacher_labels.py \
  --pool data/verifier_eval.jsonl \
  --out  data/verifier_eval_labelled.jsonl \
  --db   data/teacher_cache.sqlite \
  --provider deepinfra \
  --max-cost-usd 2
```

```python
# Cell 6 — SAVE DATA TO GOOGLE DRIVE before session dies
from google.colab import drive
drive.mount('/content/drive')
!cp -r data/ /content/drive/MyDrive/adaptivethink_data/
print("Data backed up to Drive")
```

```python
# Cell 7 — train verifier (~2-3 h on T4, resumes if interrupted)
!python src/adaptivethink/verifier/train.py \
  --train data/teacher_labels.jsonl \
  --eval  data/verifier_eval_labelled.jsonl \
  --out   outputs/verifier-400m/best.pt \
  --epochs 3 --batch 32 --lr 2e-5
```

```python
# Cell 8 — push verifier to HF Hub (so Vast.ai can download it)
from huggingface_hub import HfApi
import os, shutil
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("statezero/verifier-400m", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj="outputs/verifier-400m/best.pt",
    path_in_repo="best.pt",
    repo_id="statezero/verifier-400m",
)
print("Verifier uploaded to HF Hub")
```

---

## ENVIRONMENT B — Vast.ai RTX 4090
**Use for:** Step 4 (GRPO training, ~36 h)
**Cost:** ~$0.35–0.50/hr × 40 h ≈ $14–20

### Setup Vast.ai instance:
1. Go to https://vast.ai/console/create/
2. Select: **RTX 4090**, **PyTorch 2.4 CUDA 12.1** template, **50 GB disk**
3. Click "Rent" → wait ~2 min for it to start
4. Click "Connect" → copy the SSH command, e.g.:
   `ssh -p 12345 root@123.45.67.89`

### On the Vast.ai box (SSH):

```bash
# One-time setup
git clone https://github.com/YOUR_ORG/adaptivethink.git
cd adaptivethink
bash scripts/01_setup.sh

# Set keys
cat > .env << 'EOF'
DEEPINFRA_API_KEY=your_key
WANDB_API_KEY=your_key
HF_TOKEN=your_key
EOF

# Download verifier from HF Hub
mkdir -p outputs/verifier-400m
python -c "
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download('statezero/verifier-400m', 'best.pt', token=os.environ['HF_TOKEN'])
shutil.copy(p, 'outputs/verifier-400m/best.pt')
print('Verifier downloaded')
"

# Download labelled training data from HF Hub (upload it there from Colab first)
# OR re-run teacher labelling here (it resumes from cache):
# python src/adaptivethink/data/teacher_labels.py --pool data/verifier_pool.jsonl ...

# Prepare GRPO training data (GSM8K train with difficulty scores pre-cached)
python -c "
import json, sys; sys.path.insert(0,'src')
from adaptivethink.data.loaders import load_gsm8k
from adaptivethink.verifier.model import load_verifier
from transformers import AutoTokenizer
import pathlib; pathlib.Path('data').mkdir(exist_ok=True)

items = load_gsm8k('train')
verifier, vtok = load_verifier('outputs/verifier-400m/best.pt', 'cuda')
difficulties = verifier.score([it['question'] for it in items], vtok)
with open('data/gsm8k_train_labelled.jsonl','w') as f:
    for it, d in zip(items, difficulties):
        f.write(json.dumps({**it, 'difficulty': d}) + '\n')
print(f'Saved {len(items)} labelled items')
"

# Launch GRPO in tmux (survives SSH disconnect)
tmux new -s grpo
bash scripts/04_train_grpo_router.sh 0   # seed=0
# Ctrl-b d  to detach
# tmux attach -t grpo  to reattach
```

```bash
# Monitor from your laptop (separate terminal):
# wandb dashboard: https://wandb.ai/YOUR_ORG/adaptivethink
# Or tail the log:
ssh -p PORT root@IP "tail -f adaptivethink/logs/grpo_seed0.log"
```

```bash
# After seed-0 finishes (~36 h), run seeds 1 and 2:
bash scripts/04_train_grpo_router.sh 1
bash scripts/04_train_grpo_router.sh 2
```

---

## ENVIRONMENT C — Colab Free T4 (again, after GRPO)
**Use for:** Steps 5–7 (eval, quantise, demo prep)

```python
# Cell 1 — restore data from Drive
from google.colab import drive; drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/adaptivethink_data/ data/
```

```python
# Cell 2 — download trained router from HF Hub
from huggingface_hub import snapshot_download
import os
snapshot_download(
    "statezero/router-1p5b-seed0",
    local_dir="outputs/router-seed0",
    token=os.environ["HF_TOKEN"]
)
```

```python
# Cell 3 — run eval (Pass@1 on GSM8K test, ~20 min on T4)
!python src/adaptivethink/eval/run_eval.py \
  --model outputs/router-seed0 \
  --bench gsm8k \
  --out results/eval_seed0.json
```

```python
# Cell 4 — quantise to GGUF (CPU, ~10 min)
!bash scripts/07_quantize_gguf.sh outputs/router-seed0 outputs/router-q4km.gguf
```

---

## Quick reference: which step runs where

| Step | What | Where | Time |
|------|------|-------|------|
| 1 | Build data pool | Colab T4 free | 5 min |
| 2 | Teacher labels | Colab T4 free | 2–4 h (API-bound) |
| 3 | Train verifier | Colab T4 free | 2–3 h GPU |
| **4** | **GRPO router training** | **Vast.ai RTX 4090** | **36 h GPU** |
| 5 | Eval baselines | Colab T4 free | 1–2 h |
| 6 | Quantise GGUF | Colab T4 free (CPU) | 10 min |
| 7 | On-device bench | Galaxy phone + ADB | 1 h |

---

## If Colab disconnects mid-teacher-labelling

The SQLite cache (`data/teacher_cache.sqlite`) saves every completed item.
Just re-run the same command — it skips already-cached items automatically.
**Always back up `teacher_cache.sqlite` to Drive after each session.**

## If Vast.ai GRPO run disconnects

Checkpoints are saved every 50 steps to `outputs/router-seedX/checkpoint-*/`
AND pushed to HF Hub. Re-run `bash scripts/04_train_grpo_router.sh 0` —
it auto-resumes from the latest checkpoint.
