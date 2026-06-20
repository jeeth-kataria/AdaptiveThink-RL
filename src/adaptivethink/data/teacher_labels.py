"""
Teacher label generation — robust version.
- Correct DeepInfra model names
- Exponential backoff on API errors
- N_CALLS=3 (cheaper, still reliable)
- Resumes from SQLite cache across sessions
"""
import json, os, hashlib, sqlite3, time, argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TEACHER_PROMPT = """You are estimating problem difficulty for a 1.5B-parameter math/reasoning model (DeepSeek-R1-Distill-Qwen-1.5B).
Rate this question's difficulty for that model on a 0.0–1.0 scale:
  0.0 = trivial, single arithmetic step, no chain-of-thought needed
  1.0 = requires explicit multi-step chain-of-thought to have any chance

Be bimodal: most questions should be near 0.1 or near 0.9, not 0.5.
Output ONLY valid JSON: {{"difficulty": <float 0-1>, "reason": "<15 words max>"}}

Question: {question}"""

N_CALLS = 3  # 3 calls per item is sufficient and cheaper than 5

# Cost per 1M tokens (input+output combined rough estimate)
COST_PER_M = {"deepinfra": 0.14, "openai": 0.60, "together": 0.20}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _get_client(provider: str):
    from openai import OpenAI
    if provider == "deepinfra":
        return OpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url="https://api.deepinfra.com/v1/openai",
        ), "deepseek-ai/DeepSeek-V3"   # correct DeepInfra model ID
    elif provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o-mini"
    elif provider == "together":
        return OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        ), "Qwen/Qwen2.5-72B-Instruct-Turbo"
    raise ValueError(f"Unknown provider: {provider}")


def _query_once(client, model: str, question: str, retries: int = 3) -> float | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": TEACHER_PROMPT.format(question=question)}],
                temperature=0.2,
                max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            data = json.loads(raw)
            d = float(data["difficulty"])
            return max(0.0, min(1.0, d))
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [warn] API error (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    return None


def label_items(items: list[dict], db_path: str, provider: str, max_cost_usd: float = 50.0) -> list[dict]:
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, difficulty REAL)")
    db.commit()

    client, model = _get_client(provider)
    results = []
    total_calls = 0
    cost_per_m = COST_PER_M.get(provider, 0.20)

    for i, item in enumerate(items):
        key = _hash(item["question"] + model)
        row = db.execute("SELECT difficulty FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            results.append({**item, "difficulty": row[0]})
            continue

        scores = [s for _ in range(N_CALLS) if (s := _query_once(client, model, item["question"])) is not None]
        total_calls += N_CALLS

        d = sum(scores) / len(scores) if scores else 0.5
        db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, d))
        db.commit()
        results.append({**item, "difficulty": d})

        if (i + 1) % 200 == 0:
            est_cost = total_calls * 250 * cost_per_m / 1_000_000
            print(f"  {i+1}/{len(items)} labelled | est cost ${est_cost:.2f}")
            if est_cost > max_cost_usd:
                print(f"Cost guard hit. Stopping at {i+1} items.")
                break

        time.sleep(0.03)  # gentle rate limiting

    db.close()
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool",          default="data/verifier_pool.jsonl")
    p.add_argument("--out",           default="data/teacher_labels.jsonl")
    p.add_argument("--db",            default="data/teacher_cache.sqlite")
    p.add_argument("--provider",      default="deepinfra",
                   choices=["deepinfra", "openai", "together"])
    p.add_argument("--max-cost-usd",  type=float, default=50.0)
    args = p.parse_args()

    items = [json.loads(l) for l in open(args.pool)]
    print(f"Labelling {len(items)} items via {args.provider}...")
    labelled = label_items(items, args.db, args.provider, args.max_cost_usd)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in labelled:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(labelled)} labelled items → {args.out}")
