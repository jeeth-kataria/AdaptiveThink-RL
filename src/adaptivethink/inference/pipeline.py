"""
On-device adaptive reasoning pipeline (Stage 3).

Flow:  question -> 400M verifier -> difficulty d -> route -> answer.

Routing modes:
  - "model":      the RL-trained router decides via its first emitted token
                  (<think> / <no_think>). This is the headline system.
  - "threshold":  hard-route on the verifier score (d < threshold -> System-1,
                  else System-2). Matches the brief's Stage-3 description and is
                  the deterministic fallback.
  - "always_think" / "never_think": fixed-policy baselines for the Pareto chart.

Hard (System-2) branches use s1-style budget-forced extension: the first think
generation is given a *small* initial budget (`initial_think_tokens`); if the
model stops before producing a boxed answer and budget remains, we append a
short "Wait" continuation and top up toward `max_think_tokens`. Giving the first
pass the full budget would saturate the headroom check and forcing would never
fire — see bug #2 below.

Verifier-aware override (the whole point of the project): in "model" mode the
RL router self-routes on its first token, but the external difficulty `d` must
actually steer inference. If the model picks System-1 (no_think) yet the
verifier says the question is hard (`d >= override_threshold`), we override and
force a think pass. Without this, `d` is decorative in the headline mode.

Two backends:
  - "hf":   transformers (+ optional PEFT router adapter). Default for eval.
  - "gguf": llama-cpp-python over a Q4_K_M file. The on-device path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from adaptivethink.router.prompt import make_prompt, make_forced_prompt
from adaptivethink.router.reward import extract_answer, decision_from_response

BUDGET_FORCE_TEXT = "\nWait, let me re-check the reasoning step by step.\n"

# GGUF context window (must match GGUFBackend's n_ctx). Used to cap budget-force
# top-ups so prompt + completion + continuation can never silently overflow.
GGUF_N_CTX = 4096

# How many leading characters of a completion to scan for the routing tag.
# A stray BOS / leading space must not hide the <think> / <no_think> marker, so
# we look within a short window instead of requiring a strict startswith.
_DECISION_SCAN_CHARS = 16


@dataclass(frozen=True)
class RouteResult:
    question: str
    answer: str | None
    decision: str | None       # "think" | "no_think" | None
    difficulty: float
    n_tokens: int
    latency_s: float
    completion: str
    overridden: bool = False   # verifier forced a think pass over the model's choice


# ── Backends ───────────────────────────────────────────────────────────────────


class HFBackend:
    """transformers generation, optionally with a PEFT router adapter merged in."""

    def __init__(self, model_name: str, adapter_path: str | None, device: str, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype or torch.bfloat16
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
        self.model = model.to(device).eval()
        self.device = device
        # Real context window, so budget-forcing uses the model's true limit
        # rather than the conservative GGUF fallback.
        self.n_ctx = getattr(self.model.config, "max_position_embeddings", GGUF_N_CTX)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, greedy: bool) -> tuple[str, int]:
        import torch

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=temperature if not greedy else None,
                top_p=0.95 if not greedy else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0][enc.input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        return text, int(new_ids.shape[0])

    def count_tokens(self, text: str) -> int:
        """Exact prompt token length (used to budget against context)."""
        return len(self.tokenizer(text)["input_ids"])


class GGUFBackend:
    """llama-cpp-python over a quantised Q4_K_M file (the on-device path)."""

    def __init__(self, gguf_path: str, n_ctx: int = GGUF_N_CTX, n_gpu_layers: int = -1):
        from llama_cpp import Llama

        self.n_ctx = n_ctx
        self.llm = Llama(
            model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, greedy: bool) -> tuple[str, int]:
        out = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0 if greedy else temperature,
            top_p=0.95,
            echo=False,
        )
        text = out["choices"][0]["text"]
        n_tok = int(out.get("usage", {}).get("completion_tokens", 0)) or _rough_token_count(text)
        return text, n_tok

    def count_tokens(self, text: str) -> int:
        """Exact prompt token length via llama.cpp's tokenizer (budgeting)."""
        return len(self.llm.tokenize(text.encode("utf-8")))


def _rough_token_count(text: str) -> int:
    return max(1, len(text.split()))


def _detect_decision(completion: str) -> str | None:
    """Hardened routing-tag detection for "model" mode.

    `decision_from_response` (router.reward) requires a strict startswith, so a
    stray BOS token, leading newline, or space yields None and the override
    never gets a chance to fire. Here we first strip leading whitespace and
    common special-token noise, then look for the tag within the first
    ~16 chars before falling back to the canonical detector. Signature of the
    imported `decision_from_response` is left untouched.
    """
    if not completion:
        return None
    # Drop leading whitespace and any leading <|...|> special tokens (e.g. BOS).
    head = completion.lstrip()
    while head.startswith("<|"):
        end = head.find("|>")
        if end == -1:
            break
        head = head[end + 2:].lstrip()
    window = head[:_DECISION_SCAN_CHARS]
    if "<no_think>" in window:
        return "no_think"
    if "<think>" in window:
        return "think"
    # Fall back to the canonical detector on the stripped head.
    return decision_from_response(head)


# ── Pipeline ────────────────────────────────────────────────────────────────────


class AdaptivePipeline:
    def __init__(
        self,
        backend,
        verifier=None,
        verifier_tok=None,
        device: str = "cuda",
        route_mode: str = "model",
        threshold: float = 0.5,
        max_think_tokens: int = 1024,
        max_answer_tokens: int = 256,
        budget_force: bool = True,
        budget_force_max_rounds: int = 2,
        temperature: float = 0.7,
        verifier_override: bool = True,
        override_threshold: float = 0.6,
        initial_think_tokens: int | None = None,
    ):
        assert route_mode in {"model", "threshold", "always_think", "never_think"}
        self.backend = backend
        self.verifier = verifier
        self.verifier_tok = verifier_tok
        self.device = device
        self.route_mode = route_mode
        self.threshold = threshold
        self.max_think_tokens = max_think_tokens
        self.max_answer_tokens = max_answer_tokens
        self.budget_force = budget_force
        self.budget_force_max_rounds = budget_force_max_rounds
        self.temperature = temperature
        # Verifier-aware override (bug #1): in "model" mode, force a think pass
        # when the model self-routes to no_think but d says the question is hard.
        self.verifier_override = verifier_override
        self.override_threshold = override_threshold
        # Initial think budget (bug #2): the FIRST think generation gets a small
        # budget so _budget_extend has headroom to top up. Default reserves at
        # least half the budget (min(max_think_tokens//2, 512)) so budget-forcing
        # can never be a silent no-op even when max_think_tokens <= 512.
        self.initial_think_tokens = (
            min(max_think_tokens // 2, 512) if initial_think_tokens is None
            else min(initial_think_tokens, max_think_tokens)
        )

    # difficulty ---------------------------------------------------------------

    def difficulty(self, question: str) -> float:
        if self.verifier is None or self.verifier_tok is None:
            return 0.5
        return float(self.verifier.score([question], self.verifier_tok, device=self.device)[0])

    # routing ------------------------------------------------------------------

    def _resolve_decision(self, question: str, d: float) -> str | None:
        if self.route_mode == "always_think":
            return "think"
        if self.route_mode == "never_think":
            return "no_think"
        if self.route_mode == "threshold":
            return "think" if d >= self.threshold else "no_think"
        return None  # "model" mode: the model decides during generation

    # generation ---------------------------------------------------------------

    def answer(self, question: str, greedy: bool = True) -> RouteResult:
        t0 = time.time()
        d = self.difficulty(question)
        forced = self._resolve_decision(question, d)
        overridden = False
        prefilled = False  # whether `prompt` already contains the routing token

        if forced is None:
            # "model" mode: the RL router self-routes via its first token.
            prompt = make_prompt(question)
            # Cap the first pass at the small initial budget (a no_think answer
            # stops well short anyway) so budget-forcing has headroom to top up
            # toward max_think_tokens later — see bug #2.
            budget = self.initial_think_tokens
            completion, n_tok = self.backend.generate(prompt, budget, self.temperature, greedy)
            decision = _detect_decision(completion)

            # Bug #1 — verifier-aware override: the external difficulty signal
            # must enter the inference loop. If the model chose System-1 but the
            # verifier says the question is hard, force a think pass.
            if (
                self.verifier_override
                and decision != "think"
                and d >= self.override_threshold
            ):
                wasted = n_tok  # first-pass tokens are real cost — keep counting them
                prompt = make_forced_prompt(question, "think")
                completion, n_tok = self.backend.generate(
                    prompt, self.initial_think_tokens, self.temperature, greedy
                )
                completion = "<think>" + completion  # prefilled token is logical part
                n_tok += wasted
                decision = "think"
                overridden = True
                prefilled = True
        else:
            prompt = make_forced_prompt(question, forced)
            # Forced-think also starts on the small budget so forcing can fire.
            budget = self.initial_think_tokens if forced == "think" else self.max_answer_tokens
            completion, n_tok = self.backend.generate(prompt, budget, self.temperature, greedy)
            decision = forced
            # the routing token is prefilled, so it is part of the logical completion
            completion = ("<think>" if forced == "think" else "<no_think>") + completion
            prefilled = True

        is_think = decision == "think"
        if is_think and self.budget_force:
            completion, extra = self._budget_extend(
                question, prompt, completion, n_tok, greedy, prefilled
            )
            n_tok += extra

        return RouteResult(
            question=question,
            answer=extract_answer(completion),
            decision=decision,
            difficulty=d,
            n_tokens=n_tok,
            latency_s=time.time() - t0,
            completion=completion,
            overridden=overridden,
        )

    def _budget_extend(self, question, prompt, completion, n_tok, greedy, prefilled=False) -> tuple[str, int]:
        """s1-style forcing: if no boxed answer yet and budget remains, push more thinking.

        Because the first think pass is capped at `initial_think_tokens`
        (< max_think_tokens), there is real headroom for these top-up rounds —
        without that cap the check below would already be saturated and forcing
        would never fire (bug #2).

        The continuation is also capped against the model context window so that
        prompt + completion + "Wait" continuation can never silently overflow.

        `prefilled` means `prompt` already ends with the routing token that is
        also prepended to `completion`; we strip that duplicate from the body so
        the continuation prompt never contains '<think><think>'.
        """
        extra_total = 0
        for _ in range(self.budget_force_max_rounds):
            if extract_answer(completion) is not None:
                break
            if n_tok + extra_total >= self.max_think_tokens:
                break

            body = (completion[len("<think>"):]
                    if prefilled and completion.startswith("<think>") else completion)
            cont_prompt = prompt + body + BUDGET_FORCE_TEXT
            # Token budget still left under the hard think cap.
            remaining = self.max_think_tokens - (n_tok + extra_total)
            # Token budget still left under the context window (prompt+completion
            # already consume space). Cap the top-up so we never overflow n_ctx.
            ctx_remaining = self._context_headroom(cont_prompt)
            allowed = min(remaining, self.max_answer_tokens, ctx_remaining)
            if allowed <= 0:
                break

            more, more_tok = self.backend.generate(
                cont_prompt, allowed, self.temperature, greedy
            )
            completion += BUDGET_FORCE_TEXT + more
            extra_total += more_tok
        return completion, extra_total

    def _context_headroom(self, prompt_text: str) -> int:
        """Tokens left in the context window for new generation after `prompt_text`.

        Uses the backend's exact tokenizer when available (HF / GGUF expose
        `count_tokens`); otherwise falls back to a rough word count. A small
        safety margin is reserved for special tokens the tokenizer may add.
        """
        n_ctx = getattr(self.backend, "n_ctx", GGUF_N_CTX)
        counter = getattr(self.backend, "count_tokens", None)
        try:
            used = counter(prompt_text) if counter else _rough_token_count(prompt_text)
        except Exception:
            used = _rough_token_count(prompt_text)
        margin = 8  # headroom for BOS/EOS/template specials
        return max(0, n_ctx - used - margin)


# ── Loaders ─────────────────────────────────────────────────────────────────────


def build_pipeline(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str | None = None,
    verifier_ckpt: str | None = None,
    gguf_path: str | None = None,
    device: str = "cuda",
    **kwargs,
) -> AdaptivePipeline:
    if gguf_path:
        backend = GGUFBackend(gguf_path)
    else:
        backend = HFBackend(model_name, adapter_path, device)

    verifier = vtok = None
    if verifier_ckpt:
        from adaptivethink.verifier.model import load_verifier

        verifier, vtok = load_verifier(verifier_ckpt, device)

    return AdaptivePipeline(backend, verifier, vtok, device=device, **kwargs)
