"""Inference utilities: autoregressive sampling with a KV cache, and a loader."""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model import NanoLM
from .tokenizer import CharTokenizer


def _top_k_filter(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if not top_k:
        return logits
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    # Keep tokens up to and including the one crossing top_p.
    remove_sorted = (cumprobs - probs) > top_p
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


@torch.no_grad()
def generate(
    model: NanoLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    use_cache: bool = True,
) -> torch.Tensor:
    """Autoregressively generate tokens, returning prompt + completion ids.

    The generation loop feeds the whole prompt once, then feeds one token at a
    time reusing the cached keys/values.  Sampling supports temperature,
    top-k, top-p (nucleus) and a repetition penalty.
    """
    model.eval()
    was_training = model.training
    idx = idx.to(next(model.parameters()).device)

    finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    kv_caches = None
    cur = idx

    for _ in range(max_new_tokens):
        logits, _, kv_caches = model(cur, kv_caches=kv_caches, use_cache=use_cache)
        logits = logits[:, -1, :].float()

        # Repetition penalty (applied to logits).
        if repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                for t in torch.unique(cur[b]).tolist():
                    logit = logits[b, t]
                    logits[b, t] = (
                        logit / repetition_penalty
                        if logit > 0
                        else logit * repetition_penalty
                    )

        logits = logits / max(temperature, 1e-6)
        logits = _top_k_filter(logits, top_k)
        logits = _top_p_filter(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Do not overwrite already-finished sequences with new tokens.
        if eos_token_id is not None:
            next_id = next_id.masked_fill(finished.unsqueeze(1), eos_token_id)
            finished = finished | (next_id.squeeze(1) == eos_token_id)

        idx = torch.cat([idx, next_id], dim=1)
        cur = next_id  # only the newest token on the next iteration

        if eos_token_id is not None and finished.all():
            break

    if was_training:
        model.train()
    return idx
# ---------------------------------------------------------------------- #
# Checkpoint loading / saving
# ---------------------------------------------------------------------- #
def save_checkpoint(
    path: str,
    model: NanoLM,
    tokenizer: CharTokenizer,
    extra: Optional[dict] = None,
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "config": model.cfg.to_dict(),
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns ``(model, tokenizer, extra)``.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(payload["config"])
    model = NanoLM(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload.get("extra", {})


class ModelRunner:
    """Convenience wrapper bundling a model + tokenizer for generation."""

    def __init__(self, model: NanoLM, tokenizer: CharTokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ModelRunner":
        model, tokenizer, _ = load_checkpoint(path, device=device)
        return cls(model, tokenizer, device=device)

    def _make_ids(self, prompt: str, add_eos: bool = False) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_eos=add_eos)
        if not ids:
            return torch.empty((0,), dtype=torch.long)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95,
        add_eos_to_prompt: bool = False,
        stop_at_eos: bool = True,
        repetition_penalty: float = 1.0,
    ) -> str:
        idx = self._make_ids(prompt, add_eos=add_eos_to_prompt)
        if idx.numel() == 0:
            return ""
        out = generate(
            self.model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_id if stop_at_eos else None,
            repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.decode(out[0].tolist(), stop_at="\n\n")

    @torch.no_grad()
    def complete_ids(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95,
        stop_at_eos: bool = True,
    ) -> List[int]:
        idx = self._make_ids(prompt)
        if idx.numel() == 0:
            return []
        out = generate(
            self.model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_id if stop_at_eos else None,
        )
        return out[0].tolist()


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with a trained NanoLM.")
    parser.add_argument("checkpoint", help="Path to a checkpoint saved by train.py")
    parser.add_argument("prompt", nargs="?", default="", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    runner = ModelRunner.from_checkpoint(args.checkpoint, device=args.device)
    text = runner.complete(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(text)


if __name__ == "__main__":  # pragma: no cover
    _main()