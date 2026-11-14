"""The ``NanoLM`` model: a decoder-only transformer language model."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import TokenEmbedding
from .layers import TransformerBlock, RMSNorm


class NanoLM(nn.Module):
    """A GPT-style decoder-only transformer built entirely from scratch."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = TokenEmbedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layer)]
        )
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Weight tying: the output projection shares the input embedding.
            self.lm_head.weight = self.token_embedding.emb.weight

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        """Run the model.

        Parameters
        ----------
        idx : LongTensor of shape (B, T)  -- token ids.
        targets : optional LongTensor (B, T) -- next-token labels for the loss.
        kv_caches : optional list of (k, v) per layer, for incremental decoding.
        use_cache : whether to return updated KV caches.

        Returns
        -------
        (logits, loss, new_caches)
        """
        x = self.token_embedding(idx)  # (B, T, C)

        new_caches: Optional[List] = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if (use_cache and kv_caches is not None) else None
            x, new_kv = block(x, kv_cache=kv, use_cache=use_cache)
            if use_cache:
                new_caches.append(new_kv)  # type: ignore[union-attr]

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss, new_caches

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.emb.weight.numel()
        return n

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ):
        """Create an AdamW optimizer with weight decay on 2-D params only."""
        decay, no_decay = set(), set()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).add(name)

        param_dict = {n: p for n, p in self.named_parameters()}
        groups = [
            {"params": [param_dict[n] for n in decay], "weight_decay": weight_decay},
            {"params": [param_dict[n] for n in no_decay], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)