"""Core transformer building blocks (all implemented from scratch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import RotaryEmbedding, apply_rope


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV cache."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.rope = RotaryEmbedding(self.head_dim, cfg.block_size, cfg.rope_theta)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        offset = kv_cache[0].shape[2] if kv_cache is not None else 0
        cos, sin = self.rope(T, offset)
        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # Build the attention mask / causal flag.
        if kv_cache is None and T > 1:
            attn_mask, is_causal = None, True
        elif kv_cache is None:
            attn_mask, is_causal = None, False
        else:
            S = k.shape[2]
            q_pos = torch.arange(offset, offset + T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(S, device=x.device).unsqueeze(0)
            attn_mask = (k_pos <= q_pos)[None, None]  # broadcast over B, heads
            is_causal = False

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.wo(y))
        return y, new_cache


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.d_ff, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.n_embd, cfg.d_ff, bias=cfg.bias)
        self.w_down = nn.Linear(cfg.d_ff, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class TransformerBlock(nn.Module):
    """A pre-norm transformer block: attention sub-layer + MLP sub-layer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, kv_cache=None, use_cache: bool = False):
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv