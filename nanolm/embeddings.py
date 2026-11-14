"""Embeddings: token embeddings and rotary positional embeddings (RoPE)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """A plain lookup table mapping token ids to dense vectors."""

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: [x1, x2] -> [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to query and key tensors.

    ``q`` / ``k`` have shape ``(batch, n_head, seq, head_dim)`` and
    ``cos`` / ``sin`` have shape ``(seq, head_dim)``.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings with a precomputed cos/sin cache.

    RoPE encodes absolute position by rotating each 2-D slice of a vector by an
    angle proportional to the token's position.  It is *relative* by
    construction, which is why it composes nicely with KV caching: a token's
    rotation only depends on its absolute position, never on the sequence
    length.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.theta = theta
        self.max_seq_len = max(1, max_seq_len)

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(self.max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int, offset: int = 0):
        """Return ``(cos, sin)`` for positions ``[offset, offset + seq_len)``."""
        needed = offset + seq_len
        if needed > self.max_seq_len:
            self._build_cache(needed)
        cos = self.cos_cached[offset : offset + seq_len]
        sin = self.sin_cached[offset : offset + seq_len]
        return cos, sin