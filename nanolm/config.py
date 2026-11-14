"""Model / training configuration for the from-scratch ``nanolm`` transformer.

The architecture is a modern decoder-only (GPT / Llama style) transformer:

* token + rotary positional embeddings (no learned absolute positions)
* pre-norm residual blocks using RMSNorm
* causal multi-head self-attention with rotary embeddings (RoPE)
* SwiGLU feed-forward networks
* optional weight tying between the token embedding and the LM head
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Hyper-parameters describing a :class:`nanolm.model.NanoLM`."""

    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    bias: bool = False
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("n_layer, n_head and n_embd must all be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def d_ff(self) -> int:
        """Hidden width of the SwiGLU MLP (rounded up to a multiple of 16)."""
        hidden = int(self.n_embd * 8 / 3)
        return ((hidden + 15) // 16) * 16

    # ------------------------------------------------------------------ #
    # (De)serialisation helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        # Only keep keys the dataclass knows about, so old checkpoints still load.
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})