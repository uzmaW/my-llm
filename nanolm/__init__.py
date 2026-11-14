"""nanolm -- a from-scratch, dependency-light decoder-only transformer LM.

Public API
----------
    from nanolm.config import ModelConfig
    from nanolm.model import NanoLM
    from nanolm.inference import ModelRunner, generate
    from nanolm.reasoning import ReasoningEngine

Everything (embeddings, RoPE, attention, MLP, sampling, reasoning) is implemented
from scratch in PyTorch -- no HuggingFace, no external model code.
"""

from .config import ModelConfig
from .model import NanoLM
from .tokenizer import CharTokenizer
from .inference import ModelRunner, generate, load_checkpoint, save_checkpoint
from .reasoning import ReasoningEngine, ReasoningResult

__all__ = [
    "ModelConfig",
    "NanoLM",
    "CharTokenizer",
    "ModelRunner",
    "generate",
    "load_checkpoint",
    "save_checkpoint",
    "ReasoningEngine",
    "ReasoningResult",
]

__version__ = "0.1.0"