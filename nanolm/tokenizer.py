"""A tiny character-level tokenizer.

The tokenizer is deliberately simple and dependency free: every *character* in
the training corpus becomes one token.  Three special tokens are always present
at the front of the vocabulary:

    index 0 -> <unk>   (unknown characters)
    index 1 -> <pad>
    index 2 -> <eos>

This is enough to train a small model on synthetic reasoning data, yet still
generalises to arbitrary text (unknown characters fall back to ``<unk>``).
"""

from __future__ import annotations

import json
from typing import Iterable, List, Sequence

SPECIALS: List[str] = ["<unk>", "<pad>", "<eos>"]


class CharTokenizer:
    """Map characters <-> integer ids."""

    def __init__(self, chars: Sequence[str], specials: Sequence[str] = tuple(SPECIALS)):
        specials = list(specials)
        # De-duplicate characters while keeping deterministic ordering, and make
        # sure a character never shadows a special token.
        seen: set[str] = set(specials)
        clean_chars: List[str] = []
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                clean_chars.append(ch)

        self.specials: List[str] = specials
        self.chars: List[str] = clean_chars
        self.itos: List[str] = specials + clean_chars
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vocab_size = len(self.itos)
        self.unk_id = self.stoi["<unk>"]
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, text: str, specials: Sequence[str] = tuple(SPECIALS)) -> "CharTokenizer":
        """Build a tokenizer from the set of characters found in *text*."""
        return cls(sorted(set(text)), specials=specials)

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], stop_at: Optional[str] = None) -> str:
        out: List[str] = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok == "<eos>" or tok == "<pad>":
                break  # stop at <eos>/<pad>
            if tok == "<unk>":
                continue
            if stop_at is not None:
                out.append(tok)
                joined = "".join(out)
                idx = joined.find(stop_at)
                if idx != -1:
                    return joined[:idx]
            else:
                out.append(tok)
        return "".join(out)

    def encode_batch(self, texts: Sequence[str], add_eos: bool = False) -> List[List[int]]:
        return [self.encode(t, add_eos=add_eos) for t in texts]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "itos": self.itos,
            "specials": self.specials,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        itos = data["itos"]
        specials = data.get("specials", SPECIALS)
        # Reconstruct from the token list; specials come first by construction.
        obj = cls.__new__(cls)
        obj.specials = list(specials)
        obj.itos = list(itos)
        obj.stoi = {tok: i for i, tok in enumerate(obj.itos)}
        obj.chars = [t for t in obj.itos if t not in obj.specials]
        obj.vocab_size = len(obj.itos)
        obj.unk_id = obj.stoi["<unk>"]
        obj.pad_id = obj.stoi["<pad>"]
        obj.eos_id = obj.stoi["<eos>"]
        return obj

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))