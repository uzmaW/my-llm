"""Datasets: synthetic chain-of-thought arithmetic + a generic text dataset."""

from __future__ import annotations

import random
from typing import List, Tuple

import torch

from .tokenizer import CharTokenizer


# ---------------------------------------------------------------------- #
# Synthetic reasoning corpus
# ---------------------------------------------------------------------- #
def generate_arithmetic_dataset(
    n_samples: int = 8000,
    min_terms: int = 2,
    max_terms: int = 4,
    min_val: int = 0,
    max_val: int = 50,
    seed: int = 1337,
) -> List[str]:
    """Generate addition problems with explicit step-by-step reasoning.

    Each sample looks like::

        Q: 12 + 7 + 5 = ?
        A: Step 1: 12 + 7 = 19
        Step 2: 19 + 5 = 24
        Answer: 24
        <eos>

    Training on this teaches the model a *chain of thought*: it must emit the
    intermediate running sums before committing to a final answer.
    """
    rng = random.Random(seed)
    samples: List[str] = []
    eos = "<eos>"
    for _ in range(n_samples):
        k = rng.randint(min_terms, max_terms)
        nums = [rng.randint(min_val, max_val) for _ in range(k)]
        expr = " + ".join(str(x) for x in nums)

        running = nums[0]
        steps: List[str] = []
        for i, x in enumerate(nums[1:], start=1):
            nxt = running + x
            steps.append(f"Step {i}: {running} + {x} = {nxt}")
            running = nxt

        cot = "\n".join(steps)
        sample = f"Q: {expr} = ?\nA: {cot}\nAnswer: {running}\n{eos}"
        samples.append(sample)
    return samples


def arithmetic_dataset_text(**kwargs) -> str:
    """Return the synthetic dataset joined with newlines."""
    return "\n".join(generate_arithmetic_dataset(**kwargs))


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------- #
# Generic sliding-window dataset
# ---------------------------------------------------------------------- #
class TextDataset:
    """Tokenise a corpus once and sample random (x, y) sliding windows.

    The corpus is split into a train and a (smaller) validation split so we can
    report a held-out loss.  Windows are sampled with a random start offset so
    successive batches are decorrelated (better than reading sequentially).
    """

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        block_size: int,
        val_fraction: float = 0.05,
    ):
        self.block_size = block_size
        data = tokenizer.encode(text)
        if len(data) < block_size + 2:
            raise ValueError(
                f"Corpus is too short ({len(data)} tokens) for block_size={block_size}"
            )
        data = torch.tensor(data, dtype=torch.long)
        n_val = max(1, int(len(data) * val_fraction))
        self.train_data = data[: len(data) - n_val]
        self.val_data = data[len(data) - n_val :]
        self.vocab_size = tokenizer.vocab_size

    def __len__(self) -> int:
        return max(1, len(self.train_data) - self.block_size - 1)

    def get_batch(
        self, batch_size: int, split: str = "train", device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError(f"unknown split: {split!r}")
        hi = max(1, len(data) - self.block_size - 1)
        ix = torch.randint(0, hi, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


def make_arithmetic_dataset(
    block_size: int,
    tokenizer: CharTokenizer,
    n_samples: int = 8000,
    seed: int = 1337,
    val_fraction: float = 0.05,
    **gen_kwargs,
) -> TextDataset:
    text = arithmetic_dataset_text(n_samples=n_samples, seed=seed, **gen_kwargs)
    return TextDataset(text, tokenizer, block_size, val_fraction=val_fraction)