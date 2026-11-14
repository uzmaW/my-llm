"""Self-contained tests for nanolm (runnable with plain python or pytest).

    python tests/test_nanolm.py
    # or
    pytest tests/test_nanolm.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanolm.config import ModelConfig
from nanolm.model import NanoLM
from nanolm.tokenizer import CharTokenizer
from nanolm.data import generate_arithmetic_dataset, TextDataset
from nanolm.inference import generate, ModelRunner
from nanolm.reasoning import (
    ReasoningEngine,
    parse_answer,
    parse_question_numbers,
    safe_sum,
)


def _tiny_setup():
    text = "\n".join(generate_arithmetic_dataset(n_samples=40, seed=0))
    tok = CharTokenizer.build(text)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, n_layer=2, n_head=2, n_embd=32, block_size=64
    )
    model = NanoLM(cfg)
    model.eval()
    return model, tok, cfg


def test_tokenizer_roundtrip():
    text = "Q: 12 + 7 = ?\nA: Answer: 19\n<eos>"
    tok = CharTokenizer.build(text)
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text
    # Characters outside the vocab map to <unk> and are dropped on decode.
    assert tok.decode(tok.encode("12 + 7", add_eos=True)) == "12 + 7"
    assert tok.decode(tok.encode("xyz")) == ""  # unknown chars -> <unk>
    print("ok  tokenizer roundtrip")


def test_model_forward_and_loss():
    model, tok, cfg = _tiny_setup()
    x = torch.randint(0, tok.vocab_size, (2, 16))
    logits, loss, cache = model(x, targets=x)
    assert logits.shape == (2, 16, tok.vocab_size)
    assert loss is not None and loss.item() > 0
    assert cache is None  # use_cache defaults to False
    print("ok  forward + loss")


def test_kv_cache_matches_full_forward():
    """Greedy generation with the KV cache must equal a full recompute each step."""
    torch.manual_seed(0)
    model, tok, cfg = _tiny_setup()

    prompt = tok.encode("Q: 5 + 3 = ?\nA: ")
    idx = torch.tensor([prompt], dtype=torch.long)
    n_new = 12

    # (1) Cached generation (temperature ~ 0 -> greedy).
    cached = generate(
        model, idx.clone(), max_new_tokens=n_new,
        temperature=1e-6, top_k=1, use_cache=True,
    )[0].tolist()

    # (2) Reference: recompute the full sequence every step, take argmax.
    seq = list(prompt)
    for _ in range(n_new):
        with torch.no_grad():
            logits, _, _ = model(torch.tensor([seq], dtype=torch.long))
        next_id = int(logits[0, -1].argmax().item())
        seq.append(next_id)
    reference = seq

    assert cached == reference, f"\ncached   = {cached}\nreference= {reference}"
    print("ok  KV cache matches full forward")


def test_reasoning_parsers():
    assert parse_answer("Step 1: 1 + 2 = 3\nAnswer: 42") == 42
    assert parse_answer("no answer here") is None
    assert parse_question_numbers("Q: 12 + 7 + 5 = ?") == [12, 7, 5]
    assert parse_question_numbers("Q: 3 + 4 = ?") == [3, 4]
    assert parse_question_numbers("not math") is None
    assert safe_sum([12, 7, 5]) == 24
    print("ok  reasoning parsers")


def test_reasoning_engine_runs():
    model, tok, cfg = _tiny_setup()
    runner = ModelRunner(model, tok, device="cpu")
    engine = ReasoningEngine(runner, default_max_new_tokens=24)
    result = engine.solve("12 + 7 + 5 = ?", n_samples=2, reflect=False)
    assert result.ground_truth == 24
    assert isinstance(result.correct, bool)
    assert result.traces  # we produced at least one completion
    # Prediction may be None on an untrained model, but the pipeline must run.
    print(f"ok  reasoning engine (pred={result.prediction}, truth={result.ground_truth})")


def test_dataset_batches():
    text = "\n".join(generate_arithmetic_dataset(n_samples=40, seed=1))
    tok = CharTokenizer.build(text)
    ds = TextDataset(text, tok, block_size=32)
    x, y = ds.get_batch(4)
    assert x.shape == (4, 32) and y.shape == (4, 32)
    # y is x shifted by one within each window
    assert torch.equal(x[:, 1:], y[:, :-1])
    print("ok  dataset batches")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()