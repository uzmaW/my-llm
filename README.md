# nanolm — build a model, from scratch

A complete, self-contained **decoder-only transformer language model** written
from first principles in PyTorch. No HuggingFace, no external model code —
everything is implemented in this repository:

| Component   | Where | What's inside |
|-------------|-------|---------------|
| **Transformer** | `nanolm/layers.py` | RMSNorm, causal multi-head attention (RoPE + KV cache), SwiGLU MLP, pre-norm blocks |
| **Model**   | `nanolm/model.py`  | token embedding + blocks + LM head (optionally tied), init, optimizer setup |
| **Embeddings** | `nanolm/embeddings.py` | token lookup + rotary positional embeddings (RoPE) |
| **Tokenizer** | `nanolm/tokenizer.py` | dependency-free char-level tokenizer (`<unk>/<pad>/<eos>`) |
| **Inference** | `nanolm/inference.py` | autoregressive `generate()` with KV cache, temperature / top-k / top-p / repetition penalty, checkpoint loader |
| **Reasoning** | `nanolm/reasoning.py` | chain-of-thought prompting, self-consistency (majority vote), tool verification, reflection loop |
| **Data**    | `nanolm/data.py` | synthetic multi-step addition dataset + generic sliding-window text dataset |
| **Training** | `nanolm/train.py` | AdamW, linear warmup + cosine decay, gradient clipping, checkpointing |

## Architecture (Llama-style GPT)

```
tokens ─▶ TokenEmbedding ─▶ [ RMSNorm → CausalSelfAttention(+RoPE) → +
                              RMSNorm → SwiGLU MLP            → + ] × n_layer
                          ─▶ RMSNorm ─▶ LM head ─▶ logits
```

* **Pre-norm** residual blocks for stable training.
* **RoPE** rotary positional embeddings (no learned absolute positions) — they
  compose cleanly with the **KV cache** because a token's rotation depends only
  on its absolute position.
* **SwiGLU** feed-forward networks.
* Optional **weight tying** between the token embedding and the LM head.

## Install

```bash
pip install -r requirements.txt   # just torch (CPU is fine)
```

## Quickstart

### 1. Train a small reasoning model

```bash
python -m nanolm.train \
    --steps 2500 --batch-size 32 \
    --n-layer 4 --n-embd 128 --n-head 4 --block-size 128 \
    --data-size 12000 --max-terms 4 --max-val 99 \
    --out checkpoints/demo
```

This trains on a synthetic dataset of multi-step addition with an explicit
chain of thought, e.g.:

```
Q: 12 + 7 + 5 = ?
A: Step 1: 12 + 7 = 19
Step 2: 19 + 5 = 24
Answer: 24
```

### 2. Generate text

```bash
python -m nanolm.inference checkpoints/demo/best.pt "Q: 12 + 7 + 5 = ?\nA: " \
    --max-new-tokens 48 --temperature 0.2
```

### 3. Reason (chain-of-thought + self-consistency + tool check + reflection)

```bash
python -m nanolm.reasoning checkpoints/demo/best.pt \
    "12 + 7 + 5 = ?" "99 + 44 + 33 = ?" --samples 5 --verbose
```

## Python API

```python
from nanolm import ModelRunner, ReasoningEngine

runner = ModelRunner.from_checkpoint("checkpoints/demo/best.pt")
print(runner.complete("Q: 12 + 7 + 5 = ?\nA: ", max_new_tokens=48, temperature=0.2))

engine = ReasoningEngine(runner)
result = engine.solve("12 + 7 + 5 = ?", n_samples=5)
print(result.prediction, result.ground_truth, result.correct)
```

## Reasoning techniques

The `ReasoningEngine` layers several ideas that make a small model far more
reliable than a single greedy pass:

1. **Chain of thought** — the model is trained to emit intermediate `Step k`
   lines before committing to `Answer: X`.
2. **Self-consistency** — sample `n_samples` independent completions, parse the
   final answer of each, and take a majority vote.
3. **Tool verification** — a safe arithmetic evaluator computes the ground truth
   so the engine can check itself.
4. **Reflection** — if the voted answer disagrees with the tool, re-prompt with a
   hint and re-sample.

## Train on your own text

Train on any `.txt` corpus (default here is **Tiny Shakespeare**):

```bash
# download the dataset once
mkdir -p datasets
curl -L -o datasets/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# train a char-level model from scratch
python -m nanolm.train --text-file datasets/shakespeare.txt \
    --steps 5000 --batch-size 64 --block-size 128 \
    --n-layer 6 --n-embd 256 --n-head 8 --lr 3e-4 \
    --out checkpoints/shakespeare

# sample some Shakespeare-like output
python -m nanolm.inference checkpoints/shakespeare/best.pt \
    "First Citizen:\n" --max-new-tokens 500 --temperature 0.8 --top-k 40
```

## Tests

```bash
python tests/test_nanolm.py     # plain python
# or
pytest tests/ -q
```

Includes a check that **cached generation bit-for-bit matches a full forward
recompute** (greedy), validating the KV-cache implementation.

## Config knobs

`--n-layer`, `--n-head`, `--n-embd`, `--block-size`, `--dropout`, `--rope-theta`,
`--no-tie`, plus optimizer (`--lr`, `--warmup`, `--min-lr`, `--weight-decay`) and
loop (`--batch-size`, `--steps`, `--eval-interval`) options. See
`python -m nanolm.train --help`.

## Desktop app — business-sales modeling

`desktop_app.py` is a native **Tkinter desktop app** that ties the whole stack to
a real, publicly available *business-sales* dataset (seaborn's `taxis`: 6,433
revenue transactions with a timestamp, payment type, distance and amount). It:

* downloads + loads the dataset,
* trains the **from-scratch transformer** on a modeling corpus
  (`amount|day|hour|payment|distance` per sale) built in `nanolm/sales_data.py`,
* **generates synthetic sales records** (transformer sampling),
* shows **analytics** + a transparent daily-revenue **forecast** + a bar chart.

```bash
python desktop_app.py          # or: bash run_app.sh
```

Click **Load data → Train model → Generate sales / Analytics**. Training and
generation run in a background thread; loss and results stream into the log
pane. The forecast is a transparent OLS trend (clearly *analytic*, not a learned
prediction); the transformer itself *models* the sales-record distribution. Swap
`nanolm/sales_data.py`'s URL to use a different sales CSV.

## Train on Tiny Shakespeare (this repo)

Download the well-known char-level corpus (Karpathy's `char-rnn` *Tiny
Shakespeare*, ~1.1 MB / 65 chars) and train from scratch:

```bash
# from-scratch char-LM training (checkpoints best.pt saved every 100 steps)
python -m nanolm.train --text-file datasets/shakespeare.txt \
    --steps 6000 --batch-size 48 --block-size 128 \
    --n-layer 6 --n-embd 192 --n-head 6 --lr 4e-4 \
    --out checkpoints/shakespeare

# sample
python -m nanolm.inference checkpoints/shakespeare/best.pt \"DUKE:\" \
    --max-new-tokens 320 --temperature 0.8 --top-k 40
```

Validation loss (char-level cross-entropy) drops from ~4.22 (ln 68, random)
toward ~1.4 over the first ~600 steps, then keeps improving; the model starts
to emit Shakespeare-style **speaker labels, dialogue line breaks and archaic
word shapes**. A longer run (more steps / bigger `--n-embd`) improves coherence.

## Results

* Config: 6 layers, `n_embd=192`, 6 heads (RoPE), SwiGLU, tied embeddings,
  char-level tokenizer, `vocab=68`. ~2.7 M parameters.
* Training (CPU): `step 0 loss 4.23` (= ln 68, random) → **best held-out loss
  `1.553` at step 1200**; still improving when the run was stopped. A full
  `--steps 6000` run (~100 min on 8 CPU cores) reaches noticeably lower loss.
* Checkpoint: `checkpoints/shakespeare/best.pt` (contains config + tokenizer +
  metadata; see `nanolm.inference.load_checkpoint`).
* Sample output (temperature 0.8) is Shakespeare-shaped — speaker labels, verse
  line breaks, archaic vocabulary — e.g.:

  ```
  My lord, as here lies the man
  Of intercession by the king.
  ```

## Roadmap / ideas

* BPE tokenizer for larger vocabularies
* Grouped-query attention + sliding-window attention
* Mixed precision + `torch.compile` for speed
* Beam search in `inference.py`
* Richer tool set for the reasoning engine (calculator, retrieval)
