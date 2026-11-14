"""Train a NanoLM from scratch on a synthetic chain-of-thought dataset.

Example
-------
    python -m nanolm.train --steps 3000 --out checkpoints/demo

The script:
  * builds a synthetic multi-step addition dataset (or reads --text-file),
  * builds a character tokenizer,
  * initialises a small decoder-only transformer,
  * trains with AdamW + linear warmup + cosine decay,
  * periodically reports loss and saves a checkpoint.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from .config import ModelConfig
from .data import TextDataset, load_text_file
from .tokenizer import CharTokenizer
from .model import NanoLM
from .inference import save_checkpoint


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_corpus(args) -> str:
    if args.text_file:
        return load_text_file(args.text_file)
    from .data import generate_arithmetic_dataset

    return "\n".join(
        generate_arithmetic_dataset(
            n_samples=args.data_size,
            min_terms=args.min_terms,
            max_terms=args.max_terms,
            min_val=0,
            max_val=args.max_val,
            seed=args.seed,
        )
    )


def build_dataset(args):
    """Return ``(dataset, tokenizer)``."""
    text = build_corpus(args)
    tokenizer = CharTokenizer.build(text)
    dataset = TextDataset(text, tokenizer, args.block_size, val_fraction=args.val_frac)
    return dataset, tokenizer


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a NanoLM from scratch.")
    # Data
    p.add_argument("--text-file", type=str, default=None,
                   help="Train on an arbitrary text file instead of synthetic data.")
    p.add_argument("--data-size", type=int, default=8000)
    p.add_argument("--min-terms", type=int, default=2)
    p.add_argument("--max-terms", type=int, default=4)
    p.add_argument("--max-val", type=int, default=50)
    # Model
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of the corpus held out for validation.")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--no-tie", action="store_true")
    # Optim
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Loop
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--out", type=str, default="checkpoints/demo")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args(argv)


def train(args) -> None:
    torch.manual_seed(args.seed)
    device = args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    dataset, tokenizer = build_dataset(args)

    cfg = ModelConfig(
        vocab_size=dataset.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        tie_embeddings=not args.no_tie,
    )

    model = NanoLM(cfg).to(device)
    print(f"Model parameters: {model.num_params():,} "
          f"(non-embedding {model.num_params(non_embedding=True):,})")
    print(f"Vocab size: {dataset.vocab_size}, block size: {cfg.block_size}")

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    best_loss = float("inf")
    os.makedirs(args.out, exist_ok=True)
    model.train()
    t0 = time.time()

    for step in range(args.steps):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)

        x, y = dataset.get_batch(args.batch_size, device=device)
        _, loss, _ = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            n_tokens = args.batch_size * args.block_size * max(1, args.log_interval)
            tok_per_sec = n_tokens / max(dt, 1e-9) if step > 0 else float("nan")
            print(
                f"step {step:5d}/{args.steps} | loss {loss.item():.4f} "
                f"| lr {lr:.2e} | {tok_per_sec:,.0f} tok/s"
            )

        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(args.eval_iters):
                    xb, yb = dataset.get_batch(args.batch_size, split="val", device=device)
                    _, lb, _ = model(xb, targets=yb)
                    losses.append(lb.item())
                eval_loss = sum(losses) / len(losses)
            model.train()
            print(f"  >> eval @ {step}: loss {eval_loss:.4f}")
            if eval_loss < best_loss:
                best_loss = eval_loss
                out_path = os.path.join(args.out, "best.pt")
                save_checkpoint(
                    out_path, model, tokenizer,
                    extra={"step": step, "eval_loss": eval_loss, "config": cfg.to_dict()},
                )

    final_path = os.path.join(args.out, "final.pt")
    save_checkpoint(
        final_path, model, tokenizer,
        extra={"step": args.steps, "config": cfg.to_dict()},
    )

    # Report held-out loss / perplexity over a few validation batches.
    model.eval()
    import math as _math
    with torch.no_grad():
        vlosses = []
        for _ in range(args.eval_iters):
            xb, yb = dataset.get_batch(args.batch_size, split="val", device=device)
            _, lb, _ = model(xb, targets=yb)
            vlosses.append(lb.item())
    final_val = sum(vlosses) / len(vlosses)
    ppl = _math.exp(final_val)
    print(f"Saved final checkpoint -> {final_path}")
    print(f"Best eval loss : {best_loss:.4f}")
    print(f"Final val loss : {final_val:.4f}   (perplexity ~ {ppl:.2f})")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()