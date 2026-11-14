#!/usr/bin/env bash
# End-to-end quickstart: train -> generate -> reason, with a small/fast config.
set -e
cd "$(dirname "$0")/.."

CKPT=checkpoints/demo/best.pt

if [ ! -f "$CKPT" ]; then
  echo "==> Training small model"
  python -m nanolm.train \
      --steps 1500 --batch-size 32 \
      --n-layer 4 --n-embd 128 --n-head 4 --block-size 128 \
      --data-size 10000 --max-terms 4 --max-val 99 \
      --log-interval 100 --eval-interval 250 \
      --out checkpoints/demo
fi

echo "==> Generating a chain of thought"
python -m nanolm.inference "$CKPT" "Q: 12 + 7 + 5 = ?
A: " --max-new-tokens 48 --temperature 0.2 --repetition-penalty 1.0

echo "==> Reasoning"
python -m nanolm.reasoning "$CKPT" \
    "12 + 7 + 5 = ?" "99 + 44 + 33 = ?" --samples 5 --temperature 0.2