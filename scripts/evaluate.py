"""Evaluate reasoning accuracy of a trained checkpoint on held-out problems.

    python scripts/evaluate.py checkpoints/demo/best.pt --n 40 --samples 3
"""

import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanolm.reasoning import ReasoningEngine


def make_questions(n: int, max_terms: int = 4, max_val: int = 99, seed: int = 2024):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(2, max_terms)
        nums = [rng.randint(0, max_val) for _ in range(k)]
        out.append(" + ".join(str(x) for x in nums) + " = ?")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-val", type=int, default=99)
    parser.add_argument("--max-terms", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-reflect", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    engine = ReasoningEngine.from_checkpoint(args.checkpoint, device=args.device)
    questions = make_questions(args.n, args.max_terms, args.max_val, args.seed)

    stats = engine.evaluate(
        questions,
        n_samples=args.samples,
        temperature=args.temperature,
        reflect=not args.no_reflect,
    )
    print(f"\nAccuracy: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")
    num_reflected = sum(1 for r in stats["results"] if r.reflected)
    print(f"Reflections triggered: {num_reflected}")

    print("\nPer-problem results:")
    for r in stats["results"]:
        flag = "OK " if r.correct else "BAD"
        print(f"  [{flag}] {r.question:22s} pred={r.prediction} truth={r.ground_truth}")


if __name__ == "__main__":
    main()