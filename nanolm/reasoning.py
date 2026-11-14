"""A lightweight reasoning engine on top of a trained NanoLM.

The engine combines several classic *reasoning* techniques that work well with a
small model:

1. **Chain-of-thought prompting** -- the model is trained to emit intermediate
   ``Step k: ...`` lines before the ``Answer:``.
2. **Self-consistency** -- sample several independent completions and take a
   majority vote over the parsed final answers.
3. **Tool use / verification** -- a safe arithmetic evaluator computes the ground
   truth so the engine can self-check its own answer.
4. **Reflection** -- if the voted answer disagrees with the tool, the engine
   re-prompts with an explicit hint and re-samples once.

Everything is intentionally dependency free and easy to audit.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .inference import ModelRunner, generate

_NUM = r"-?\d+"
_ANSWER_RE = re.compile(rf"Answer:\s*({_NUM})")
_EXPR_RE = re.compile(rf"^{_NUM}(?:\s*\+\s*{_NUM})*$")


def parse_answer(text: str) -> Optional[int]:
    """Return the last ``Answer: <int>`` found in *text*, or ``None``."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def parse_question_numbers(question: str) -> Optional[List[int]]:
    """Extract the integers from an ``a + b + c = ?`` style question."""
    expr = question.strip().replace("Q:", "").strip()
    expr = re.sub(r"=\s*\??\s*$", "", expr).strip()
    if not _EXPR_RE.match(expr):
        return None
    return [int(x) for x in re.findall(_NUM, expr)]


def safe_sum(numbers: List[int]) -> int:
    """A 'tool': exact arithmetic, used to verify the model."""
    return sum(numbers)


@dataclass
class ReasoningResult:
    question: str
    prediction: Optional[int]
    ground_truth: Optional[int]
    correct: bool
    num_samples: int
    method: str
    votes: dict = field(default_factory=dict)
    traces: List[str] = field(default_factory=list)
    reflected: bool = False


class ReasoningEngine:
    """Chain-of-thought + self-consistency + tool verification + reflection."""

    PROMPT_TEMPLATE = "Q: {q} = ?\nA: "

    def __init__(
        self,
        runner: ModelRunner,
        default_max_new_tokens: int = 64,
        default_temperature: float = 0.8,
        default_top_k: int = 40,
        default_top_p: float = 0.95,
    ):
        self.runner = runner
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.default_top_k = default_top_k
        self.default_top_p = default_top_p

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ReasoningEngine":
        return cls(ModelRunner.from_checkpoint(path, device=device))

    def _prompt(self, question: str) -> str:
        return self.PROMPT_TEMPLATE.format(q=question.strip())

    def _sample_once(self, prompt, temperature, top_k, top_p, max_new_tokens) -> str:
        return self.runner.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.1,
        )
    # ------------------------------------------------------------------ #
    def solve(
        self,
        question: str,
        n_samples: int = 5,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        use_tool: bool = True,
        reflect: bool = True,
        verbose: bool = False,
    ) -> ReasoningResult:
        """Solve one problem and return a structured result."""
        temperature = self.default_temperature if temperature is None else temperature
        top_k = self.default_top_k if top_k is None else top_k
        top_p = self.default_top_p if top_p is None else top_p
        max_new_tokens = (
            self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        )

        numbers = parse_question_numbers(question)
        truth = safe_sum(numbers) if (use_tool and numbers is not None) else None

        prompt = self._prompt(question)
        traces: List[str] = []
        votes: Counter = Counter()

        for _ in range(max(1, n_samples)):
            completion = self._sample_once(
                prompt, temperature, top_k, top_p, max_new_tokens
            )
            traces.append(completion.strip())
            ans = parse_answer(completion)
            if ans is not None:
                votes[ans] += 1

        prediction = votes.most_common(1)[0][0] if votes else None

        # Reflection: if we have a tool-verified answer and disagree, re-prompt.
        reflected = False
        if reflect and truth is not None and prediction is not None and prediction != truth:
            reflected = True
            hint_prompt = (
                prompt
                + "Hint: recompute each running sum carefully.\nA: "
            )
            for _ in range(max(2, n_samples)):
                completion = self._sample_once(
                    hint_prompt, temperature, top_k, top_p, max_new_tokens
                )
                traces.append(completion.strip())
                ans = parse_answer(completion)
                if ans is not None:
                    votes[ans] += 2  # weight reflection samples a bit higher
            prediction = votes.most_common(1)[0][0]

        correct = prediction is not None and truth is not None and prediction == truth

        if verbose:
            for i, tr in enumerate(traces):
                print(f"--- sample {i} ---\n{tr}\n")

        method = "self-consistency"
        if truth is None:
            method += " (no tool)"
        if reflected:
            method += " + reflection"

        return ReasoningResult(
            question=question.strip(),
            prediction=prediction,
            ground_truth=truth,
            correct=correct,
            num_samples=n_samples,
            method=method,
            votes=dict(votes),
            traces=traces,
            reflected=reflected,
        )

    def evaluate(self, questions: List[str], n_samples: int = 5, **kwargs) -> dict:
        """Solve a batch of questions and report accuracy."""
        results = [self.solve(q, n_samples=n_samples, **kwargs) for q in questions]
        total = len(results)
        correct = sum(1 for r in results if r.correct)
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "results": results,
        }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Reason with a trained NanoLM.")
    parser.add_argument("checkpoint", help="Path to a checkpoint saved by train.py")
    parser.add_argument("questions", nargs="+", help='e.g. "12 + 7 + 5 = ?"')
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-tool", action="store_true")
    parser.add_argument("--no-reflect", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    engine = ReasoningEngine.from_checkpoint(args.checkpoint, device=args.device)
    for q in args.questions:
        result = engine.solve(
            q,
            n_samples=args.samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_tool=not args.no_tool,
            reflect=not args.no_reflect,
            verbose=args.verbose,
        )
        status = "OK " if result.correct else "BAD"
        print(
            f"[{status}] {result.question}  -> pred={result.prediction} "
            f"truth={result.ground_truth}  ({result.method})"
        )


if __name__ == "__main__":  # pragma: no cover
    _main()