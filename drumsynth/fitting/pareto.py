"""Which candidates are worth reporting, and which one to hand back.

Anything with `n_scalars`, `relative_mse` and `n_frames` can be ranked here —
a single hit or a whole instrument. The policy is the same either way, and
keeping it in one place is what stops "best so far" and "best" from drifting
apart.

Two numbers describe every evaluated candidate — how big the model is and how
wrong it is — and they trade against each other. The frontier is the set of
candidates that nothing else beats on both counts; everything else is strictly
worse than something already in hand, and is dropped from the report.

`preferred` is the one selection policy, in one place: the smallest model that
meets the quality target. When nothing meets it the most accurate candidate
wins instead, and the caller is told the target was missed — a fit that
quietly returns a model ten times worse than asked for is the failure this
whole search exists to prevent.
"""

from __future__ import annotations

import math
from functools import reduce
from typing import Iterable, Sequence


def preferred(a, b, target_mse: float):
    """The better of two evaluations under the fit policy.

    Meeting the target beats missing it; among candidates that meet it the
    smallest model wins, ties broken by accuracy and then by frame count;
    among candidates that miss it, accuracy is all that matters.
    """
    a_ok = a.relative_mse <= target_mse
    b_ok = b.relative_mse <= target_mse

    if a_ok != b_ok:
        return a if a_ok else b
    if not a_ok:
        return a if a.relative_mse <= b.relative_mse else b

    key = lambda e: (e.n_scalars, e.relative_mse, e.n_frames)  # noqa: E731
    return a if key(a) <= key(b) else b


def choose_best(evaluations: Sequence, target_mse: float):
    """Apply `preferred` across a whole set of evaluations."""
    if not evaluations:
        raise ValueError("no candidates were evaluated")
    return reduce(lambda a, b: preferred(a, b, target_mse), evaluations)


def pareto_frontier(evaluations: Iterable) -> list:
    """Non-dominated candidates, smallest model first.

    Sorted by size and then error, a candidate is dominated exactly when
    something earlier in that order already reached an error at least as low —
    so one sweep with a running minimum is enough.
    """
    ordered = sorted(
        (e for e in evaluations if math.isfinite(e.relative_mse)),
        key=lambda e: (e.n_scalars, e.relative_mse),
    )

    frontier: list = []
    best_error = math.inf
    for evaluation in ordered:
        if evaluation.relative_mse < best_error:
            frontier.append(evaluation)
            best_error = evaluation.relative_mse
    return frontier
