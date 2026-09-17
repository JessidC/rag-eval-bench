"""Paired statistics for comparing two runs over the same golden set.

Two configs answer the same questions, so the honest test looks only at the questions
where they disagree (McNemar). With a few dozen questions a 3-point gain is usually noise,
and the gate should say so instead of shipping it.
"""
from __future__ import annotations

import random
from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b: only baseline right, c: only candidate right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_counts(base: list[bool], cand: list[bool]) -> tuple[int, int]:
    b = sum(1 for x, y in zip(base, cand) if x and not y)
    c = sum(1 for x, y in zip(base, cand) if y and not x)
    return b, c


def bootstrap_diff(base: list[bool], cand: list[bool], n: int = 10000, seed: int = 7) -> tuple[float, float, float]:
    """Mean difference (cand - base) in accuracy with a 95% paired bootstrap interval."""
    rng = random.Random(seed)
    m = len(base)
    diffs = [int(y) - int(x) for x, y in zip(base, cand)]
    point = sum(diffs) / m
    samples = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return point, samples[int(0.025 * n)], samples[int(0.975 * n) - 1]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]
