"""Small statistics helpers (CPU, stdlib + numpy). Used by step_10 analysis and
step_05 stability: bootstrap CIs, Cohen's d, set Jaccard, and a Cochran's Q /
I-squared heterogeneity test for the per-cell gains.
"""
import math
import random
from typing import Dict, List, Sequence, Tuple


def jaccard(a: Sequence, b: Sequence) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cohens_d(xs: Sequence[float]) -> float:
    """One-sample Cohen's d against 0 (effect size of a mean gain)."""
    s = std(xs)
    return (mean(xs) / s) if s > 0 else 0.0


def bootstrap_ci(xs: Sequence[float], n: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float]:
    xs = list(xs)
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    k = len(xs)
    for _ in range(n):
        sample = [xs[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[int((1 - alpha / 2) * n) - 1]
    return (lo, hi)


def cochran_q(effects: Sequence[float], variances: Sequence[float]) -> Dict:
    """Inverse-variance Cochran's Q and I-squared across cells. variances are the
    per-cell sampling variances of the effect (e.g. from a per-cell bootstrap)."""
    pairs = [(e, v) for e, v in zip(effects, variances) if v and v > 0]
    if len(pairs) < 2:
        return {"q": None, "df": 0, "i_squared": None, "n_cells": len(pairs)}
    w = [1.0 / v for _, v in pairs]
    wsum = sum(w)
    wbar = sum(wi * e for wi, (e, _) in zip(w, pairs)) / wsum
    q = sum(wi * (e - wbar) ** 2 for wi, (e, _) in zip(w, pairs))
    df = len(pairs) - 1
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return {"q": round(q, 4), "df": df, "i_squared": round(i2, 2), "n_cells": len(pairs),
            "pooled_effect": round(wbar, 4)}
