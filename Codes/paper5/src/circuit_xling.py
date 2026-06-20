"""
circuit_xling.py - Stage E: cross-lingual circuit sharing and the alignment-vs-competence
decomposition.

Two questions:
  1. Is the bias circuit shared across languages? -> Jaccard overlap of the per-language
     cut sets. High en-hi overlap with lower en-bn overlap is evidence of a shared core
     plus language-specific periphery.
  2. Where the English cut fails to transfer, is it because the circuit is NOT shared
     (alignment) or because the model cannot read the language (competence)? -> a simple
     linear decomposition of the transfer gap on (1 - overlap) and (1 - competence).
"""

# =====================================================================
# CITATION(S) for this module: none (analysis utility).
# =====================================================================

from typing import Dict, List, Tuple


def jaccard(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return round(inter / union, 4) if union else 0.0


def overlap_matrix(cuts_by_lang: Dict[str, List[Tuple[int, int]]]) -> List[Dict]:
    langs = list(cuts_by_lang.keys())
    rows = []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            rows.append({"lang_a": langs[i], "lang_b": langs[j],
                         "jaccard": jaccard(cuts_by_lang[langs[i]], cuts_by_lang[langs[j]])})
    return rows


def decompose_transfer(transfer_gap: Dict[str, float], overlap_to_en: Dict[str, float],
                       competence: Dict[str, float]) -> Dict:
    """
    Ordinary least squares of transfer_gap on [1 - overlap_to_en, 1 - competence].
    Returns coefficients b_align, b_comp (and the intercept). Pure-python OLS so the
    module has no heavy dependency; languages with missing terms are skipped.
    """
    xs, ys = [], []
    for lang, gap in transfer_gap.items():
        if lang in overlap_to_en and lang in competence:
            xs.append([1.0, 1.0 - overlap_to_en[lang], 1.0 - competence[lang]])
            ys.append(gap)
    if len(xs) < 2:
        return {"b_intercept": None, "b_align": None, "b_comp": None, "n": len(xs)}
    # normal equations: beta = (X^T X)^-1 X^T y  (3x3 solve)
    import itertools
    n, p = len(xs), 3
    XtX = [[sum(xs[r][i] * xs[r][k] for r in range(n)) for k in range(p)] for i in range(p)]
    Xty = [sum(xs[r][i] * ys[r] for r in range(n)) for i in range(p)]
    beta = _solve_3x3(XtX, Xty)
    if beta is None:
        return {"b_intercept": None, "b_align": None, "b_comp": None, "n": n}
    return {"b_intercept": round(beta[0], 4), "b_align": round(beta[1], 4),
            "b_comp": round(beta[2], 4), "n": n}


def _solve_3x3(A, b):
    """Gaussian elimination for a 3x3 system; returns None if singular."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        M[col] = [x / pivval for x in M[col]]
        for r in range(3):
            if r != col:
                factor = M[r][col]
                M[r] = [M[r][k] - factor * M[col][k] for k in range(4)]
    return [M[0][3], M[1][3], M[2][3]]
