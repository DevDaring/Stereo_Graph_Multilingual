"""Unit tests for the leakage-free core: split determinism, retrieval self-
exclusion, context builders, and the stats helpers. No data files or network.

Run:  python tests/test_core.py   (or: pytest tests/test_core.py)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Robust_Rerun/

from lib.splits import assign_splits, _bucket  # noqa: E402
from lib.rag_leakfree import _retrieve, _context, SAFETY  # noqa: E402
from lib import stats as S  # noqa: E402


def test_split_is_deterministic():
    concepts = {f"g{i}": f"c{i % 50}" for i in range(200)}
    a = assign_splits(concepts, 0.30, 0.15, seed=42)
    b = assign_splits(concepts, 0.30, 0.15, seed=42)
    assert a == b
    assert set(a.values()) <= {"train", "val", "test"}


def test_split_seed_changes_assignment():
    concepts = {f"c{i}": f"c{i}" for i in range(300)}
    a = assign_splits(concepts, 0.30, 0.15, seed=1)
    b = assign_splits(concepts, 0.30, 0.15, seed=2)
    assert a != b


def test_bucket_stable():
    assert _bucket("c1", 42) == _bucket("c1", 42)
    assert 0.0 <= _bucket("c1", 42) < 1.0


def test_retrieve_excludes_self_surface_and_concept():
    pools = {
        "by_type_lang": {("gender", "hi"): [("alpha", "cA"), ("beta", "cB"), ("gamma", "cC")]},
        "by_type_lang_xling": {("gender", "hi"): [("alpha", "cA"), ("beta", "cB")]},
        "by_lang": {"hi": [("alpha", "cA"), ("beta", "cB"), ("delta", "cD")]},
    }
    rng = random.Random(0)
    got = _retrieve(pools, "kg_rag_monolingual", "hi", "gender",
                    exclude_surfaces={"alpha"}, exclude_concepts={"cB"}, n_facts=3, rng=rng)
    assert "alpha" not in got          # excluded by surface
    assert "beta" not in got           # excluded by concept cB
    assert "gamma" in got


def test_context_builders():
    assert _context("safety_prompt", []) == SAFETY
    assert _context("base", []) == ""
    ctx = _context("kg_rag", ["x", "y"])
    assert "x" in ctx and "y" in ctx
    assert _context("kg_rag", []) == ""   # no facts -> no context


def test_stats_helpers():
    assert S.jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    lo, hi = S.bootstrap_ci([1.0, 2.0, 3.0, 4.0], n=500, seed=1)
    assert lo <= 2.5 <= hi
    assert abs(S.cohens_d([2.0, 2.0, 2.0])) >= 0.0
    q = S.cochran_q([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    assert q["n_cells"] == 3


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"ALL {len(fns)} CORE TESTS PASSED")


if __name__ == "__main__":
    _run()
