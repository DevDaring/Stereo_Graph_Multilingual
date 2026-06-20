"""
ablate_eval.py - Stage D: apply a cut, measure bias + utility, and run the controls
that make the result convincing.

A cut is a set of residual units (layer, dim) that are zeroed via forward hooks. The
learned minimal cut (C1) is compared against, at the SAME cut size:
  C2 random cut          - random units (targeting must matter)
  C3 magnitude cut       - units with the smallest baseline activation magnitude
The bias-utility-cut-size Pareto frontier is the honest headline object: it shows how
much bias is removed for each tolerated drop in utility.
"""

# =====================================================================
# CITATION(S) for this module:
#   [nangia2020crows] Nangia et al., "CrowS-Pairs," EMNLP 2020 (paired bias score).
# =====================================================================

import logging
import random
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger("ablate_eval")


def _cut_hooks(llm, cut_units: List[Tuple[int, int]]):
    """Register hooks that zero the given residual units (layer, dim)."""
    by_layer: Dict[int, List[int]] = {}
    for L, j in cut_units:
        by_layer.setdefault(L, []).append(j)
    layer_mods = llm.layers
    handles = []

    def _mk(L, dims):
        idx = torch.tensor(dims, device=llm.model.device, dtype=torch.long)

        def _hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, :, idx] = 0.0
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return _hook

    for L, dims in by_layer.items():
        handles.append(layer_mods[L].register_forward_hook(_mk(L, dims)))
    return handles


@torch.no_grad()
def intrinsic_bias(llm, pairs: List[Dict], cut_units=None, max_pairs: int = 2000) -> Dict:
    """Percentage of stereotypical wins, optionally with a cut applied."""
    handles = _cut_hooks(llm, cut_units) if cut_units else []
    try:
        wins = []
        for p in pairs[:max_pairs]:
            lp_st, _ = llm.sequence_log_likelihood(p["sentence_stereotypical"])
            lp_an, _ = llm.sequence_log_likelihood(p["sentence_anti_stereotypical"])
            wins.append(lp_st > lp_an)
        n = len(wins)
        score = (sum(wins) / n * 100.0) if n else 0.0
        return {"bias_score": round(score, 4),
                "fairness_deviation": round(abs(score - 50.0), 4), "n_pairs": n}
    finally:
        for h in handles:
            h.remove()


def random_cut(band: List[int], dim: int, size: int, seed: int = 42) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    allu = [(L, j) for L in band for j in range(dim)]
    rng.shuffle(allu)
    return allu[:size]


@torch.no_grad()
def magnitude_cut(llm, pairs: List[Dict], band: List[int], size: int,
                  max_pairs: int = 200) -> List[Tuple[int, int]]:
    """Units with the smallest mean absolute activation (a non-targeted control)."""
    d = llm.model.config.hidden_size
    acc = {L: torch.zeros(d) for L in band}
    n = 0
    for p in pairs[:max_pairs]:
        hs = llm.hidden_states(p["sentence_stereotypical"])     # [L+1, T, d]
        for L in band:
            acc[L] += hs[L + 1].abs().mean(dim=0)
        n += 1
    flat = []
    for L in band:
        v = acc[L] / max(n, 1)
        for j in range(d):
            flat.append((float(v[j]), L, j))
    flat.sort()                                                 # smallest magnitude first
    return [(L, j) for _s, L, j in flat[:size]]


def evaluate_cuts(llm, pairs_by_lang: Dict[str, List[Dict]], learned_cut: List[Tuple[int, int]],
                  band: List[int], seed: int = 42) -> List[Dict]:
    """Evaluate learned/random/magnitude cuts (same size) per language."""
    d = llm.model.config.hidden_size
    size = len(learned_cut)
    rnd = random_cut(band, d, size, seed)
    mag = magnitude_cut(llm, pairs_by_lang.get("en", []), band, size)
    rows = []
    for lang, pairs in pairs_by_lang.items():
        base = intrinsic_bias(llm, pairs)
        variants = {"learned_min_cut": learned_cut, "random_cut": rnd, "magnitude_cut": mag}
        for vname, cut in variants.items():
            res = intrinsic_bias(llm, pairs, cut_units=cut)
            cbr = round(base["fairness_deviation"] - res["fairness_deviation"], 4)
            rows.append({"language": lang, "variant": vname, "cut_size": size,
                         "bias_baseline": base["bias_score"], "bias_after_cut": res["bias_score"],
                         "cut_bias_reduction": cbr, "n_pairs": res["n_pairs"]})
            logger.info("[OK] %s/%s size=%d base=%.2f cut=%.2f CBR=%.2f",
                        vname, lang, size, base["bias_score"], res["bias_score"], cbr)
    return rows
