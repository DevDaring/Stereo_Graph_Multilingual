"""Common_00 (GPU-side utility) - model loading, pair building, and bias scoring.

Reuses Paper 5's verified compute (LocalCausalLLM, learn_minimal_cut, band_from_lens,
jaccard) so results stay comparable. Flash-attention is used automatically when the
pre-compiled wheel is installed and config.backbone.use_flash_attention is true
(LocalCausalLLM falls back to sdpa otherwise).
"""
import os
import re
import sys
from typing import Dict, List, Tuple

from Common_00.common import resolve
from Common_00.dataio import read_bias_rows

# --- reuse Paper 5 compute via a path shim --------------------------------- #
_P5SRC = resolve("../paper5/src")
if _P5SRC not in sys.path:
    sys.path.insert(0, _P5SRC)

from backbone_llm import LocalCausalLLM            # noqa: E402
from minimal_cut import learn_minimal_cut, ResidualKeepMask  # noqa: E402
from localize import band_from_lens, llm_logit_lens_trajectory  # noqa: E402
from circuit_xling import jaccard                  # noqa: E402

MASK = "MASK"


def fill(sentence: str, target: str) -> str:
    return re.sub(rf"\b{MASK}\b", lambda _m: target, sentence)


def build_pairs(config: Dict, dataset_key: str, language: str,
                max_pairs: int = 0) -> List[Dict]:
    """Pairs in the exact format Paper 5's learn_minimal_cut expects."""
    out = []
    for r in read_bias_rows(config, dataset_key):
        if r["lang"] != language or not r["group_stereo"] or not r["group_anti"]:
            continue
        st = fill(r["sentence"], r["group_stereo"])
        an = fill(r["sentence"], r["group_anti"])
        if st == an:
            continue
        out.append({
            "index": r["index"], "language": language, "bias_type": r["bias_type"],
            "target_stereotypical": r["group_stereo"],
            "target_anti_stereotypical": r["group_anti"],
            "sentence_stereotypical": st, "sentence_anti_stereotypical": an,
        })
    if max_pairs and len(out) > max_pairs:
        out = out[:max_pairs]
    return out


def load_llm(config: Dict, hf_id: str) -> LocalCausalLLM:
    llm = LocalCausalLLM(hf_id, config["quantization"], config["backbone"])
    llm.load()
    return llm


def _ll(llm, text: str) -> float:
    ll = llm.sequence_log_likelihood(text)
    return float(ll[0] if isinstance(ll, (tuple, list)) else ll)


def bias_score(llm, pairs: List[Dict]) -> float:
    """Percentage of pairs the model prefers stereotypically (50 = fair)."""
    if not pairs:
        return 50.0
    wins = 0
    for p in pairs:
        if _ll(llm, p["sentence_stereotypical"]) > _ll(llm, p["sentence_anti_stereotypical"]):
            wins += 1
    return 100.0 * wins / len(pairs)


def deviation(b: float) -> float:
    return abs(b - 50.0)


def eval_with_mask(llm, pairs: List[Dict], mask: ResidualKeepMask) -> float:
    """Bias score with the keep-mask applied (hard cut)."""
    handles = mask.hooks(llm, hard=True)
    try:
        return bias_score(llm, pairs)
    finally:
        for h in handles:
            h.remove()


def cut_units(mask: ResidualKeepMask) -> List[Tuple[int, int]]:
    return mask.binary_cut()
