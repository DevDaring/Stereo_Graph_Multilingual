"""
metrics.py - Paper 5 derived metrics.

  Circuit Size        - fraction of residual units cut (per band).
  Cut Bias Reduction  - fairness_dev(baseline) - fairness_dev(after cut).
  Circuit-CLTI        - mean over {hi, bn} of CBR for the English-found cut.
  Circuit Overlap     - Jaccard of per-language cut sets (circuit_xling.py).
"""

# =====================================================================
# CITATION(S) for this module: none (aggregation utility).
# =====================================================================

from typing import Dict, List, Optional


def circuit_size_fraction(cut_units: int, total_units: int) -> float:
    return round(cut_units / total_units, 6) if total_units else 0.0


def circuit_clti(eval_rows: List[Dict], variant: str = "learned_min_cut") -> Optional[float]:
    """Mean Cut Bias Reduction over Hindi and Bengali for one cut variant."""
    vals = [r["cut_bias_reduction"] for r in eval_rows
            if r["variant"] == variant and r["language"] in ("hi", "bn")]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def competence_gated_circuit_clti(eval_rows: List[Dict], competence: Dict[str, bool],
                                  variant: str = "learned_min_cut") -> Optional[float]:
    vals = [r["cut_bias_reduction"] for r in eval_rows
            if r["variant"] == variant and r["language"] in ("hi", "bn")
            and competence.get(r["language"], False)]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def beats_controls(eval_rows: List[Dict], language: str = "en") -> Optional[bool]:
    """True if the learned cut removes more bias than random AND magnitude (same size)."""
    def _cbr(v):
        xs = [r["cut_bias_reduction"] for r in eval_rows
              if r["variant"] == v and r["language"] == language]
        return xs[0] if xs else None
    learned, rnd, mag = _cbr("learned_min_cut"), _cbr("random_cut"), _cbr("magnitude_cut")
    if None in (learned, rnd, mag):
        return None
    return bool(learned > rnd and learned > mag)
