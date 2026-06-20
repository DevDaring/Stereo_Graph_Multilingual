"""
encoder_ops.py - Encoder (XLM-R, MuRIL, mBERT) path for Paper 5.

Keeps the three frozen encoders (Paper 1's model family) in full symmetry with the
LLM path: the SAME GraphBiasFilter is inserted at the encoder's bias-emergence layer,
and bias is measured with the SAME CrowS-Pairs pseudo-log-likelihood metric used in
Papers 1 and 2.

  layer_energy_trajectory - per-layer ||mean(stereo) - mean(anti)|| -> emergence layer
  encoder_bias            - PLL bias per language (baseline or with the filter hook)
  capture_layer_states    - states at L* for layer-local filter training
"""

# =====================================================================
# CITATION(S) for this module:
#   [nangia2020crows] Nangia et al., "CrowS-Pairs," EMNLP 2020 (PLL bias).
#   [conneau2020xlmr] Conneau et al., XLM-R, ACL 2020. arXiv:1911.02116.
#   [khanuja2021muril] Khanuja et al., MuRIL, 2021. arXiv:2103.10730.
# =====================================================================

import contextlib
import logging
from typing import Dict, List

import pandas as pd
import torch

from backbone import get_mlm_components
from eval.metrics_bias_pll import compute_pll_bias

logger = logging.getLogger("encoder_ops")


def encoder_layers(masked_lm) -> List[torch.nn.Module]:
    """The transformer block list of the base encoder (BERT/RoBERTa family)."""
    base = masked_lm.base_model
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        return list(base.encoder.layer)
    raise AttributeError("[FAIL] Could not locate encoder layers (expected base.encoder.layer).")


@contextlib.contextmanager
def insert_filter(masked_lm, layer_idx: int, flt):
    """Insert the GraphBiasFilter at encoder layer `layer_idx` via a forward hook."""
    layer = encoder_layers(masked_lm)[layer_idx]
    flt = flt.to(next(masked_lm.parameters()).device)

    def _hook(_m, _i, out):
        if isinstance(out, tuple):
            new0 = flt(out[0].float()).to(out[0].dtype)
            return (new0,) + tuple(out[1:])
        return flt(out.float()).to(out.dtype)

    handle = layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def _pooled_layers(masked_lm, tokenizer, text: str, device, max_length: int) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = masked_lm.base_model(**enc, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0).squeeze(1)      # [L+1, T, d]
    return hs.float().mean(dim=1).cpu()                        # [L+1, d] mean over tokens


@torch.no_grad()
def layer_energy_trajectory(masked_lm, tokenizer, pairs: List[Dict],
                            max_pairs: int = 300, max_length: int = 128) -> List[float]:
    """Per-layer bias-direction energy ||mean(stereo)-mean(anti)|| (the emergence signal)."""
    device = next(masked_lm.parameters()).device
    sum_st = sum_an = None
    n = 0
    for p in pairs[:max_pairs]:
        st = _pooled_layers(masked_lm, tokenizer, p["sentence_stereotypical"], device, max_length)
        an = _pooled_layers(masked_lm, tokenizer, p["sentence_anti_stereotypical"], device, max_length)
        sum_st = st if sum_st is None else sum_st + st
        sum_an = an if sum_an is None else sum_an + an
        n += 1
    if n == 0:
        return []
    energy = ((sum_st / n) - (sum_an / n)).norm(dim=-1)        # [L+1]
    return [float(x) for x in energy.tolist()]


@torch.no_grad()
def capture_layer_states(masked_lm, tokenizer, sentences: List[str], layer_idx: int,
                         max_length: int = 128) -> List[torch.Tensor]:
    device = next(masked_lm.parameters()).device
    states = []
    for s in sentences:
        enc = tokenizer(s, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = masked_lm.base_model(**enc, output_hidden_states=True)
        states.append(out.hidden_states[layer_idx + 1].squeeze(0).float().cpu())  # [T, d]
    return states


def _pairs_to_df(pairs: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "sentence_stereotypical": p["sentence_stereotypical"],
        "sentence_anti_stereotypical": p["sentence_anti_stereotypical"],
        "language": p["language"], "bias_type": p["bias_type"],
        "row_id": f'{p["dataset"]}:{p["row_index"]}:{p["language"]}',
    } for p in pairs])


def encoder_bias(masked_lm, tokenizer, pairs: List[Dict], flt=None, layer_idx: int = None) -> Dict:
    """PLL bias over `pairs`, optionally with the filter inserted at `layer_idx`."""
    encoder, head = get_mlm_components(masked_lm)
    df = _pairs_to_df(pairs)
    if flt is None or layer_idx is None:
        return compute_pll_bias(encoder, head, tokenizer, df)
    with insert_filter(masked_lm, layer_idx, flt):
        return compute_pll_bias(encoder, head, tokenizer, df)
