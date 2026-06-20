"""
localize.py - Stage B: locate where stereotype bias emerges inside the model.

Two independent localisers; their agreement gives a trustworthy emergence band L*:

  1. logit-lens gap   : project each layer's residual stream through the model head
                        and read the stereotype-minus-anti token-logit gap; the layer
                        where this gap rises sharply is where bias accumulates.
  2. causal patching  : run a clean (anti) and corrupted (stereo) pass; patch one
                        layer's residual from corrupted into clean; the layer whose
                        patch most changes the bias metric is where bias is decided.

For causal LLMs both methods use the model's own residual stream. For encoders the
logit-lens analogue uses the MLM head at the masked target position.
"""

# =====================================================================
# CITATION(S) for this module:
#   [belrose2023tunedlens] Belrose et al., 2023. arXiv:2303.08112 (logit lens).
#   [vig2020causal] Vig et al., "Causal Mediation Analysis for Interpreting
#     Neural NLP: The Case of Gender Bias," NeurIPS 2020. arXiv:2004.12265.
#   [meng2022rome] Meng et al., "Locating and Editing Factual Associations,"
#     NeurIPS 2022. arXiv:2202.05262 (activation patching).
# =====================================================================

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("localize")


def llm_logit_lens_trajectory(llm, pairs: List[Dict], max_pairs: int = 400) -> List[float]:
    """
    Mean per-layer logit-lens gap over pairs. For each pair the two target words
    (stereotypical, anti) are the candidate continuation tokens, and the prompt is
    the shared sentence template up to the target slot.
    Returns [mean_gap_layer_0, ..., mean_gap_layer_L].
    """
    acc: Optional[List[float]] = None
    n = 0
    for p in pairs[:max_pairs]:
        prompt = p["sentence_template"].split("MASK")[0].strip()
        if not prompt:
            continue   # MASK-initial template: no left context to read a logit-lens gap
        gaps = llm.logit_lens_gap(prompt, p["target_stereotypical"], p["target_anti_stereotypical"])
        if not gaps:
            continue
        if acc is None:
            acc = [0.0] * len(gaps)
        for i, g in enumerate(gaps):
            acc[i] += g
        n += 1
    if not acc or n == 0:
        return []
    return [a / n for a in acc]


@torch.no_grad()
def llm_causal_patch_effect(llm, pairs: List[Dict], max_pairs: int = 200) -> List[float]:
    """
    Per-layer causal patching effect on the intrinsic bias signal.
    For each pair, run the anti sentence (clean) and the stereo sentence (corrupt),
    capture the corrupt residual at every layer, then re-run clean while patching
    layer L's last-position residual with the corrupt one; record how much the
    sentence log-likelihood gap shifts toward the stereotype. The per-layer mean
    absolute shift is the causal effect profile.
    CITATION: vig2020causal, meng2022rome.
    """
    llm.load()
    nL = llm.num_layers
    effect = [0.0] * nL
    n = 0
    for p in pairs[:max_pairs]:
        s_anti = p["sentence_anti_stereotypical"]
        s_stereo = p["sentence_stereotypical"]
        # corrupt-run residuals (per layer, last position)
        hs_corrupt = llm.hidden_states(s_stereo)            # [L+1, T, d]
        base_lp, _ = llm.sequence_log_likelihood(s_anti)
        enc = llm.tokenizer(s_anti, return_tensors="pt", truncation=True,
                            max_length=llm.backbone_cfg.get("max_length", 128))
        input_ids = enc["input_ids"].to(llm.model.device)
        attn = enc["attention_mask"].to(llm.model.device)
        T = input_ids.shape[1]
        T_min = min(T, hs_corrupt.shape[1])
        for L in range(nL):
            # Patch the residual at ALL scored positions, not just the last token:
            # the last position predicts beyond the sequence and is never scored, so
            # patching it alone produced an all-zero causal profile.
            patch_block = hs_corrupt[L + 1, :T_min, :].to(llm.model.device)   # [T_min, d]

            def _patch(hidden, _block=patch_block, _tm=T_min):
                hidden = hidden.clone()
                hidden[:, :_tm, :] = _block.to(hidden.dtype).unsqueeze(0)
                return hidden

            with llm.insert_hook(L, _patch):
                out = llm.model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                lp = torch.log_softmax(logits, dim=-1).gather(
                    -1, targets.unsqueeze(-1)).squeeze(-1).sum().item()
            effect[L] += abs(lp - base_lp)
        n += 1
    if n == 0:
        return effect
    return [e / n for e in effect]


def emergence_band(causal_effect: List[float], k: int = 3) -> List[int]:
    """Top-k layers by causal effect (the bias-emergence band L*)."""
    if not causal_effect:
        return []
    order = sorted(range(len(causal_effect)), key=lambda i: causal_effect[i], reverse=True)
    return sorted(order[:k])


def band_from_lens(logit_lens_gap: List[float], k: int = 3) -> List[int]:
    """Emergence band from the logit-lens trajectory: the contiguous window around
    the layer where the stereotype-vs-anti gap CHANGES most (its marginal
    contribution). Unlike raw causal patching, the per-layer marginal is not
    confounded by network depth, so it does not collapse onto the earliest layers.
    CITATION: nostalgebraist2020logitlens, geva2022transformer.
    """
    n = len(logit_lens_gap)                          # nL + 1 (entry 0 = embeddings)
    if n < 4:
        return list(range(max(0, n - 1)))
    # Marginal contribution of transformer layer L (0-indexed) = |gap[L+1] - gap[L]|,
    # so the band is returned in transformer-layer indices (0..nL-1), matching the
    # causal-patch and filter/cut conventions.
    marg = [abs(logit_lens_gap[L + 1] - logit_lens_gap[L]) for L in range(n - 1)]  # len nL
    nL = len(marg)
    cand = list(range(1, nL - 1)) or list(range(nL))  # drop first + final layer (output-trivial)
    peak = max(cand, key=lambda L: marg[L])
    half = k // 2
    lo = max(0, min(peak - half, nL - k))
    return list(range(lo, min(nL, lo + k)))
