"""
train_filter.py - Stage C training: fit the graph bias-filter with a LAYER-LOCAL
objective, so gradients never flow through the frozen base model (cheap on an L4).

Protocol:
  1. Capture the residual stream at the emergence layer L* for stereotypical and
     anti-stereotypical sentences (English only, for the transfer test).
  2. Estimate the stereotype direction u at L* (mean(stereo) - mean(anti), pooled),
     normalised. This is the axis a fair representation should not encode.
  3. Train the GraphBiasFilter so that filtered token representations have minimal
     projection onto u (bias removal) while staying close to the originals
     (reconstruction / utility preservation):

         L = mean(|<filter(H), u>|)  +  lambda_recon * ||filter(H) - H||^2

The trained filter is then inserted at L* as a forward hook for end-to-end
bias/utility evaluation (transfer_eval.py).
"""

# =====================================================================
# CITATION(S) for this module:
#   [ravfogel2020inlp] Ravfogel et al., "Null It Out (INLP)," ACL 2020.
#     arXiv:2004.07667 (removing a concept direction from representations).
#   [liang2020sentencedebias] Liang et al., "SentenceDebias," ACL 2020.
#     arXiv:2007.08100 (bias-direction estimation from definitional pairs).
# =====================================================================

import logging
from typing import Dict, List

import torch

from .graph_filter import GraphBiasFilter

logger = logging.getLogger("train_filter")


def _capture_layer_states(llm, sentences: List[str], layer_idx: int) -> List[torch.Tensor]:
    """Residual stream at `layer_idx` (hidden_states index layer_idx+1) per sentence."""
    states = []
    for s in sentences:
        hs = llm.hidden_states(s)                  # [L+1, T, d]
        states.append(hs[layer_idx + 1])           # [T, d]
    return states


def estimate_bias_direction(stereo_states: List[torch.Tensor],
                            anti_states: List[torch.Tensor]) -> torch.Tensor:
    """Normalised stereotype direction u at the layer (mean pooled difference)."""
    def _pool(states):
        return torch.stack([s.mean(dim=0) for s in states], dim=0).mean(dim=0)  # [d]
    u = _pool(stereo_states) - _pool(anti_states)
    return u / (u.norm() + 1e-8)


def train_filter_from_states(stereo_states: List[torch.Tensor], anti_states: List[torch.Tensor],
                             cfg: Dict, layer_idx: int = -1) -> GraphBiasFilter:
    """
    Architecture-agnostic core: fit a GraphBiasFilter from already-captured layer
    states (works for both encoders and decoder LLMs). Layer-local objective; no
    backprop through any base model.
    """
    if not stereo_states:
        raise RuntimeError("[FAIL] No layer states captured for filter training.")
    rank = cfg.get("rank", 64)
    knn = cfg.get("knn", 8)
    epochs = cfg.get("epochs", 30)
    lr = cfg.get("lr", 1e-3)
    lambda_recon = cfg.get("lambda_recon", 1.0)

    dim = stereo_states[0].shape[-1]
    # Train the tiny filter on the GPU (the captured states are small; running this
    # on CPU leaves the H100 idle and is ~10-50x slower).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u = estimate_bias_direction(stereo_states, anti_states).to(device)   # [d]
    flt = GraphBiasFilter(dim, rank=rank, knn=knn).to(device)
    opt = torch.optim.Adam(flt.parameters(), lr=lr)
    all_states = [H.float().to(device) for H in (stereo_states + anti_states)]
    logger.info("[INFO] training filter @layer %d | params=%d | examples=%d | device=%s",
                layer_idx, flt.param_count(), len(all_states), device)

    for ep in range(epochs):
        total = 0.0
        for H in all_states:
            opt.zero_grad()
            Hf = flt(H)                                    # [T, d]
            bias_loss = (Hf @ u).abs().mean()              # projection onto bias axis
            recon = ((Hf - H) ** 2).mean()
            loss = bias_loss + lambda_recon * recon
            loss.backward()
            opt.step()
            total += float(loss.item())
        if (ep + 1) % max(1, epochs // 5) == 0:
            logger.info("[INFO] filter epoch %d/%d loss=%.5f", ep + 1, epochs, total / len(all_states))
    flt.eval()
    return flt


def train_filter(llm, pairs: List[Dict], layer_idx: int, cfg: Dict) -> GraphBiasFilter:
    """Fit a GraphBiasFilter at `layer_idx` from English stereo/anti pairs (decoder LLM)."""
    llm.load()
    pairs = pairs[:cfg.get("max_train_pairs", 400)]
    stereo = _capture_layer_states(llm, [p["sentence_stereotypical"] for p in pairs], layer_idx)
    anti = _capture_layer_states(llm, [p["sentence_anti_stereotypical"] for p in pairs], layer_idx)
    return train_filter_from_states(stereo, anti, cfg, layer_idx)
