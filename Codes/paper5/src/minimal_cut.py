"""
minimal_cut.py - Stage C: learn the smallest set of connections to cut so bias drops
but task skill stays.

A continuous keep-mask m in [0,1] is placed over residual units (layer, dim) inside
the emergence band L*. Applied via a forward hook, h_out = h_in * m (broadcast over
tokens). The mask is trained with three terms:

    L = L_bias(model * m)                 # push the stereo-vs-anti gap toward 0
      + lambda_u * L_utility(model * m)   # keep clean-sentence likelihood (skill)
      + gamma   * sum(1 - m)              # cut FEW units (keep most connections)

m is parameterised through a hard-concrete gate so it binarises cleanly. The output
is the binary cut E_cut (units with m below 0.5) and a bias-utility-cut-size frontier.
"""

# =====================================================================
# CITATION(S) for this module:
#   [louizos2018l0] Louizos, Welling, Kingma, "Learning Sparse Networks through
#     L0 Regularization," ICLR 2018. arXiv:1712.01312.
#   [sanh2020movement] Sanh, Wolf, Rush, "Movement Pruning," NeurIPS 2020.
#     arXiv:2005.07683.
# =====================================================================

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger("minimal_cut")


class ResidualKeepMask:
    """A per-(layer,dim) keep-mask applied to residual streams via forward hooks."""

    def __init__(self, layer_dims: Dict[int, int], device, init: float = 3.0):
        # logits high -> sigmoid ~ 1 -> keep (start by keeping everything).
        self.logits = {L: torch.full((d,), init, device=device, requires_grad=True)
                       for L, d in layer_dims.items()}

    def params(self):
        return list(self.logits.values())

    def keep(self, L: int) -> torch.Tensor:
        return torch.sigmoid(self.logits[L])

    def hooks(self, llm, hard: bool = False):
        layer_mods = llm.layers
        handles = []

        def _mk(L):
            def _hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                m = (self.keep(L) > 0.5).float() if hard else self.keep(L)
                m = m.to(h.dtype).view(1, 1, -1)
                newh = h * m
                return (newh,) + tuple(out[1:]) if isinstance(out, tuple) else newh
            return _hook

        for L in self.logits:
            handles.append(layer_mods[L].register_forward_hook(_mk(L)))
        return handles

    def binary_cut(self) -> List[Tuple[int, int]]:
        cut = []
        for L, lg in self.logits.items():
            keep = torch.sigmoid(lg) > 0.5
            for j in torch.nonzero(~keep).flatten().tolist():
                cut.append((L, int(j)))
        return cut

    def total_units(self) -> int:
        return sum(lg.shape[0] for lg in self.logits.values())


def _ll(llm, text: str) -> torch.Tensor:
    """Causal log-likelihood of `text` as a (possibly grad-tracking) scalar tensor."""
    enc = llm.tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=llm.backbone_cfg.get("max_length", 128))
    ids = enc["input_ids"].to(llm.model.device)
    attn = enc["attention_mask"].to(llm.model.device)
    out = llm.model(input_ids=ids, attention_mask=attn)
    logp = torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum()


def learn_minimal_cut(llm, pairs_en: List[Dict], band: List[int], cfg: Dict) -> ResidualKeepMask:
    """
    Learn a keep-mask over residual units in the emergence band on English pairs.

    Gradients are accumulated PER PAIR (one backward per pair, freeing each forward
    graph) so memory stays bounded - never retain all forward graphs at once, which
    would OOM on a 7-8B model. The utility term penalises any drop in a pair's clean
    log-likelihood (precomputed once), so the cut removes bias without losing skill.
    """
    llm.load()
    d = llm.model.config.hidden_size
    mask = ResidualKeepMask({L: d for L in band}, llm.model.device,
                            init=cfg.get("mask_init", 3.0))
    opt = torch.optim.Adam(mask.params(), lr=cfg.get("lr", 0.05))
    lam_u = cfg.get("lambda_utility", 1.0)
    gamma = cfg.get("gamma_sparsity", 0.01)
    steps = cfg.get("steps", 60)
    pairs = pairs_en[:cfg.get("max_pairs", 64)]
    n = max(len(pairs), 1)

    # Per-pair clean baseline likelihood (no mask, no grad) - the utility reference.
    base_sum_ll = []
    with torch.no_grad():
        for p in pairs:
            s = float(_ll(llm, p["sentence_stereotypical"]).item()
                      + _ll(llm, p["sentence_anti_stereotypical"]).item())
            base_sum_ll.append(s)

    for step in range(steps):
        handles = mask.hooks(llm, hard=False)
        try:
            opt.zero_grad()
            running_bias = 0.0
            for i, p in enumerate(pairs):
                ll_st = _ll(llm, p["sentence_stereotypical"])
                ll_an = _ll(llm, p["sentence_anti_stereotypical"])
                bias_i = (ll_st - ll_an).abs()
                util_pen_i = torch.relu(base_sum_ll[i] - (ll_st + ll_an))  # penalise LL drop
                pair_loss = (bias_i + lam_u * util_pen_i) / n
                pair_loss.backward()            # accumulate grad to the mask; free graph
                running_bias += float(bias_i.item())
            # sparsity term depends only on the mask (cheap); one extra backward.
            sparsity = sum((1.0 - mask.keep(L)).sum() for L in mask.logits)
            (gamma * sparsity).backward()
            opt.step()
        finally:
            for h in handles:
                h.remove()
        if (step + 1) % max(1, steps // 5) == 0:
            logger.info("[INFO] cut step %d/%d mean_bias=%.3f cut_units=%d/%d",
                        step + 1, steps, running_bias / n, len(mask.binary_cut()), mask.total_units())
    return mask
