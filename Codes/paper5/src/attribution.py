"""
attribution.py - Stage B: attribute stereotype bias to connections (units) of the model.

A "connection/unit" here is a residual-stream coordinate at a layer: the value at
hidden dimension j of layer L. Attribution patching estimates each unit's influence
on the bias signal in a forward + backward pass, instead of brute-force patching
every unit one at a time. For a sentence the influence of unit (L, j) on the
sentence log-likelihood is approximated by grad * activation:

    attr(L, j) = ( d LL / d h[L, :, j] ) . h[L, :, j]     (summed over tokens)

The bias attribution of a pair is attr(stereo) - attr(anti); larger magnitude means
the unit pushes the model toward the stereotype. Attribution is restricted to the
emergence band L* (from Paper 4 / localize.py) to keep the search small.
"""

# =====================================================================
# CITATION(S) for this module:
#   [syed2023eap] Syed, Rager, Conmy, "Attribution Patching Outperforms
#     Automated Circuit Discovery," 2023. arXiv:2310.10348.
#   [kramar2024atp] Kramar et al., "AtP*," 2024. arXiv:2403.00745.
#   [dai2022knowledge] Dai et al., "Knowledge Neurons," ACL 2022. arXiv:2104.08696.
# =====================================================================

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("attribution")


def _ll_with_layer_grads(llm, text: str, layers: List[int]) -> Dict[int, torch.Tensor]:
    """
    Forward `text`, retain grad on each requested layer output, backward the total
    log-likelihood, and return {layer: (grad * activation) summed over tokens} as a
    per-dimension attribution vector [d] on CPU.
    """
    llm.load()
    enc = llm.tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=llm.backbone_cfg.get("max_length", 128))
    input_ids = enc["input_ids"].to(llm.model.device)
    attn = enc["attention_mask"].to(llm.model.device)

    captured: Dict[int, torch.Tensor] = {}
    handles = []
    layer_mods = llm.layers

    def _mk_hook(L):
        def _hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad()
            captured[L] = h
            return out
        return _hook

    for L in layers:
        handles.append(layer_mods[L].register_forward_hook(_mk_hook(L)))

    try:
        out = llm.model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        ll = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum()
        if not ll.requires_grad:
            # No grad graph (input-require-grads not enabled); return zeros, never crash.
            return {L: torch.zeros(captured[L].shape[-1]) for L in captured}
        llm.model.zero_grad(set_to_none=True)
        ll.backward()
        attr = {}
        for L, h in captured.items():
            if h.grad is None:
                attr[L] = torch.zeros(h.shape[-1])
            else:
                attr[L] = (h.grad * h).sum(dim=1).squeeze(0).detach().float().cpu()  # [d]
        return attr
    finally:
        for hd in handles:
            hd.remove()


def attribute_units(llm, pairs: List[Dict], layers: List[int],
                    max_pairs: int = 600) -> Dict[int, torch.Tensor]:
    """
    Mean absolute bias attribution per unit, per layer in `layers`.
    Returns {layer: tensor[d]} where larger = more bias-relevant.
    """
    llm.load()
    # Frozen 4-bit weights + integer inputs leave no tensor requiring grad, so the
    # embedding output is made grad-requiring; this lets gradients flow to the
    # captured layer activations (attribution patching needs this gradient).
    try:
        llm.model.enable_input_require_grads()
    except Exception as e:
        logger.warning("[WARN] enable_input_require_grads failed (%s); attribution may be zero.",
                       str(e)[:120])
    acc: Dict[int, torch.Tensor] = {}
    n = 0
    for p in pairs[:max_pairs]:
        a_st = _ll_with_layer_grads(llm, p["sentence_stereotypical"], layers)
        a_an = _ll_with_layer_grads(llm, p["sentence_anti_stereotypical"], layers)
        for L in layers:
            diff = (a_st.get(L) - a_an.get(L)).abs()
            acc[L] = diff if L not in acc else acc[L] + diff
        n += 1
    if n == 0:
        return acc
    return {L: v / n for L, v in acc.items()}


def top_units(attribution: Dict[int, torch.Tensor], fraction: float) -> List:
    """Return a list of (layer, dim) for the top `fraction` of units by attribution."""
    flat = []
    for L, vec in attribution.items():
        for j in range(vec.shape[0]):
            flat.append((float(vec[j]), L, j))
    flat.sort(reverse=True)
    k = max(1, int(len(flat) * fraction))
    return [(L, j) for _s, L, j in flat[:k]]
