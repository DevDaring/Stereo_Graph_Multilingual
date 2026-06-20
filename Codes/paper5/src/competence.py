"""
competence.py - Per-language comprehension control (Stage A, shared by Papers 4 and 5).

Several subject models do not officially support Bengali (Aya-23 lists 23 languages
WITHOUT Bengali; Llama-3.3 lists Hindi but not Bengali). A model that shows "low
bias" in a language it cannot read is not fair, only confused. This module measures
comprehension directly with multilingual HellaSwag (4-way next-sentence choice) so
every later bias claim can be split into a genuine bias gap versus a competence gap.

For a causal LLM the score is HellaSwag accuracy by total continuation
log-likelihood (chance = 0.25). A per-token negative-log-likelihood "fluency"
proxy is also recorded. The English -> Hindi -> Bengali fall in accuracy is the
support gradient referenced throughout the paper.
"""

# =====================================================================
# CITATION(S) for this module:
#   [zellers2019hellaswag] Zellers et al., "HellaSwag," ACL 2019. arXiv:1905.07830.
# =====================================================================

import logging
from typing import Dict, List

logger = logging.getLogger("competence")


def _hellaswag_items_by_language(items: List[Dict]) -> Dict[str, List[Dict]]:
    by_lang: Dict[str, List[Dict]] = {}
    for it in items:
        lang = str(it.get("language", it.get("lang", "en"))).lower()
        by_lang.setdefault(lang, []).append(it)
    return by_lang


def measure_competence_causal(llm, items: List[Dict], adequacy_threshold: float = 0.40) -> List[Dict]:
    """
    HellaSwag accuracy per language for a causal LLM (uses sequence_log_likelihood).
    `items` rows need: ctx (str), endings (list of 4 str), label (int 0..3), language.
    Returns one dict per language with accuracy, fluency_nll, n, adequate flag.
    """
    out = []
    for lang, rows in _hellaswag_items_by_language(items).items():
        correct = 0
        nll_sum = 0.0
        nll_tokens = 0
        n = 0
        for r in rows:
            endings = r.get("endings") or []
            if len(endings) < 2 or r.get("label") is None:
                continue
            ctx = str(r.get("ctx", ""))
            scores = []
            for end in endings:
                lp, ntok = llm.sequence_log_likelihood(f"{ctx} {end}")
                scores.append(lp / max(ntok, 1))   # length-normalised
                nll_sum += -lp
                nll_tokens += ntok
            pred = max(range(len(scores)), key=lambda i: scores[i])
            correct += int(pred == int(r["label"]))
            n += 1
        acc = correct / n if n else 0.0
        fluency = nll_sum / nll_tokens if nll_tokens else float("nan")
        out.append({
            "language": lang, "hellaswag_acc": round(acc, 4),
            "fluency_nll": round(fluency, 4), "n": n, "chance": 0.25,
            "adequate": bool(acc >= adequacy_threshold),
        })
        logger.info("[OK] competence %s: acc=%.3f (n=%d) adequate=%s",
                    lang, acc, n, acc >= adequacy_threshold)
    return out
