"""
json_extract.py - Robust extraction of the model's choice from its reply.

Strategy (judge-free first, as required):
  1. Parse strict JSON {"choice": "A"|"B", "refused": bool} directly.
  2. If that fails, regex-extract the first {...} block and parse it.
  3. If that fails, look for a bare 'A' / 'B' / refusal phrase.
  4. Only if all of the above fail does the caller fall back to an LLM judge.

Returns a dict: {"choice": "A"|"B"|None, "refused": bool, "method": str}.
"""

# =====================================================================
# CITATION(S) for this module: none (parsing utility).
# =====================================================================

import json
import re
from typing import Dict, Optional

_REFUSAL_PATTERNS = [
    r"\bi (?:cannot|can't|won't|will not|refuse|am not able)\b",
    r"\bas an ai\b",
    r"\bi'm sorry\b",
    r"\bnot appropriate\b",
    r"\bcannot answer\b",
]


def _norm_choice(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().upper()
    if s in ("A", "B"):
        return s
    if s.startswith("A"):
        return "A"
    if s.startswith("B"):
        return "B"
    return None


def parse_choice(text: str) -> Dict:
    """Extract the choice without an LLM judge. method='unparseable' if it fails."""
    if text is None:
        return {"choice": None, "refused": False, "method": "empty"}
    raw = text.strip()

    # 1. strict JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and ("choice" in obj or "refused" in obj):
            return {
                "choice": _norm_choice(obj.get("choice")),
                "refused": bool(obj.get("refused", False)),
                "method": "strict_json",
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. first {...} block
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and ("choice" in obj or "refused" in obj):
                return {
                    "choice": _norm_choice(obj.get("choice")),
                    "refused": bool(obj.get("refused", False)),
                    "method": "regex_json",
                }
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. refusal phrase
    low = raw.lower()
    for pat in _REFUSAL_PATTERNS:
        if re.search(pat, low):
            return {"choice": None, "refused": True, "method": "refusal_phrase"}

    # 4. bare letter
    bare = re.search(r"\b([AB])\b", raw)
    if bare:
        return {"choice": bare.group(1).upper(), "refused": False, "method": "bare_letter"}

    return {"choice": None, "refused": False, "method": "unparseable"}
