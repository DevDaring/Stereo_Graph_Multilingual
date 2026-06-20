"""
env_utils.py - Environment / secret loading and round-robin key rotation.

All secrets load from .env via python-dotenv. Nothing is hard-coded.
Key values are never logged; only NAMES and presence (True/False) are printed.
"""

# =====================================================================
# CITATION(S) for this module: none (infrastructure utility).
# =====================================================================

import logging
import os
import threading
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger("env_utils")


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return an environment variable value, or default if unset/empty."""
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()


def first_present(names: List[str], default: Optional[str] = None) -> Optional[str]:
    """Return the value of the first env var in `names` that is set and non-empty."""
    for n in names:
        v = get_env(n)
        if v is not None:
            return v
    return default


def collect_keys(names: List[str]) -> List[str]:
    """
    Collect all non-empty key values for the given env var NAMES.
    De-duplicates while preserving order, so listing both GCP_Key* and
    GEMINI_API_KEY_* aliases never yields the same key twice.
    """
    keys, seen = [], set()
    for n in names:
        v = get_env(n)
        if v and v not in seen:
            keys.append(v)
            seen.add(v)
    return keys


class RoundRobin:
    """
    Thread-safe round-robin selector over a list of API keys for one provider.
    There is NO cross-provider fallback; rotate within one provider only.
    A threading.Lock guards the counter so concurrent callers each get a
    distinct key without races.
    """

    def __init__(self, keys: List[str], provider: str):
        self.provider = provider
        self.keys = list(keys)
        self._idx = 0
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return len(self.keys) > 0

    def next(self) -> str:
        if not self.keys:
            raise RuntimeError(f"[FAIL] No API keys configured for provider '{self.provider}'.")
        with self._lock:
            key = self.keys[self._idx % len(self.keys)]
            self._idx += 1
            return key

    def count(self) -> int:
        return len(self.keys)


def mask(value: Optional[str]) -> str:
    """Render a secret as a safe presence indicator for logs (never the value)."""
    if not value:
        return "MISSING"
    return f"present(len={len(value)})"


import re as _re

_SECRET_RE = [
    _re.compile(r"(key=)[^&\s\"']+", _re.IGNORECASE),     # URL query ?key=...
    _re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+"),          # Authorization headers
    _re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),              # Google API key
    _re.compile(r"sk-[A-Za-z0-9\-]{10,}"),                # OpenAI / DeepSeek style
    _re.compile(r"hf_[A-Za-z0-9]{10,}"),                  # Hugging Face token
]


def redact(text) -> str:
    """Remove anything key-like from a string before it is written to a report or log."""
    s = str(text)
    s = _SECRET_RE[0].sub(r"\1REDACTED", s)
    s = _SECRET_RE[1].sub(r"\1REDACTED", s)
    for pat in _SECRET_RE[2:]:
        s = pat.sub("REDACTED", s)
    return s
