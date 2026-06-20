"""Paper 6 shared utilities (CPU): config/env loading, round-robin keys,
resume-capable CSV I/O, and robust answer/JSON extraction.

No secret value ever appears in this file. All keys load from .env at runtime.
"""
import csv
import hashlib
import itertools
import json
import os
import random
import re
import threading
from typing import Dict, Iterable, List, Optional

import yaml
from dotenv import find_dotenv, load_dotenv

# Load .env once, from the nearest .env up the tree (repo-root .gitignore protects it).
load_dotenv(find_dotenv(usecwd=True))


# --------------------------------------------------------------------------- #
# Paths and config
# --------------------------------------------------------------------------- #
def paper6_root() -> str:
    """Absolute path to Codes/paper6/ (parent of this file's folder)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: Optional[str] = None) -> Dict:
    if path is None:
        path = os.path.join(paper6_root(), "config", "default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_rel_to_paper6: str) -> str:
    """Resolve a config path (relative to paper6/) to an absolute path."""
    if os.path.isabs(path_rel_to_paper6):
        return path_rel_to_paper6
    return os.path.normpath(os.path.join(paper6_root(), path_rel_to_paper6))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Secrets / round-robin keys
# --------------------------------------------------------------------------- #
def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if (v is not None and v.strip() != "") else default


def get_keys(env_var_names: Iterable[str]) -> List[str]:
    """Return the non-empty values of the named env vars (preserving order)."""
    out = []
    for n in env_var_names:
        v = get_env(n)
        if v:
            out.append(v)
    return out


class RoundRobin:
    """Thread-safe round-robin over a fixed list of items (e.g., API keys)."""

    def __init__(self, items: List[str]):
        self._items = list(items)
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(self._items) if self._items else None

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def next(self) -> str:
        if not self._items:
            raise RuntimeError("RoundRobin has no items (no keys configured).")
        with self._lock:
            return next(self._cycle)


# --------------------------------------------------------------------------- #
# Resume-capable CSV I/O (incremental append; skip already-done rows)
# --------------------------------------------------------------------------- #
def read_csv_dicts(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def done_keys(path: str, key_cols: List[str]) -> set:
    """Set of composite keys already written to `path` (for --resume)."""
    rows = read_csv_dicts(path)
    return {tuple(str(r.get(c, "")) for c in key_cols) for r in rows}


def append_row(path: str, row: Dict, header: Optional[List[str]] = None) -> None:
    """Append one row, writing the header if the file is new. Atomic-enough for
    incremental checkpointing."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    cols = header or list(row.keys())
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Robust answer / JSON extraction from model text
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response, tolerating fences and prose."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def extract_choice(text: str, options: List[str]) -> Optional[str]:
    """Return the first option that the model picked, matching whole words,
    case-insensitively. Used for stereo/anti or A/B style judgements."""
    if not text:
        return None
    low = text.lower()
    # exact JSON answer first
    j = extract_json(text)
    if j:
        for v in j.values():
            if isinstance(v, str) and v.strip().lower() in [o.lower() for o in options]:
                return next(o for o in options if o.lower() == v.strip().lower())
    for opt in options:
        if re.search(r"\b" + re.escape(opt.lower()) + r"\b", low):
            return opt
    return None
