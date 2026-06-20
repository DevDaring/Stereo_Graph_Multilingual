"""Reading and normalising the parallel bias datasets for KG construction and the
integrity checks. CPU only. Mirrors the parsing convention of paper5/src/data_loader.py.
"""
import ast
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List

from Common_00.common import resolve


def dataset_path(config: Dict, dataset_key: str) -> str:
    fname = config["data"][dataset_key]["file"]
    return resolve(os.path.join(config["paths"]["data_raw"], fname))


def parse_target(raw) -> str:
    """Normalise a Target cell. Cells may be a stringified list (\"['she']\") or a
    plain string. Returns a single clean surface form."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)) and val:
                s = str(val[0])
        except Exception:
            s = s.strip("[]'\" ")
    return re.sub(r"\s+", " ", s).strip().strip("'\"")


def read_bias_rows(config: Dict, dataset_key: str) -> List[Dict]:
    """Return normalised rows with keys: index, lang, bias_type, group_stereo,
    group_anti, sentence."""
    path = dataset_path(config, dataset_key)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "index": str(r.get("Index", "")).strip(),
                "lang": str(r.get("language", "")).strip(),
                "bias_type": str(r.get("bias_type", "")).strip(),
                "group_stereo": parse_target(r.get("Target_Stereotypical")),
                "group_anti": parse_target(r.get("Target_Anti-Stereotypical")),
                "sentence": str(r.get("Sentence", "")),
            })
    return rows


def alignment_key(r: Dict) -> tuple:
    """The cross-lingual alignment key. indian_bias holds several bias_type rows per
    Index, so the unique parallel unit is (Index, bias_type) - not Index alone."""
    return (r["index"], r["bias_type"])


def group_by_index(rows: List[Dict]) -> Dict[tuple, Dict[str, Dict]]:
    """(index, bias_type) -> {lang -> row}. Rows sharing this key across languages are
    translations of each other, which is the cross-lingual backbone of the KG. Using
    (index, bias_type) avoids collapsing the multiple bias_type rows that indian_bias
    stores under one Index."""
    out = defaultdict(dict)
    for r in rows:
        if r["index"] and r["lang"]:
            out[alignment_key(r)][r["lang"]] = r
    return out
