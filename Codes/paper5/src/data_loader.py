"""
data_loader.py - Load the trilingual bias datasets and build sentence pairs (Paper 5).

Reuses the SAME raw CSV files as Papers 1, 2 and 3 (full dataset symmetry):
  - multicrows_pairs.csv          (CrowS-Pairs, 9 categories, en/hi/bn)
  - indian_multilingual_bias.csv  (caste/religion/gender/race, en/hi/bn)

Integrity is re-checked on EVERY run: duplicate rows, corrupted rows (missing
MASK placeholder, empty/malformed target words, MASK fill that produces identical
sentences) are reported and dropped. A sha256 content hash is recorded so silent
data changes become visible across reruns. The data directory may be overridden
with env PAPER5_DATA_DIR (falls back to PAPER3_DATA_DIR, then the config path).
"""

# =====================================================================
# CITATION(S) for this module:
#   [nangia2020crows] Nangia et al., "CrowS-Pairs," EMNLP 2020.
#     Implements: stereotypical vs anti-stereotypical paired sentences.
# =====================================================================

import ast
import hashlib
import json
import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("data_loader")

MASK_TOKEN = "MASK"


def _resolve_data_dir(config: Dict, this_dir: str) -> str:
    for env_name in ("PAPER5_DATA_DIR", "PAPER3_DATA_DIR"):
        override = os.getenv(env_name)
        if override and override.strip():
            return override.strip()
    rel = config["paths"]["data_raw"]
    return os.path.normpath(os.path.join(this_dir, rel))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_target(raw) -> Optional[str]:
    """Parse a target cell like "['black']" -> "black"; None if malformed/empty."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            val = [str(x).strip() for x in val if str(x).strip()]
            return val[0] if val else None
        token = str(val).strip()
        return token or None
    except (ValueError, SyntaxError):
        token = s.strip("[]'\" ")
        return token or None


def _fill(sentence: str, target: str) -> str:
    """Replace the whole-word MASK placeholder with the target word (literal)."""
    return re.sub(rf"\b{MASK_TOKEN}\b", lambda _m: target, sentence)


def load_bias_dataset(dataset_key: str, config: Dict, this_dir: str,
                      languages: Optional[List[str]] = None) -> Dict:
    """Load one bias dataset; return {'rows': [...], 'integrity': {...}}."""
    ds_cfg = config["data"][dataset_key]
    data_dir = _resolve_data_dir(config, this_dir)
    path = os.path.join(data_dir, ds_cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"[FAIL] Dataset file not found: {path}")

    langs = languages or ds_cfg["languages"]
    df = pd.read_csv(path)
    required = {"Target_Stereotypical", "Target_Anti-Stereotypical", "Sentence",
                "language", "bias_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[FAIL] {ds_cfg['file']} missing columns: {missing}")

    total_raw = len(df)
    rows: List[Dict] = []
    dropped_corrupt = 0
    duplicates = 0
    seen = set()

    for idx, r in df.iterrows():
        lang = str(r["language"]).strip().lower()
        if lang not in langs:
            continue
        sentence = str(r["Sentence"]) if not pd.isna(r["Sentence"]) else ""
        if MASK_TOKEN not in sentence:
            dropped_corrupt += 1
            continue
        stereo = _parse_target(r["Target_Stereotypical"])
        anti = _parse_target(r["Target_Anti-Stereotypical"])
        if not stereo or not anti:
            dropped_corrupt += 1
            continue
        bias_type = str(r["bias_type"]).strip().lower()
        sig = (lang, bias_type, sentence, stereo, anti)
        if sig in seen:
            duplicates += 1
            continue
        seen.add(sig)
        sent_stereo = _fill(sentence, stereo)
        sent_anti = _fill(sentence, anti)
        if sent_stereo == sent_anti:
            dropped_corrupt += 1
            continue
        rows.append({
            "dataset": dataset_key,
            "row_index": int(r.get("Index", idx)),
            "language": lang,
            "bias_type": bias_type,
            "target_stereotypical": stereo,
            "target_anti_stereotypical": anti,
            "sentence_template": sentence,
            "sentence_stereotypical": sent_stereo,
            "sentence_anti_stereotypical": sent_anti,
        })

    per_language, per_category = {}, {}
    for row in rows:
        per_language[row["language"]] = per_language.get(row["language"], 0) + 1
        per_category[row["bias_type"]] = per_category.get(row["bias_type"], 0) + 1

    integrity = {
        "dataset": dataset_key, "file": ds_cfg["file"], "sha256": file_sha256(path),
        "total_rows_in_file": total_raw, "valid_pairs": len(rows),
        "dropped_corrupt": dropped_corrupt, "duplicates_removed": duplicates,
        "per_language": per_language, "per_category": per_category,
    }
    logger.info("[OK] %s: %d valid pairs (%d corrupt dropped, %d duplicates removed) | langs=%s",
                ds_cfg["file"], len(rows), dropped_corrupt, duplicates, per_language)
    if duplicates:
        logger.warning("[WARN] %s contained %d duplicate rows (removed).", ds_cfg["file"], duplicates)
    if dropped_corrupt:
        logger.warning("[WARN] %s contained %d corrupt/uninformative rows (dropped).",
                       ds_cfg["file"], dropped_corrupt)
    return {"rows": rows, "integrity": integrity}


def load_hellaswag_subset(config: Dict, this_dir: str, per_lang: Optional[int] = None) -> List[Dict]:
    """Load a small balanced multilingual HellaSwag subset for the competence probe."""
    ds_cfg = config["data"]["hellaswag"]
    data_dir = _resolve_data_dir(config, this_dir)
    path = os.path.join(data_dir, ds_cfg["file"])
    if not os.path.exists(path):
        logger.warning("[WARN] HellaSwag file not found at %s; competence probe will be skipped.", path)
        return []
    cap = per_lang or ds_cfg.get("competence_subset_per_language",
                                 ds_cfg.get("utility_subset_per_language", 500))
    by_lang: Dict[str, List[Dict]] = {}
    dropped = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            lang = str(item.get("language", item.get("lang", "en"))).lower()
            if lang not in ds_cfg["languages"]:
                continue
            # `endings` is stored as a stringified JSON list ("[...]"); parse it to a
            # real list of 4 strings, and coerce `label` to int. Items that cannot be
            # normalised are dropped so the competence probe never iterates characters.
            endings = item.get("endings")
            if isinstance(endings, str):
                try:
                    endings = ast.literal_eval(endings)
                except (ValueError, SyntaxError):
                    dropped += 1
                    continue
            if not isinstance(endings, (list, tuple)) or len(endings) < 2:
                dropped += 1
                continue
            try:
                label = int(item.get("label"))
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (0 <= label < len(endings)):
                dropped += 1
                continue
            clean = {"ctx": str(item.get("ctx", "")),
                     "endings": [str(e) for e in endings],
                     "label": label, "language": lang}
            bucket = by_lang.setdefault(lang, [])
            if len(bucket) < cap:
                bucket.append(clean)
    out = [x for bucket in by_lang.values() for x in bucket]
    logger.info("[OK] HellaSwag subset: %d items (%s) | %d malformed dropped",
                len(out), {k: len(v) for k, v in by_lang.items()}, dropped)
    return out
