"""Loaders for Papers 2-5 result CSVs. Paper 6 REUSES these as baselines instead
of recomputing them, which is where the GPU-hour savings come from.

Each loader returns a list of dict rows (or a derived structure). A reuse audit
records which baseline files were found and loaded.
"""
import os
from typing import Dict, List, Optional

from Common_00.common import read_csv_dicts, resolve


def _p(config: Dict, key: str, fname: str) -> str:
    return resolve(os.path.join(config["paths"][key], fname))


# ---- direct file loaders -------------------------------------------------- #
def filter_results(config) -> List[Dict]:        # Paper 4: per-language filter FBR
    return read_csv_dicts(_p(config, "paper4_results", "filter_results.csv"))


def cut_results(config) -> List[Dict]:            # Paper 5: learned/random/magnitude cut
    return read_csv_dicts(_p(config, "paper5_results", "cut_results.csv"))


def circuit_overlap(config) -> List[Dict]:        # Paper 5: Jaccard ~0.035 baseline
    return read_csv_dicts(_p(config, "paper5_results", "circuit_overlap.csv"))


def minimal_cut(config) -> List[Dict]:            # Paper 5: emergence band + total_units
    return read_csv_dicts(_p(config, "paper5_results", "minimal_cut.csv"))


def emergence_layers(config) -> List[Dict]:       # Paper 4: emergence layer per model
    return read_csv_dicts(_p(config, "paper4_results", "emergence_layers.csv"))


def competence(config) -> List[Dict]:             # Paper 4: HellaSwag competence
    return read_csv_dicts(_p(config, "paper4_results", "competence_by_language.csv"))


def llm_clti(config) -> List[Dict]:               # Paper 3: expressed-bias baseline
    return read_csv_dicts(_p(config, "paper3_results", "llm_clti.csv"))


def transfer_results(config) -> List[Dict]:       # Paper 2: projection transfer
    return read_csv_dicts(_p(config, "paper2_results", "transfer_results.csv"))


# ---- derived helpers ------------------------------------------------------ #
def emergence_band_for(config, subject_model: str, dataset: str) -> Optional[List[int]]:
    """Reuse the localized band ('7|8|9') from Paper 5; do not re-localize globally."""
    for r in minimal_cut(config):
        if r.get("subject_model") == subject_model and r.get("dataset") == dataset:
            band = r.get("emergence_band", "")
            if band:
                return [int(x) for x in band.split("|") if x != ""]
    # fall back to Paper 4 emergence layer +/- k
    for r in emergence_layers(config):
        if r.get("subject_model") == subject_model and r.get("dataset") == dataset:
            try:
                L = int(r.get("peak_layer"))
                return [L - 1, L, L + 1]
            except Exception:
                pass
    return None


def competence_gate(config) -> Dict:
    """(subject_model, language) -> bool adequate (>=0.40)."""
    out = {}
    for r in competence(config):
        try:
            ok = str(r.get("adequate", "")).lower() == "true" or \
                 float(r.get("hellaswag_accuracy", 0)) >= 0.40
        except Exception:
            ok = False
        out[(r.get("subject_model"), r.get("language"))] = ok
    return out


def reuse_audit(config) -> Dict:
    """Report which baseline files exist (loaded, not recomputed)."""
    checks = {
        "paper2/transfer_results.csv": _p(config, "paper2_results", "transfer_results.csv"),
        "paper3/llm_clti.csv": _p(config, "paper3_results", "llm_clti.csv"),
        "paper4/filter_results.csv": _p(config, "paper4_results", "filter_results.csv"),
        "paper4/emergence_layers.csv": _p(config, "paper4_results", "emergence_layers.csv"),
        "paper4/competence_by_language.csv": _p(config, "paper4_results", "competence_by_language.csv"),
        "paper5/cut_results.csv": _p(config, "paper5_results", "cut_results.csv"),
        "paper5/circuit_overlap.csv": _p(config, "paper5_results", "circuit_overlap.csv"),
        "paper5/minimal_cut.csv": _p(config, "paper5_results", "minimal_cut.csv"),
    }
    return {name: {"path": path, "found": os.path.exists(path),
                   "rows": len(read_csv_dicts(path)) if os.path.exists(path) else 0}
            for name, path in checks.items()}
