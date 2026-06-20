"""
reporting.py - Long-format result tables for Paper 5 (append-only, resume-safe).

Output files (under results/):
  competence_by_language.csv  - HellaSwag accuracy + fluency per model x language
  bias_attribution.csv        - top attributed units per model x dataset x layer
  minimal_cut.csv             - the chosen cut, its size, total units, fraction
  cut_results.csv             - learned/random/magnitude CBR per language (Pareto inputs)
  circuit_overlap.csv         - Jaccard overlap of per-language cut sets
  decomposition.csv           - alignment vs competence coefficients
  metrics_summary.csv         - Circuit-CLTI, beats-controls flag per model x dataset
  run_manifest.json / integrity_report.json / dry_run_report.json
"""

# =====================================================================
# CITATION(S) for this module: none (reporting utility).
# =====================================================================

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger("reporting")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResultWriter:
    FIELDS = {
        "competence_by_language.csv": [
            "timestamp_utc", "subject_model", "language", "hellaswag_accuracy",
            "fluency_nll", "n", "chance", "adequate",
        ],
        "bias_attribution.csv": [
            "timestamp_utc", "subject_model", "dataset", "layer", "dim",
            "attribution", "rank",
        ],
        "minimal_cut.csv": [
            "timestamp_utc", "subject_model", "dataset", "emergence_band",
            "cut_units", "total_units", "cut_fraction", "seed",
        ],
        "cut_results.csv": [
            "timestamp_utc", "subject_model", "dataset", "language", "variant",
            "cut_size", "bias_baseline", "bias_after_cut", "cut_bias_reduction",
            "n_pairs", "seed",
        ],
        "circuit_overlap.csv": [
            "timestamp_utc", "subject_model", "dataset", "lang_a", "lang_b", "jaccard",
        ],
        "decomposition.csv": [
            "timestamp_utc", "subject_model", "dataset", "b_intercept", "b_align",
            "b_comp", "n",
        ],
        "metrics_summary.csv": [
            "timestamp_utc", "subject_model", "dataset", "circuit_clti",
            "competence_gated_circuit_clti", "beats_controls_en", "cut_fraction",
        ],
    }

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.results_dir, filename)

    def append(self, filename: str, rows: List[Dict]) -> None:
        if not rows:
            return
        fields = self.FIELDS[filename]
        path = self._path(filename)
        new_file = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            for r in rows:
                r.setdefault("timestamp_utc", _utc_now())
                writer.writerow(r)

    def save_json(self, name: str, obj) -> None:
        with open(self._path(name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        logger.info("[OK] Wrote %s", self._path(name))

    def completed_cells(self, filename: str, key_cols: List[str]) -> set:
        path = self._path(filename)
        done = set()
        if not os.path.exists(path):
            return done
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                done.add(tuple(r.get(k, "") for k in key_cols))
        return done
