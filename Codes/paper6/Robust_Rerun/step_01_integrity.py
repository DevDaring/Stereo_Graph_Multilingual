"""STEP 01 (CPU) - data integrity. Runs on EVERY (re)run.

Checks both bias datasets for duplicate rows, duplicate (index, language,
bias_type) keys, empty/corrupted fields, missing mask slots, and broken
cross-lingual parallelism. Writes results/integrity_report.json and exits
non-zero on any hard corruption, so no later step ever runs on bad data.

Usage:  python step_01_integrity.py
"""
import os
import sys

from lib.paths import cfg
from lib.integrity import check_all
from Common_00.common import resolve, write_json


def main():
    config = cfg()
    rep = check_all(config)
    out = resolve(os.path.join(config["paths"]["results"], "integrity_report.json"))
    write_json(out, rep)
    print("=== STEP 01  data integrity ===")
    for r in rep["datasets"]:
        print(f"  {r['dataset']:14s} rows={r['n_rows']:5d} units={r['n_alignment_units']:5d} "
              f"dup_rows={r['duplicate_rows']} dup_keys={r['duplicate_unique_keys']} "
              f"missing_parallel={r['units_missing_a_language']} -> {r['status']}")
    print(f"  report: {out}")
    if rep["total_hard_corruption"] > 0:
        print(f"  HARD CORRUPTION found ({rep['total_hard_corruption']}). Fix data before proceeding.")
        sys.exit(2)
    print("  clean.")


if __name__ == "__main__":
    main()
