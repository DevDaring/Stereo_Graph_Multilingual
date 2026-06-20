"""STEP 01  (Dataset_Prep_01, CPU) - Data integrity check.

Runs on every (re)run. For each bias dataset it checks for duplicate rows,
duplicate (index, language, bias_type) keys, corrupted/empty fields, missing mask
slots, and cross-lingual parallelism (every (Index, bias_type) unit should exist in
en/hi/bn). The unique unit is (index, language, bias_type): indian_bias stores
several bias_type rows under one Index, which is valid, not a duplicate. Writes
results/data_integrity_report.json and prints a summary. Non-zero exit if any hard
corruption is found, so the pipeline can stop early.

Usage:  python Dataset_Prep_01/check_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from collections import Counter, defaultdict

from Common_00.common import file_sha256, load_config, write_json, resolve
from Common_00.dataio import dataset_path, group_by_index, read_bias_rows


def check_dataset(config, dataset_key):
    expected_langs = set(config["data"][dataset_key]["languages"])
    rows = read_bias_rows(config, dataset_key)
    path = dataset_path(config, dataset_key)

    seen, dup_rows = set(), 0
    key_counts = Counter()
    corrupt = defaultdict(int)
    for r in rows:
        sig = (r["index"], r["lang"], r["bias_type"], r["sentence"],
               r["group_stereo"], r["group_anti"])
        if sig in seen:
            dup_rows += 1
        seen.add(sig)
        key_counts[(r["index"], r["lang"], r["bias_type"])] += 1
        if not r["index"]:
            corrupt["missing_index"] += 1
        if r["lang"] not in expected_langs:
            corrupt["unexpected_language"] += 1
        if not r["group_stereo"] or not r["group_anti"]:
            corrupt["empty_target"] += 1
        if not r["sentence"].strip():
            corrupt["empty_sentence"] += 1
        elif "MASK" not in r["sentence"].upper():
            corrupt["missing_mask_slot"] += 1

    dup_keys = {f"{i}|{l}|{b}": c for (i, l, b), c in key_counts.items() if c > 1}

    by_idx = group_by_index(rows)  # keyed on (index, bias_type)
    missing_parallel = [f"{idx}|{btype}" for (idx, btype), d in by_idx.items()
                        if not expected_langs.issubset(set(d.keys()))]

    report = {
        "dataset": dataset_key,
        "file": os.path.basename(path),
        "sha256": file_sha256(path),
        "n_rows": len(rows),
        "n_alignment_units": len(by_idx),
        "duplicate_rows": dup_rows,
        "duplicate_unique_keys": len(dup_keys),
        "duplicate_key_examples": dict(list(dup_keys.items())[:10]),
        "corruption": dict(corrupt),
        "units_missing_a_language": len(missing_parallel),
        "missing_parallel_examples": missing_parallel[:10],
    }
    # hard corruption = anything that would break downstream parsing
    hard = (dup_rows + len(dup_keys) + corrupt["missing_index"]
            + corrupt["empty_target"] + corrupt["empty_sentence"]
            + corrupt["unexpected_language"])
    report["hard_corruption_count"] = hard
    report["status"] = "ok" if hard == 0 else "issues_found"
    return report


def main():
    config = load_config()
    reports = [check_dataset(config, k) for k in ("crows_pairs", "indian_bias")]
    out = resolve(os.path.join(config["paths"]["results"], "data_integrity_report.json"))
    write_json(out, {"datasets": reports})

    print("=== STEP 01  data integrity ===")
    issues = 0
    for r in reports:
        print(f"  {r['dataset']:14s} rows={r['n_rows']:5d} units={r['n_alignment_units']:5d} "
              f"dup_rows={r['duplicate_rows']} dup_keys={r['duplicate_unique_keys']} "
              f"missing_parallel={r['units_missing_a_language']} -> {r['status']}")
        issues += r["hard_corruption_count"]
    print(f"  report: {out}")
    if issues:
        print(f"  HARD CORRUPTION found ({issues}). Fix the data before proceeding.")
        sys.exit(2)
    print("  clean.")


if __name__ == "__main__":
    main()
