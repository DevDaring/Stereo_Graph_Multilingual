"""Data-integrity guard - runs on EVERY (re)run.

For each bias dataset it checks for duplicate rows, duplicate
(index, language, bias_type) keys, corrupted/empty fields, missing mask slots,
and broken cross-lingual parallelism. Returns a per-dataset report and a hard
corruption count; the caller (step_01 / run_all) stops the pipeline if any hard
corruption is found, so no downstream step ever runs on bad data.
"""
from collections import Counter, defaultdict
from typing import Dict, List

from Common_00.common import file_sha256
from Common_00.dataio import dataset_path, group_by_index, read_bias_rows


def check_dataset(config: Dict, dataset_key: str) -> Dict:
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
    by_idx = group_by_index(rows)
    missing_parallel = [f"{idx}|{btype}" for (idx, btype), d in by_idx.items()
                        if not expected_langs.issubset(set(d.keys()))]

    hard = (dup_rows + len(dup_keys) + corrupt["missing_index"]
            + corrupt["empty_target"] + corrupt["empty_sentence"]
            + corrupt["unexpected_language"])
    return {
        "dataset": dataset_key,
        "sha256": file_sha256(path),
        "n_rows": len(rows),
        "n_alignment_units": len(by_idx),
        "duplicate_rows": dup_rows,
        "duplicate_unique_keys": len(dup_keys),
        "duplicate_key_examples": dict(list(dup_keys.items())[:10]),
        "corruption": dict(corrupt),
        "units_missing_a_language": len(missing_parallel),
        "missing_parallel_examples": missing_parallel[:10],
        "hard_corruption_count": hard,
        "status": "ok" if hard == 0 else "issues_found",
    }


def check_all(config: Dict) -> Dict:
    reports: List[Dict] = [check_dataset(config, k) for k in ("crows_pairs", "indian_bias")]
    total_hard = sum(r["hard_corruption_count"] for r in reports)
    return {"datasets": reports, "total_hard_corruption": total_hard,
            "status": "ok" if total_hard == 0 else "issues_found"}
