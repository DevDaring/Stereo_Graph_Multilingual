"""STEP 02 (CPU) - concept-level train/val/test split (the leakage fix).

Forms canonical cross-lingual concepts by union-find over the same_as group pairs
and assigns each concept deterministically to train / val / test. The split is the
foundation of every later leakage-free step. Writes results/concept_split.json.

Usage:  python step_02_split.py
"""
import os

from lib.paths import cfg
from lib.splits import build_split
from Common_00.common import resolve, write_json


def main():
    config = cfg()
    split = build_split(config)
    out = resolve(config["paths"]["split_file"])
    write_json(out, split)
    print("=== STEP 02  concept split ===")
    print(f"  unit={split['unit']} seed={split['seed']} "
          f"test={split['test_fraction']} val={split['val_fraction']}")
    print(f"  concepts: {split['n_concepts']}  counts: {split['counts']}")
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
