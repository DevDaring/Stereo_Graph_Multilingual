"""STEP 03 (CPU) - build the MS-SKG from TRAIN concepts only.

Test and val concepts are never added, so later retrieval on held-out concepts
cannot return a query item's own answer. Writes results/kg/{nodes.csv, edges.csv,
graph.json, kg_stats.json}.

Usage:  python step_03_build_kg.py
"""
from lib.paths import cfg, load_split
from lib.kg_build import build
from Common_00.kg_io import save_kg


def main():
    config = cfg()
    split = load_split(config)
    nodes, edges, stats = build(config, split)
    save_kg(config, nodes, edges, stats)
    print("=== STEP 03  build MS-SKG (train concepts only) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
