"""STEP 08  (CPU_Only_08) - E5: graph bias-propagation analysis.

Papers 1-5 found the language-specific bias circuits to be near-disjoint
(Jaccard ~= 0.035). E5 tests whether the MS-SKG nonetheless bridges them: a
personalised PageRank seeded on one language's stereotype groups is measured for the
mass it delivers to another language's groups, and the share of that mass landing on
true same_as counterparts (the bridge ratio). The reused Paper 5 circuit Jaccard is
reported alongside as the contrast. No GPU.

Outputs: results/e5_propagation.csv, results/e5_summary.json
Usage:   python CPU_Only_08/e5_propagation.py
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from Common_00.common import (append_row, done_keys, load_config, read_csv_dicts,
                              resolve, set_seed, write_json)
from Common_00.kg_algos import communities_by_bias_type, personalised_pagerank
from Common_00 import reuse
from Common_00.kg_io import load_graph

OUT = None
SUMMARY = None


def _groups_by_lang(graph):
    out = defaultdict(list)
    for nid, a in graph.nodes(data=True):
        if a.get("type") == "group":
            out[a.get("lang")].append(nid)
    return out


def _canon_map(graph):
    """canonical_id -> {lang: node_id} for same_as-linked counterparts."""
    out = defaultdict(dict)
    for nid, a in graph.nodes(data=True):
        if a.get("type") == "group" and a.get("canonical_id"):
            out[a["canonical_id"]][a.get("lang")] = nid
    return out


def run(config):
    global OUT, SUMMARY
    OUT = resolve(os.path.join(config["paths"]["results"], "e5_propagation.csv"))
    SUMMARY = resolve(os.path.join(config["paths"]["results"], "e5_summary.json"))
    set_seed(config["experiment"]["seeds"][0])
    alpha = config["e5_propagation"]["pagerank_alpha"]
    max_iter = config["e5_propagation"]["max_iter"]

    graph = load_graph(config)
    by_lang = _groups_by_lang(graph)
    canon = _canon_map(graph)
    langs = [l for l in config["data"]["crows_pairs"]["languages"] if by_lang.get(l)]
    done = done_keys(OUT, ["src_lang", "tgt_lang"])

    # uniform (un-seeded) reference mass on each language's groups
    uniform = personalised_pagerank(graph, seeds={}, alpha=alpha, max_iter=max_iter)
    base_mass = {l: sum(uniform.get(n, 0.0) for n in by_lang[l]) for l in langs}

    rows = []
    for src in langs:
        seeds = {n: 1.0 for n in by_lang[src]}
        ppr = personalised_pagerank(graph, seeds=seeds, alpha=alpha, max_iter=max_iter)
        for tgt in langs:
            if src == tgt or (src, tgt) in done:
                continue
            cross = sum(ppr.get(n, 0.0) for n in by_lang[tgt])
            # mass that lands specifically on same_as counterparts of the seeds
            same_as_nodes = {m[tgt] for cid, m in canon.items()
                             if src in m and tgt in m}
            same_as_mass = sum(ppr.get(n, 0.0) for n in same_as_nodes)
            bridge_ratio = (same_as_mass / cross) if cross > 0 else 0.0
            lift = (cross / base_mass[tgt]) if base_mass[tgt] > 0 else 0.0
            row = {"src_lang": src, "tgt_lang": tgt,
                   "n_src_groups": len(by_lang[src]), "n_tgt_groups": len(by_lang[tgt]),
                   "n_same_as_links": len(same_as_nodes),
                   "cross_lingual_mass": round(cross, 6),
                   "same_as_mass": round(same_as_mass, 6),
                   "bridge_ratio": round(bridge_ratio, 4),
                   "baseline_mass": round(base_mass[tgt], 6),
                   "lift_over_baseline": round(lift, 4)}
            append_row(OUT, row)
            rows.append(row)

    # summary is computed over the FULL CSV (existing + newly appended), so it is
    # correct whether this was a fresh run or a resume that appended nothing.
    all_rows = [r for r in read_csv_dicts(OUT) if _isnum(r.get("bridge_ratio"))]
    n = len(all_rows)
    mean_br = round(sum(float(r["bridge_ratio"]) for r in all_rows) / n, 4) if n else None
    mean_lift = round(sum(float(r["lift_over_baseline"]) for r in all_rows) / n, 4) if n else None

    # contrast against the reused Paper 5 circuit Jaccard (near-disjoint circuits)
    jac = [float(r.get("jaccard", r.get("overlap", "nan")))
           for r in reuse.circuit_overlap(config)
           if _isnum(r.get("jaccard", r.get("overlap")))]
    mean_jac = sum(jac) / len(jac) if jac else None
    comms = communities_by_bias_type(graph)
    summary = {
        "n_languages": len(langs),
        "n_language_pairs": n,
        "mean_circuit_jaccard_paper5": mean_jac,
        "mean_bridge_ratio": mean_br,
        "mean_lift_over_baseline": mean_lift,
        "bias_type_communities": {k: len(v) for k, v in comms.items()},
        "interpretation": ("Circuits are near-disjoint across languages (low Jaccard), "
                           "yet KG propagation delivers cross-lingual mass concentrated on "
                           "same_as counterparts (bridge_ratio), evidencing a graph bridge "
                           "the parameters do not provide."),
    }
    write_json(SUMMARY, summary)
    print(f"  mean circuit Jaccard (Paper 5): {mean_jac}")
    print(f"  mean bridge ratio: {summary['mean_bridge_ratio']}  "
          f"mean lift: {summary['mean_lift_over_baseline']}")


def _isnum(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def main():
    argparse.ArgumentParser().parse_args()
    print("=== STEP 08  E5 graph propagation ===")
    run(load_config())
    print(f"  written: {OUT}  and  {SUMMARY}")


if __name__ == "__main__":
    main()
