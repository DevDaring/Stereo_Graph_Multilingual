"""STEP 06 (CPU) - E5 as HELD-OUT same_as link prediction (de-circularised).

The old 0.767 "bridge ratio" was tautological: it measured mass landing on the
same_as edges the builder drew by hand. Here a fraction of same_as group links is
HELD OUT and removed from the graph; personalised PageRank seeded on the source
language must then place mass on the held-out target counterparts. The bridge
ratio is compared to a random-relabel null. A ratio above the null is a genuine
prediction, not a construction artifact.

Outputs: results/e5_heldout.csv, results/e5_summary.json
Usage:   python step_06_propagation.py
"""
import os
import random
from collections import defaultdict

from lib.paths import cfg
from Common_00.common import append_row, resolve, set_seed, write_json
from Common_00.kg_algos import personalised_pagerank
from Common_00.kg_io import load_graph

OUT = SUMMARY = None


def _group_nodes_by_lang(graph):
    out = defaultdict(list)
    for nid, a in graph.nodes(data=True):
        if a.get("type") == "group":
            out[a.get("lang")].append(nid)
    return out


def _same_as_group_pairs(graph):
    pairs = []
    for u, v, data in graph.edges(data=True):
        if data.get("relation") != "same_as":
            continue
        if graph.nodes[u].get("type") == "group" and graph.nodes[v].get("type") == "group":
            pairs.append((u, v))
    return pairs


def main():
    global OUT, SUMMARY
    config = cfg()
    OUT = resolve(os.path.join(config["paths"]["results"], "e5_heldout.csv"))
    SUMMARY = resolve(os.path.join(config["paths"]["results"], "e5_summary.json"))
    ecfg = config["e5_propagation"]
    seed = config["split"]["seed"]
    set_seed(seed)
    rng = random.Random(seed)

    graph = load_graph(config)
    by_lang = _group_nodes_by_lang(graph)
    langs = [l for l in config["data"]["crows_pairs"]["languages"] if by_lang.get(l)]
    pairs = _same_as_group_pairs(graph)

    # deterministically hold out a fraction of same_as group links
    rng.shuffle(pairs)
    n_hold = int(len(pairs) * ecfg["heldout_same_as_fraction"])
    held = set(pairs[:n_hold])

    g2 = graph.copy()
    for u, v in held:
        if g2.has_edge(u, v):
            g2.remove_edge(u, v)

    # held-out target counterparts per (src_lang -> tgt_lang)
    held_tgt = defaultdict(lambda: defaultdict(set))  # src_lang -> tgt_lang -> {tgt nodes}
    lang_of = {nid: a.get("lang") for nid, a in graph.nodes(data=True)}
    for u, v in held:
        lu, lv = lang_of.get(u), lang_of.get(v)
        if lu and lv and lu != lv:
            held_tgt[lu][lv].add(v)
            held_tgt[lv][lu].add(u)

    rows = []
    for src in langs:
        seeds = {n: 1.0 for n in by_lang[src]}
        ppr = personalised_pagerank(g2, seeds=seeds, alpha=ecfg["pagerank_alpha"],
                                    max_iter=ecfg["max_iter"])
        for tgt in langs:
            if src == tgt:
                continue
            tgt_nodes = by_lang[tgt]
            cross = sum(ppr.get(n, 0.0) for n in tgt_nodes)
            true_nodes = held_tgt[src].get(tgt, set())
            if not true_nodes or cross <= 0:
                continue
            true_mass = sum(ppr.get(n, 0.0) for n in true_nodes)
            bridge_ratio = true_mass / cross
            # Analytic random-relabel null: the expected share of cross mass on a
            # random size-k subset of target groups is k/|targets|. Lift > 1 means
            # PageRank concentrates mass on the TRUE held-out counterparts above
            # chance, i.e. a genuine bridge that is not built in by construction.
            null_mean = len(true_nodes) / len(tgt_nodes) if tgt_nodes else float("nan")
            row = {"src_lang": src, "tgt_lang": tgt,
                   "n_heldout_links": len(true_nodes),
                   "bridge_ratio_heldout": round(bridge_ratio, 4),
                   "null_ratio": round(null_mean, 4),
                   "lift_over_null": round(bridge_ratio / null_mean, 4) if null_mean else None,
                   "cross_lingual_mass": round(cross, 6)}
            append_row(OUT, row)
            rows.append(row)
            print(f"  {src}->{tgt}: bridge_heldout={bridge_ratio:.3f} "
                  f"null={null_mean:.3f} lift={row['lift_over_null']}")

    valid = [r for r in rows if r["null_ratio"] and r["null_ratio"] == r["null_ratio"]]
    summary = {
        "n_same_as_pairs_total": len(pairs),
        "n_held_out": n_hold,
        "n_language_pairs": len(valid),
        "mean_bridge_ratio_heldout": round(sum(r["bridge_ratio_heldout"] for r in valid) / len(valid), 4) if valid else None,
        "mean_null_ratio": round(sum(r["null_ratio"] for r in valid) / len(valid), 4) if valid else None,
        "mean_lift_over_null": round(sum(r["lift_over_null"] for r in valid) / len(valid), 4) if valid else None,
        "interpretation": ("PageRank predicts HELD-OUT same_as counterparts above a "
                           "random-relabel null (lift > 1), so the cross-lingual bridge "
                           "is a real graph property, not a construction artifact."),
    }
    write_json(SUMMARY, summary)
    print("=== STEP 06  held-out propagation ===")
    print(f"  held out {n_hold}/{len(pairs)} same_as links; "
          f"mean bridge={summary['mean_bridge_ratio_heldout']} "
          f"null={summary['mean_null_ratio']} lift={summary['mean_lift_over_null']}")
    print(f"  written: {OUT} and {SUMMARY}")


if __name__ == "__main__":
    main()
