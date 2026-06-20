"""STEP 02  (Dataset_Prep_02, CPU) - Build the Multilingual Stereotype Knowledge
Graph (MS-SKG).  Implements a stereotype KG from parallel diagnostic data.

Nodes:  group (social group) and context (masked stereotype sentence), per language.
Edges:  stereotype_of / anti_stereotype_of (context -> group), and same_as
        (cross-lingual), derived for free from Index-aligned rows.
Outputs: results/kg/{nodes.csv, edges.csv, graph.json, kg_stats.json}.

Usage:  python Dataset_Prep_02/build_kg.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from collections import defaultdict

from Common_00.common import load_config
from Common_00.dataio import group_by_index, read_bias_rows
from Common_00.kg_io import save_kg


def _norm(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def build(config):
    nodes = {}        # id -> node dict
    edges = []        # list of edge dicts
    same_as_pairs = []  # (id_a, id_b) undirected, for canonicalisation
    covered, total = 0, 0

    def group_id(lang, surface):
        return f"group::{lang}::{_norm(surface)}"

    def ctx_id(dataset, idx, lang, bias_type):
        return f"context::{dataset}::{idx}::{lang}::{bias_type}"

    def add_group(lang, surface, bias_type):
        gid = group_id(lang, surface)
        if gid not in nodes:
            nodes[gid] = {"id": gid, "lang": lang, "surface": surface,
                          "type": "group", "canonical_id": gid, "bias_type": bias_type}
        return gid

    def add_context(dataset, idx, lang, sentence, bias_type):
        cid = ctx_id(dataset, idx, lang, bias_type)
        if cid not in nodes:
            nodes[cid] = {"id": cid, "lang": lang, "surface": sentence,
                          "type": "context", "canonical_id": cid, "bias_type": bias_type}
        return cid

    for dataset in ("crows_pairs", "indian_bias"):
        rows = read_bias_rows(config, dataset)
        by_idx = group_by_index(rows)
        langs = config["data"][dataset]["languages"]
        for (idx, btype), per_lang in by_idx.items():
            present = [l for l in langs if l in per_lang]
            for l in present:
                r = per_lang[l]
                total += 1
                if not r["group_stereo"] or not r["group_anti"]:
                    continue
                covered += 1
                gs = add_group(l, r["group_stereo"], r["bias_type"])
                ga = add_group(l, r["group_anti"], r["bias_type"])
                cx = add_context(dataset, idx, l, r["sentence"], r["bias_type"])
                edges.append({"src": cx, "dst": gs, "relation": "stereotype_of", "weight": 1})
                edges.append({"src": cx, "dst": ga, "relation": "anti_stereotype_of", "weight": 1})
            # cross-lingual same_as from index alignment (the free bridge)
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    la, lb = present[i], present[j]
                    ra, rb = per_lang[la], per_lang[lb]
                    if ra["group_stereo"] and rb["group_stereo"]:
                        same_as_pairs.append((group_id(la, ra["group_stereo"]),
                                              group_id(lb, rb["group_stereo"])))
                    if ra["group_anti"] and rb["group_anti"]:
                        same_as_pairs.append((group_id(la, ra["group_anti"]),
                                              group_id(lb, rb["group_anti"])))
                    same_as_pairs.append((ctx_id(dataset, idx, la, btype),
                                          ctx_id(dataset, idx, lb, btype)))

    # canonicalise via connected components of same_as
    import networkx as nx
    ug = nx.Graph()
    ug.add_nodes_from(nodes.keys())
    ug.add_edges_from([(a, b) for a, b in same_as_pairs if a in nodes and b in nodes])
    for comp in nx.connected_components(ug):
        canon = sorted(comp)[0]
        for nid in comp:
            nodes[nid]["canonical_id"] = canon
    # emit same_as edges (dedup)
    seen = set()
    for a, b in same_as_pairs:
        if a in nodes and b in nodes and (a, b) not in seen and (b, a) not in seen:
            edges.append({"src": a, "dst": b, "relation": "same_as", "weight": 1})
            seen.add((a, b))

    # weight aggregation for repeated stereotype_of/anti edges
    agg = defaultdict(float)
    keep = []
    for e in edges:
        if e["relation"] in ("stereotype_of", "anti_stereotype_of"):
            agg[(e["src"], e["dst"], e["relation"])] += e["weight"]
        else:
            keep.append(e)
    for (s, d, rel), w in agg.items():
        keep.append({"src": s, "dst": d, "relation": rel, "weight": w})

    node_list = list(nodes.values())
    n_group = sum(1 for n in node_list if n["type"] == "group")
    n_ctx = sum(1 for n in node_list if n["type"] == "context")
    n_same = sum(1 for e in keep if e["relation"] == "same_as")
    n_canon = len({n["canonical_id"] for n in node_list})
    stats = {
        "n_nodes": len(node_list), "n_group_nodes": n_group, "n_context_nodes": n_ctx,
        "n_edges": len(keep), "n_same_as_edges": n_same, "n_canonical_concepts": n_canon,
        "coverage": round(covered / total, 4) if total else 0.0,
        "coverage_target": config["kg"]["coverage_target"],
        "use_conceptnet": config["kg"]["use_conceptnet"],
    }
    return node_list, keep, stats


def main():
    config = load_config()
    nodes, edges, stats = build(config)
    save_kg(config, nodes, edges, stats)
    print("=== STEP 02  build MS-SKG ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["coverage"] < stats["coverage_target"]:
        print(f"  WARNING: coverage {stats['coverage']} < target {stats['coverage_target']}")
    print(f"  written: {os.path.join('results', 'kg')}/")


if __name__ == "__main__":
    main()
