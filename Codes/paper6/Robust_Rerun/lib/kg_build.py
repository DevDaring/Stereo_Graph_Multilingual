"""Build the MS-SKG from TRAIN concepts only (leakage-free).

Identical node/edge schema to the original Paper 6 graph, but an item is added
only when BOTH of its group surfaces belong to TRAIN concepts (per lib.splits).
Test and val groups never appear, so retrieval on held-out concepts cannot
surface a query item's own targets. Saves to results/kg via Common_00.kg_io.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

from Common_00.dataio import group_by_index, read_bias_rows
from lib.splits import item_split


def _norm(s: str) -> str:
    return " ".join(str(s).split()).strip().lower()


def build(config: Dict, split: Dict) -> Tuple[List[Dict], List[Dict], Dict]:
    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []
    same_as_pairs: List[Tuple[str, str]] = []
    kept_items, skipped_items = 0, 0

    def group_id(lang, surface):
        return f"group::{lang}::{_norm(surface)}"

    def ctx_id(dataset, idx, lang, bias_type):
        return f"context::{dataset}::{idx}::{lang}::{bias_type}"

    def add_group(lang, surface, bias_type):
        gid = group_id(lang, surface)
        nodes.setdefault(gid, {"id": gid, "lang": lang, "surface": surface,
                               "type": "group", "canonical_id": gid, "bias_type": bias_type})
        return gid

    def add_context(dataset, idx, lang, sentence, bias_type):
        cid = ctx_id(dataset, idx, lang, bias_type)
        nodes.setdefault(cid, {"id": cid, "lang": lang, "surface": sentence,
                               "type": "context", "canonical_id": cid, "bias_type": bias_type})
        return cid

    for dataset in ("crows_pairs", "indian_bias"):
        rows = read_bias_rows(config, dataset)
        langs = config["data"][dataset]["languages"]
        for (idx, btype), per_lang in group_by_index(rows).items():
            present = [l for l in langs if l in per_lang]
            train_langs = []
            for l in present:
                r = per_lang[l]
                if not r["group_stereo"] or not r["group_anti"]:
                    continue
                if item_split(split, r["group_stereo"], r["group_anti"], l) != "train":
                    skipped_items += 1
                    continue
                kept_items += 1
                train_langs.append(l)
                gs = add_group(l, r["group_stereo"], r["bias_type"])
                ga = add_group(l, r["group_anti"], r["bias_type"])
                cx = add_context(dataset, idx, l, r["sentence"], r["bias_type"])
                edges.append({"src": cx, "dst": gs, "relation": "stereotype_of", "weight": 1})
                edges.append({"src": cx, "dst": ga, "relation": "anti_stereotype_of", "weight": 1})
            # same_as only among langs that were both kept in the train graph
            for i in range(len(train_langs)):
                for j in range(i + 1, len(train_langs)):
                    la, lb = train_langs[i], train_langs[j]
                    ra, rb = per_lang[la], per_lang[lb]
                    same_as_pairs.append((group_id(la, ra["group_stereo"]),
                                          group_id(lb, rb["group_stereo"])))
                    same_as_pairs.append((group_id(la, ra["group_anti"]),
                                          group_id(lb, rb["group_anti"])))
                    same_as_pairs.append((ctx_id(dataset, idx, la, btype),
                                          ctx_id(dataset, idx, lb, btype)))

    import networkx as nx
    ug = nx.Graph()
    ug.add_nodes_from(nodes.keys())
    ug.add_edges_from([(a, b) for a, b in same_as_pairs if a in nodes and b in nodes])
    for comp in nx.connected_components(ug):
        canon = sorted(comp)[0]
        for nid in comp:
            nodes[nid]["canonical_id"] = canon

    seen = set()
    for a, b in same_as_pairs:
        if a in nodes and b in nodes and (a, b) not in seen and (b, a) not in seen:
            edges.append({"src": a, "dst": b, "relation": "same_as", "weight": 1})
            seen.add((a, b))

    agg = defaultdict(float)
    keep: List[Dict] = []
    for e in edges:
        if e["relation"] in ("stereotype_of", "anti_stereotype_of"):
            agg[(e["src"], e["dst"], e["relation"])] += e["weight"]
        else:
            keep.append(e)
    for (s, d, rel), w in agg.items():
        keep.append({"src": s, "dst": d, "relation": rel, "weight": w})

    node_list = list(nodes.values())
    cs = split["concept_split"]
    stats = {
        "built_from": "train_concepts_only",
        "n_nodes": len(node_list),
        "n_group_nodes": sum(1 for n in node_list if n["type"] == "group"),
        "n_context_nodes": sum(1 for n in node_list if n["type"] == "context"),
        "n_edges": len(keep),
        "n_same_as_edges": sum(1 for e in keep if e["relation"] == "same_as"),
        "n_canonical_concepts": len({n["canonical_id"] for n in node_list}),
        "n_train_concepts": sum(1 for v in cs.values() if v == "train"),
        "n_val_concepts": sum(1 for v in cs.values() if v == "val"),
        "n_test_concepts": sum(1 for v in cs.values() if v == "test"),
        "items_kept_train": kept_items,
        "items_skipped_heldout": skipped_items,
    }
    return node_list, keep, stats
