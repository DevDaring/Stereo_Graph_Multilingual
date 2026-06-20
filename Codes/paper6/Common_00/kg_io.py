"""Save / load the Multilingual Stereotype Knowledge Graph (MS-SKG).

Schema:
  nodes.csv : id, lang, surface, type {group|attribute}, canonical_id, bias_type
  edges.csv : src, dst, relation {stereotype_of|anti_stereotype_of|same_as|related_to}, weight
  graph.json: NetworkX node-link dump (the same graph, for algorithms)
"""
import csv
import os
from typing import Dict, List

from Common_00.common import resolve, write_json

NODE_COLS = ["id", "lang", "surface", "type", "canonical_id", "bias_type"]
EDGE_COLS = ["src", "dst", "relation", "weight"]


def kg_dir(config: Dict) -> str:
    d = resolve(config["paths"]["kg_dir"])
    os.makedirs(d, exist_ok=True)
    return d


def save_kg(config: Dict, nodes: List[Dict], edges: List[Dict], stats: Dict) -> None:
    d = kg_dir(config)
    with open(os.path.join(d, "nodes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NODE_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(nodes)
    with open(os.path.join(d, "edges.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(edges)
    _save_graph_json(d, nodes, edges)
    write_json(os.path.join(d, "kg_stats.json"), stats)


def _save_graph_json(d: str, nodes: List[Dict], edges: List[Dict]) -> None:
    import networkx as nx
    g = nx.MultiDiGraph()
    for n in nodes:
        g.add_node(n["id"], **{k: n.get(k) for k in NODE_COLS if k != "id"})
    for e in edges:
        g.add_edge(e["src"], e["dst"], relation=e["relation"], weight=float(e["weight"]))
    from networkx.readwrite import json_graph
    write_json(os.path.join(d, "graph.json"), json_graph.node_link_data(g, edges="links"))


def load_graph(config: Dict):
    """Return the MS-SKG as a NetworkX MultiDiGraph."""
    import json
    from networkx.readwrite import json_graph
    path = os.path.join(kg_dir(config), "graph.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"KG not built yet: {path}. Run 02_build_kg.py first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_graph.node_link_graph(data, directed=True, multigraph=True, edges="links")


def load_nodes(config: Dict) -> List[Dict]:
    path = os.path.join(kg_dir(config), "nodes.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
