"""Graph algorithms over the MS-SKG (CPU).
Implements: sibling lookup for counterfactual swaps (E2), counter-stereotype
retrieval (E4), cross-lingual concept bridging via same_as (E3), and personalised
PageRank / label propagation for the bias-propagation analysis (E5).
"""
from collections import defaultdict
from typing import Dict, List, Optional


def group_siblings(graph, node_id: str) -> List[str]:
    """Other group nodes with the same bias_type and language - valid swap targets
    for KG-guided counterfactual augmentation (E2)."""
    if node_id not in graph:
        return []
    nd = graph.nodes[node_id]
    if nd.get("type") != "group":
        return []
    out = []
    for nid, attrs in graph.nodes(data=True):
        if (nid != node_id and attrs.get("type") == "group"
                and attrs.get("bias_type") == nd.get("bias_type")
                and attrs.get("lang") == nd.get("lang")):
            out.append(nid)
    return out


def counter_stereotype_groups(graph, context_id: str) -> List[str]:
    """Groups reached from a context by anti_stereotype_of - the counter-stereotype
    evidence to inject in KG-RAG (E4)."""
    out = []
    if context_id not in graph:
        return out
    for _, dst, data in graph.out_edges(context_id, data=True):
        if data.get("relation") == "anti_stereotype_of":
            out.append(dst)
    return out


def bridge_to_language(graph, node_id: str, target_lang: str) -> Optional[str]:
    """Follow same_as to the equivalent node in `target_lang` (E3 concept bridge)."""
    if node_id not in graph:
        return None
    canon = graph.nodes[node_id].get("canonical_id")
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("canonical_id") == canon and attrs.get("lang") == target_lang:
            return nid
    return None


def surface(graph, node_id: str) -> str:
    return graph.nodes[node_id].get("surface", "") if node_id in graph else ""


def personalised_pagerank(graph, seeds: Dict[str, float], alpha: float = 0.85,
                          max_iter: int = 100) -> Dict[str, float]:
    """Personalised PageRank with a seed mass distribution (E5). Uses an undirected
    weighted view so stereotype mass spreads to related groups."""
    import networkx as nx
    if graph.number_of_nodes() == 0:
        return {}
    ug = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        if ug.has_edge(u, v):
            ug[u][v]["weight"] += w
        else:
            ug.add_edge(u, v, weight=w)
    ug.add_nodes_from(graph.nodes())
    if not seeds:
        return nx.pagerank(ug, alpha=alpha, max_iter=max_iter, weight="weight")
    s = sum(seeds.values()) or 1.0
    pers = {n: (seeds.get(n, 0.0) / s) for n in ug.nodes()}
    try:
        return nx.pagerank(ug, alpha=alpha, personalization=pers,
                           max_iter=max_iter, weight="weight")
    except Exception:
        return nx.pagerank(ug, alpha=alpha, max_iter=max_iter, weight="weight")


def communities_by_bias_type(graph) -> Dict[str, List[str]]:
    """Cluster group nodes by bias_type (a cheap, interpretable community proxy)."""
    out = defaultdict(list)
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("type") == "group":
            out[attrs.get("bias_type", "unknown")].append(nid)
    return dict(out)
