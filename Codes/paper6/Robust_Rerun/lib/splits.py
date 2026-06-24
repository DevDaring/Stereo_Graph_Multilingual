"""Concept-level train / val / test split (the core leakage fix).

A "concept" is a canonical cross-lingual social group, e.g. the Brahmin caste
across English, Hindi, and Bengali. Concepts are formed by union-find over the
`same_as` group pairs that the parallel benchmarks induce. Each concept is then
assigned deterministically (stable hash of its id + a seed) to train, val, or
test. The MS-SKG is later built from TRAIN concepts only; retrieval is evaluated
on TEST concepts; calibration uses VAL concepts. Because test/val concepts never
enter the graph, the graph can never hand back a query item's own answer.
"""
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple

from Common_00.dataio import group_by_index, read_bias_rows


def _norm(s: str) -> str:
    return " ".join(str(s).split()).strip().lower()


def group_id(lang: str, surface: str) -> str:
    return f"group::{lang}::{_norm(surface)}"


class _UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_concepts(config: Dict) -> Dict[str, str]:
    """Map every group node id -> its canonical concept id (cross-lingual)."""
    uf = _UnionFind()
    for dataset in ("crows_pairs", "indian_bias"):
        rows = read_bias_rows(config, dataset)
        langs = config["data"][dataset]["languages"]
        for (_idx, _btype), per_lang in group_by_index(rows).items():
            present = [l for l in langs if l in per_lang]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    la, lb = present[i], present[j]
                    ra, rb = per_lang[la], per_lang[lb]
                    if ra["group_stereo"] and rb["group_stereo"]:
                        uf.union(group_id(la, ra["group_stereo"]),
                                 group_id(lb, rb["group_stereo"]))
                    if ra["group_anti"] and rb["group_anti"]:
                        uf.union(group_id(la, ra["group_anti"]),
                                 group_id(lb, rb["group_anti"]))
            # ensure singletons exist too
            for l in present:
                r = per_lang[l]
                for surf in (r["group_stereo"], r["group_anti"]):
                    if surf:
                        uf.find(group_id(l, surf))
    return {g: uf.find(g) for g in uf.parent}


def _bucket(concept_id: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{concept_id}".encode("utf-8")).hexdigest()
    return int(h, 16) / float(1 << 256)


def assign_splits(concepts: Dict[str, str], test_fraction: float,
                  val_fraction: float, seed: int) -> Dict[str, str]:
    """concept_id -> 'train' | 'val' | 'test', deterministic and reproducible."""
    canon = sorted(set(concepts.values()))
    out: Dict[str, str] = {}
    for cid in canon:
        u = _bucket(cid, seed)
        if u < test_fraction:
            out[cid] = "test"
        elif u < test_fraction + val_fraction:
            out[cid] = "val"
        else:
            out[cid] = "train"
    return out


def build_split(config: Dict) -> Dict:
    sp = config["split"]
    group_to_concept = build_concepts(config)
    concept_split = assign_splits(group_to_concept, sp["test_fraction"],
                                  sp["val_fraction"], sp["seed"])
    counts: Dict[str, int] = defaultdict(int)
    for s in concept_split.values():
        counts[s] += 1
    return {
        "unit": "concept",
        "seed": sp["seed"],
        "test_fraction": sp["test_fraction"],
        "val_fraction": sp["val_fraction"],
        "n_concepts": len(concept_split),
        "counts": dict(counts),
        "group_to_concept": group_to_concept,
        "concept_split": concept_split,
    }


def item_split(split: Dict, lang_stereo_surface: str, lang_anti_surface: str,
               lang: str) -> str:
    """Split label for one item, using BOTH its group surfaces. An item counts as
    train only when both of its groups are train concepts; if either group is
    test it is a test item; otherwise val. This keeps every test/val group OUT of
    the train graph."""
    g2c = split["group_to_concept"]
    cs = split["concept_split"]
    labels = []
    for surf in (lang_stereo_surface, lang_anti_surface):
        if not surf:
            continue
        gid = group_id(lang, surf)
        cid = g2c.get(gid)
        labels.append(cs.get(cid, "train") if cid else "train")
    if "test" in labels:
        return "test"
    if "val" in labels:
        return "val"
    return "train"


def concept_of(split: Dict, lang: str, surface: str) -> str:
    return split["group_to_concept"].get(group_id(lang, surface), "")
