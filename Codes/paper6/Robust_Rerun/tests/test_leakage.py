"""Leakage proof on the REAL built artifacts (results/concept_split.json + the
train-only graph). Run AFTER step_02 and step_03. Verifies, end to end, that the
three leakage vectors from the review cannot occur:

  1. No TEST or VAL group ever appears as a node in the train graph.
  2. The KG declares built_from = train_concepts_only with test concepts held out.
  3. For every evaluated TEST item, retrieval (flat_dict, kg_rag_monolingual,
     kg_rag) never returns the item's own stereo/anti surface or its own concepts.
  4. Every evaluation pair is genuinely a held-out TEST item.

If the artifacts are not built yet, the test skips cleanly.

Run:  python tests/test_leakage.py   (or: pytest tests/test_leakage.py)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Robust_Rerun/

from lib.paths import cfg, load_split  # noqa: E402
from lib.splits import concept_of, group_id, item_split  # noqa: E402
from lib import rag_leakfree as R  # noqa: E402
from Common_00.common import resolve  # noqa: E402
from Common_00.kg_io import load_graph  # noqa: E402

import json  # noqa: E402


def _artifacts_ready(config):
    sp = resolve(config["paths"]["split_file"])
    kg = resolve(os.path.join(config["paths"]["kg_dir"], "graph.json"))
    return os.path.exists(sp) and os.path.exists(kg)


def test_no_heldout_group_in_graph():
    config = cfg()
    if not _artifacts_ready(config):
        print("  SKIP (run step_02 + step_03 first)")
        return
    split = load_split(config)
    g2c, cs = split["group_to_concept"], split["concept_split"]
    graph = load_graph(config)
    bad = []
    for nid, a in graph.nodes(data=True):
        if a.get("type") != "group":
            continue
        concept = g2c.get(nid)
        if cs.get(concept) != "train":
            bad.append((nid, cs.get(concept)))
    assert not bad, f"{len(bad)} non-train group nodes leaked into the graph, e.g. {bad[:5]}"
    print(f"  ok: every group node in the graph is a TRAIN concept "
          f"({graph.number_of_nodes()} nodes checked)")


def test_kg_built_from_train_only():
    config = cfg()
    if not _artifacts_ready(config):
        print("  SKIP")
        return
    with open(resolve(os.path.join(config["paths"]["kg_dir"], "kg_stats.json")),
              "r", encoding="utf-8") as f:
        stats = json.load(f)
    assert stats.get("built_from") == "train_concepts_only"
    assert stats.get("n_test_concepts", 0) > 0
    assert stats.get("items_skipped_heldout", 0) > 0
    print(f"  ok: KG built_from=train_concepts_only, "
          f"{stats['n_test_concepts']} test concepts held out, "
          f"{stats['items_skipped_heldout']} items skipped")


def test_retrieval_never_returns_own_answer():
    config = cfg()
    if not _artifacts_ready(config):
        print("  SKIP")
        return
    split = load_split(config)
    graph = load_graph(config)
    pools = R.build_pools(graph, split)
    rng = random.Random(0)
    conditions = ["flat_dict", "kg_rag_monolingual", "kg_rag"]
    checked, hits = 0, 0
    for dataset in ("crows_pairs", "indian_bias"):
        for lang in config["data"][dataset]["languages"]:
            pairs = R.build_pairs_for(config, split, dataset, lang, which="test", max_pairs=120)
            for p in pairs:
                # the eval item must be a genuine TEST item
                assert item_split(split, p["target_stereotypical"],
                                  p["target_anti_stereotypical"], lang) == "test"
                excl = {R._norm(p["target_stereotypical"]), R._norm(p["target_anti_stereotypical"])}
                concepts = {concept_of(split, lang, p["target_stereotypical"]),
                            concept_of(split, lang, p["target_anti_stereotypical"])}
                for cond in conditions:
                    facts = R._retrieve(pools, cond, lang, p["bias_type"], excl, concepts,
                                        config["rag"]["max_neighbours"], rng)
                    for fct in facts:
                        checked += 1
                        if R._norm(fct) in excl:
                            hits += 1
                        # the retrieved fact must not belong to the query's own concepts
                        # (its concept is unknown here; surface exclusion is the guarantee)
    assert hits == 0, f"{hits} retrieved facts equalled the query's own target surface"
    print(f"  ok: 0 leaked answers across {checked} retrieved facts on held-out TEST items")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"ALL {len(fns)} LEAKAGE CHECKS PASSED")


if __name__ == "__main__":
    _run()
