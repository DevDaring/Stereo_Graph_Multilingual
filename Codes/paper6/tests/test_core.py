"""Core unit tests for Paper 6 - no GPU, no network. Exercises the parsers,
resume-capable CSV I/O, round-robin, KG schema/algorithms, the E4 prompt builder,
and the BTG arithmetic.

Run:  python tests/test_core.py     (or: pytest tests/test_core.py)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from Common_00.common import (RoundRobin, append_row, done_keys, extract_choice,
                             extract_json, read_csv_dicts)
from Common_00 import e4_core, kg_algos
from Common_00.kg_io import EDGE_COLS, NODE_COLS


def _tiny_graph():
    """A 2-language toy MS-SKG: en/bn 'men' groups linked by same_as, plus contexts."""
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_node("group::en::men", lang="en", surface="men", type="group",
               canonical_id="C1", bias_type="gender")
    g.add_node("group::en::women", lang="en", surface="women", type="group",
               canonical_id="C2", bias_type="gender")
    g.add_node("group::bn::purush", lang="bn", surface="purush", type="group",
               canonical_id="C1", bias_type="gender")
    g.add_node("context::crows_pairs::1::en::gender", lang="en", surface="", type="context",
               canonical_id="", bias_type="gender")
    g.add_edge("context::crows_pairs::1::en::gender", "group::en::men",
               relation="stereotype_of", weight=1.0)
    g.add_edge("context::crows_pairs::1::en::gender", "group::en::women",
               relation="anti_stereotype_of", weight=1.0)
    g.add_edge("group::en::men", "group::bn::purush", relation="same_as", weight=1.0)
    return g


def test_extract_json():
    assert extract_json('noise {"a": 1} tail')["a"] == 1
    assert extract_json("```json\n{\"choice\": \"A\"}\n```")["choice"] == "A"
    assert extract_json("no json here") is None


def test_extract_choice():
    assert extract_choice("The answer is B.", ["A", "B"]) == "B"
    assert extract_choice('{"answer":"A"}', ["A", "B"]) == "A"
    assert extract_choice("totally unrelated", ["A", "B"]) is None


def test_roundrobin():
    rr = RoundRobin(["k1", "k2"])
    assert [rr.next() for _ in range(4)] == ["k1", "k2", "k1", "k2"]
    assert bool(RoundRobin([])) is False


def test_csv_resume_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.csv")
        append_row(path, {"model": "m1", "lang": "en", "fbr": 1.0})
        append_row(path, {"model": "m1", "lang": "hi", "fbr": 2.0})
        keys = done_keys(path, ["model", "lang"])
        assert ("m1", "en") in keys and ("m1", "hi") in keys
        assert ("m1", "bn") not in keys
        assert len(read_csv_dicts(path)) == 2


def test_kg_schema_constants():
    assert NODE_COLS == ["id", "lang", "surface", "type", "canonical_id", "bias_type"]
    assert EDGE_COLS == ["src", "dst", "relation", "weight"]


def test_kg_algos():
    g = _tiny_graph()
    sibs = kg_algos.group_siblings(g, "group::en::men")
    assert "group::en::women" in sibs            # same lang + bias_type
    assert "group::bn::purush" not in sibs        # different language
    counters = kg_algos.counter_stereotype_groups(g, "context::crows_pairs::1::en::gender")
    assert counters == ["group::en::women"]
    assert kg_algos.bridge_to_language(g, "group::en::men", "bn") == "group::bn::purush"
    ppr = kg_algos.personalised_pagerank(g, {"group::en::men": 1.0}, alpha=0.85, max_iter=100)
    # seeded mass must reach the bn counterpart through same_as
    assert ppr["group::bn::purush"] > 0.0


def test_e4_prompt_and_rag():
    g = _tiny_graph()
    pair = {"index": "1", "language": "en", "bias_type": "gender",
            "target_stereotypical": "men", "target_anti_stereotypical": "women",
            "sentence_stereotypical": "The men were strong."}
    user, stereo_label = e4_core._prompt(pair, "")
    assert stereo_label in ("A", "B")
    assert "men" in user and "women" in user and "____" in user
    ctx = e4_core.rag_context(g, "crows_pairs", "1", "en", "gender")
    assert "women" in ctx


def test_e4_evaluate_deterministic():
    """A subject that always answers the stereotypical word -> expressed bias 100."""
    g = _tiny_graph()
    pairs = [{"index": "1", "language": "en", "target_stereotypical": "men",
              "target_anti_stereotypical": "women",
              "sentence_stereotypical": "The men were strong."}]

    def always_stereo(system, user):
        # echo whichever letter labels 'men'
        return "A" if user.find("A) men") != -1 else "B"

    res = e4_core.evaluate(always_stereo, pairs, g, use_rag=False, judge=None)
    assert res["expressed_bias"] == 100.0 and res["n_decided"] == 1


def test_btg_arithmetic():
    fbr_direct, fbr_bridge = -1.2, 9.4   # direct transfer fails; KG bridge repairs
    btg = fbr_bridge - fbr_direct
    assert round(btg, 4) == 10.6


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    passed = 0
    for fn in ALL:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"CORE TESTS PASSED: {passed}/{len(ALL)}")


if __name__ == "__main__":
    main()
