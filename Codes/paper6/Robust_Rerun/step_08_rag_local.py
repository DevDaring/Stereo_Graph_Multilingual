"""STEP 08 (GPU, TEST concepts) - leakage-free retrieval on the local decoders.

For each local decoder, on items whose concept is HELD OUT of the graph, every
retrieval condition is evaluated: base, safety_prompt, translate_en, flat_dict,
kg_rag_monolingual, kg_rag. Retrieval draws counter-evidence only from TRAIN
concepts and never the query item's own targets, so any movement is a real
generalisation. kg_rag uses the calibrated n_facts from step_07.

Output: results/rag_local.csv
Usage:  python step_08_rag_local.py [--resume]
"""
import argparse
import json
import os

from lib.paths import cfg, load_split
from Common_00.common import append_row, done_keys, resolve, set_seed
from Common_00.providers import get_judge
from Common_00.kg_io import load_graph
from Common_00 import model_utils as M
from lib import rag_leakfree as R

OUT = None


def recommended_n_facts(config):
    p = resolve(os.path.join(config["paths"]["results"], "calibration.json"))
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return int(json.load(f).get("recommended_n_facts", config["rag"]["max_neighbours"]))
        except Exception:
            pass
    return config["rag"]["max_neighbours"]


def run(config, resume=True):
    global OUT
    OUT = resolve(os.path.join(config["paths"]["results"], "rag_local.csv"))
    set_seed(config["experiment"]["seeds"][0])
    split = load_split(config)
    graph = load_graph(config)
    pools = R.build_pools(graph, split)
    judge = get_judge(config)
    conditions = config["rag"]["conditions"]
    langs = config["data"]["crows_pairs"]["languages"]
    datasets = ["crows_pairs", "indian_bias"]
    n_kg, n_def = recommended_n_facts(config), config["rag"]["max_neighbours"]
    done = done_keys(OUT, ["subject_model", "dataset", "language", "condition"]) if resume else set()

    for m in config["white_box"]["llms"]:
        short, hf_id = m["short"], m["hf_id"]
        pending = [(d, l, c) for d in datasets for l in langs for c in conditions
                   if (short, d, l, c) not in done]
        if not pending:
            print(f"  {short}: all done, skipping load.")
            continue
        try:
            llm = M.load_llm(config, hf_id)
        except Exception as e:
            print(f"  {short}: model load failed ({type(e).__name__}: {e}); skipping.")
            continue
        chat = lambda system, user: llm.generate(system, user, max_new_tokens=8)  # noqa: E731
        try:
            for dataset in datasets:
                english = R.english_lookup(config, dataset)
                for lang in langs:
                    pairs = R.build_pairs_for(config, split, dataset, lang, which="test",
                                              max_pairs=config["rag"]["max_pairs_per_lang"])
                    if not pairs:
                        continue
                    for cond in conditions:
                        if (short, dataset, lang, cond) in done:
                            continue
                        n_facts = n_kg if cond == "kg_rag" else n_def
                        res = R.evaluate(chat, pairs, pools, split, cond, n_facts,
                                         judge=judge, english=english)
                        append_row(OUT, {
                            "subject_model": short, "dataset": dataset, "language": lang,
                            "condition": cond, "n_facts": n_facts,
                            "expressed_bias": round(res["expressed_bias"], 4),
                            "deviation": round(res["deviation"], 4),
                            "n_decided": res["n_decided"], "refusals": res["refusals"],
                            "split": "test", "status": "ok"})
                    print(f"  {short}/{dataset}/{lang}: {len(conditions)} conditions done.")
        except Exception as e:
            print(f"  {short}: run error ({type(e).__name__}: {e}); moving on.")
        finally:
            try:
                llm.unload()
            except Exception:
                pass
        print(f"  {short}: done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    print("=== STEP 08  leakage-free RAG (local decoders, TEST concepts) ===")
    run(cfg(), resume=args.resume)
    print(f"  written: {OUT}")


if __name__ == "__main__":
    main()
