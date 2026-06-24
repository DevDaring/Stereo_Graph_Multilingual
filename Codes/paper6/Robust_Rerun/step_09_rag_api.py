"""STEP 09 (API, TEST concepts) - leakage-free retrieval on the black-box subjects.

Same protocol and conditions as step_08, but the subject models are the API
decoders (deepseek-chat, llama-3.3-70b, gpt-oss-20b). This is the dimension no
parameter-only baseline can run on: a weight-free, cross-lingual, inspectable
intervention, evaluated on held-out concepts. Answer extraction uses the active
judge (LinkAPI gemini-2.5-pro) with NO cross-provider fallback.

Output: results/rag_api.csv
Usage:  python step_09_rag_api.py [--resume]
"""
import argparse
import json
import os

from lib.paths import cfg, load_split
from Common_00.common import append_row, done_keys, resolve
from Common_00.providers import get_judge
from Common_00.api_subjects import build_subject
from Common_00.kg_io import load_graph
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
    OUT = resolve(os.path.join(config["paths"]["results"], "rag_api.csv"))
    split = load_split(config)
    graph = load_graph(config)
    pools = R.build_pools(graph, split)
    judge = get_judge(config)
    conditions = config["rag"]["conditions"]
    langs = config["data"]["crows_pairs"]["languages"]
    datasets = ["crows_pairs", "indian_bias"]
    n_kg, n_def = recommended_n_facts(config), config["rag"]["max_neighbours"]
    done = done_keys(OUT, ["subject_model", "dataset", "language", "condition"]) if resume else set()

    for short in config["api_predict_only"]:
        subj = build_subject(short)
        if subj is None:
            print(f"  {short}: no keys/route; skipping.")
            continue
        chat = subj.chat
        for dataset in datasets:
            english = R.english_lookup(config, dataset)
            for lang in langs:
                pairs = R.build_pairs_for(config, split, dataset, lang, which="test",
                                          max_pairs=config["rag"]["max_pairs_per_lang_api"])
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
        print(f"  {short}: done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    print("=== STEP 09  leakage-free RAG (API subjects, TEST concepts) ===")
    run(cfg(), resume=args.resume)
    print(f"  written: {OUT}")


if __name__ == "__main__":
    main()
