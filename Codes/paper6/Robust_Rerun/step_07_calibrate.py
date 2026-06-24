"""STEP 07 (API, VAL concepts) - calibrate retrieval to remove over-correction.

On VAL concepts (held out of the graph, distinct from TEST), sweep the number of
injected counter-evidence facts for the kg_rag condition on the black-box API
subjects, and pick the value whose post-retrieval bias lands closest to neutral
(50). The chosen n_facts is written for the TEST-time runs (steps 08-09) to use,
so the headline retrieval is calibrated rather than naively over-correcting.

Outputs: results/calibration.csv, results/calibration.json
Usage:   python step_07_calibrate.py [--resume]
"""
import argparse
import os
from collections import defaultdict

from lib.paths import cfg, load_split
from Common_00.common import append_row, done_keys, resolve, write_json
from Common_00.api_subjects import build_subject
from Common_00.kg_io import load_graph
from lib import rag_leakfree as R

OUT = JSON = None


def run(config, resume=True):
    global OUT, JSON
    OUT = resolve(os.path.join(config["paths"]["results"], "calibration.csv"))
    JSON = resolve(os.path.join(config["paths"]["results"], "calibration.json"))
    split = load_split(config)
    graph = load_graph(config)
    pools = R.build_pools(graph, split)
    grid = config["calibrate"]["n_facts_grid"]
    langs = config["data"]["crows_pairs"]["languages"]
    datasets = ["crows_pairs", "indian_bias"]
    done = done_keys(OUT, ["subject_model", "dataset", "language", "n_facts"]) if resume else set()

    dev_by_nfacts = defaultdict(list)
    for short in config["api_predict_only"]:
        subj = build_subject(short)
        if subj is None:
            print(f"  {short}: no keys/route; skipping.")
            continue
        chat = subj.chat
        for dataset in datasets:
            english = R.english_lookup(config, dataset)
            for lang in langs:
                pairs = R.build_pairs_for(config, split, dataset, lang, which="val",
                                          max_pairs=config["rag"]["max_pairs_per_lang_api"])
                if not pairs:
                    continue
                for n_facts in grid:
                    if (short, dataset, lang, str(n_facts)) in done:
                        continue
                    res = R.evaluate(chat, pairs, pools, split, "kg_rag", n_facts,
                                     judge=None, english=english)
                    append_row(OUT, {
                        "subject_model": short, "dataset": dataset, "language": lang,
                        "n_facts": n_facts, "expressed_bias": round(res["expressed_bias"], 4),
                        "deviation": round(res["deviation"], 4),
                        "n_decided": res["n_decided"], "status": "ok"})
                    if res["deviation"] == res["deviation"]:  # not NaN
                        dev_by_nfacts[n_facts].append(res["deviation"])
                print(f"  {short}/{dataset}/{lang}: swept n_facts {grid}")

    # pick the grid value with the smallest mean absolute deviation from neutral
    means = {nf: (sum(v) / len(v)) for nf, v in dev_by_nfacts.items() if v}
    recommended = min(means, key=means.get) if means else config["rag"]["max_neighbours"]
    write_json(JSON, {"recommended_n_facts": recommended,
                      "mean_deviation_by_n_facts": {str(k): round(v, 4) for k, v in means.items()}})
    print("=== STEP 07  calibration ===")
    print(f"  mean deviation by n_facts: {means}")
    print(f"  recommended n_facts: {recommended}")
    print(f"  written: {OUT} and {JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    run(cfg(), resume=args.resume)


if __name__ == "__main__":
    main()
