"""STEP 05  (GPU_Only_05) - E2: KG-guided counterfactual data augmentation (CDA).

Implements counterfactual data augmentation [Zmigrod et al., 2019], KG-guided. For
each English pair it adds counterfactuals by swapping the stereotypical group for KG
siblings (same bias_type/language), then retrains the Paper 5 minimal cut on the
augmented set and measures English bias reduction. Baselines: no-CDA (reused from
Paper 5) and random-swap CDA (no KG). GPU work = two cut trainings per (decoder,
dataset).

Output: results/e2_cda.csv
Usage:  python GPU_Only_05/e2_cda.py [--resume]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from Common_00.common import append_row, done_keys, load_config, resolve, set_seed
from Common_00 import reuse
from Common_00.kg_algos import group_siblings, surface
from Common_00.kg_io import load_graph
from Common_00 import model_utils as M

OUT = None


def _gid(lang, s):
    return f"group::{lang}::{' '.join(s.split()).strip().lower()}"


def augment(pairs, graph, lang, n_per, use_kg):
    extra = []
    group_pool = [a["surface"] for _, a in graph.nodes(data=True)
                  if a.get("type") == "group" and a.get("lang") == lang]
    for p in pairs:
        if use_kg:
            sibs = [surface(graph, s) for s in group_siblings(graph, _gid(lang, p["target_stereotypical"]))]
        else:
            sibs = list(group_pool)
        random.shuffle(sibs)
        for sib in [s for s in sibs if s and s != p["target_stereotypical"]][:n_per]:
            extra.append({
                "sentence_stereotypical": M.fill(_strip(p), sib),
                "sentence_anti_stereotypical": p["sentence_anti_stereotypical"],
                "target_stereotypical": sib,
                "target_anti_stereotypical": p["target_anti_stereotypical"]})
    return pairs + extra


def _strip(p):
    """Recover the masked template from a filled stereotypical sentence."""
    return p["sentence_stereotypical"].replace(p["target_stereotypical"], "MASK", 1)


def run(config, resume=True):
    global OUT
    OUT = resolve(os.path.join(config["paths"]["results"], "e2_cda.csv"))
    set_seed(config["experiment"]["seeds"][0])
    p5 = reuse.cut_results(config)
    done = done_keys(OUT, ["subject_model", "dataset", "language", "method"]) if resume else set()
    graph = load_graph(config)

    cut_cfg = {"steps": config["e2_cda"]["cut_steps"], "max_pairs": 96,
               "lr": 0.05, "lambda_utility": 1.0, "gamma_sparsity": 0.01, "mask_init": 3.0}
    n_per = config["e2_cda"]["max_counterfactuals_per_edge"]

    for m in config["white_box"]["llms"]:
        short, hf_id = m["short"], m["hf_id"]
        for dataset in ("crows_pairs", "indian_bias"):
            # no-CDA baseline reused from Paper 5
            base = next((r for r in p5 if r["subject_model"] == short and r["dataset"] == dataset
                         and r["language"] == "en" and r["variant"] == "learned_min_cut"), None)
            if base and (short, dataset, "en", "no_cda") not in done:
                append_row(OUT, {"subject_model": short, "dataset": dataset, "language": "en",
                                 "method": "no_cda", "bias_baseline": base["bias_baseline"],
                                 "bias_after": base["bias_after_cut"],
                                 "bias_reduction": base["cut_bias_reduction"],
                                 "n_pairs": base["n_pairs"], "seed": 42, "status": "ok"})

        pending = [(d, meth) for d in ("crows_pairs", "indian_bias")
                   for meth in ("random_cda", "kg_cda")
                   if (short, d, "en", meth) not in done]
        if not pending:
            continue
        try:
            llm = M.load_llm(config, hf_id)
        except Exception as e:
            print(f"  {short}: model load failed ({type(e).__name__}: {e}); skipping.")
            continue
        try:
            for dataset in ("crows_pairs", "indian_bias"):
                band = reuse.emergence_band_for(config, short, dataset)
                if not band:
                    continue
                pairs_en = M.build_pairs(config, dataset, "en", max_pairs=48)
                if not pairs_en:
                    continue
                base_bias = M.bias_score(llm, pairs_en)
                for meth, use_kg in (("random_cda", False), ("kg_cda", True)):
                    if (short, dataset, "en", meth) in done:
                        continue
                    aug = augment(pairs_en, graph, "en", n_per, use_kg)
                    mask = M.learn_minimal_cut(llm, aug, band, cut_cfg)
                    after = M.eval_with_mask(llm, pairs_en, mask)
                    append_row(OUT, {
                        "subject_model": short, "dataset": dataset, "language": "en",
                        "method": meth, "bias_baseline": round(base_bias, 4),
                        "bias_after": round(after, 4),
                        "bias_reduction": round(M.deviation(base_bias) - M.deviation(after), 4),
                        "stereotypes_covered": len({p["target_stereotypical"] for p in aug}),
                        "n_pairs": len(aug), "seed": 42, "status": "ok"})
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
    print("=== STEP 05  E2 KG-guided CDA ===")
    run(load_config(), resume=args.resume)
    print(f"  written: {OUT}")


if __name__ == "__main__":
    main()
