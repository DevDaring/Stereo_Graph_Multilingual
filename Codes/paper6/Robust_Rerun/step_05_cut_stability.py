"""STEP 05 (GPU) - multi-seed minimal-cut stability for the disjointness claim.

The 0.035 cross-lingual Jaccard rested on a SINGLE seed. Here the minimal cut is
relearned for 5 seeds per (decoder, dataset, language) so that step_10 can report
the overlap with variance and against a matched-sparsity random-mask null. Reuses
the verified Paper 5 cut compute (Common_00.model_utils -> paper5/src).

Output: results/cut_units.csv   (one row per model/dataset/language/seed; the cut
        unit signature is stored so all overlaps are computed on CPU in step_10).
Usage:  python step_05_cut_stability.py [--resume]
"""
import argparse
import os

from lib.paths import cfg
from Common_00.common import append_row, done_keys, resolve, set_seed
from Common_00 import reuse
from Common_00 import model_utils as M

OUT = None


def run(config, resume=True):
    global OUT
    OUT = resolve(os.path.join(config["paths"]["results"], "cut_units.csv"))
    cs = config["cut_stability"]
    seeds = cs["seeds"]
    cut_cfg = {"steps": cs["steps"], "max_pairs": cs["max_pairs"], "lr": cs["lr"],
               "lambda_utility": cs["lambda_utility"], "gamma_sparsity": cs["gamma_sparsity"],
               "mask_init": cs["mask_init"]}
    datasets = ["crows_pairs", "indian_bias"]
    langs = ["en", "hi", "bn"]
    done = done_keys(OUT, ["subject_model", "dataset", "language", "seed"]) if resume else set()

    for m in config["white_box"]["llms"]:
        short, hf_id = m["short"], m["hf_id"]
        pending = [(d, l, s) for d in datasets for l in langs for s in seeds
                   if (short, d, l, str(s)) not in done]
        if not pending:
            print(f"  {short}: all done, skipping load.")
            continue
        try:
            llm = M.load_llm(config, hf_id)
        except Exception as e:
            print(f"  {short}: model load failed ({type(e).__name__}: {e}); skipping.")
            continue
        hidden = getattr(getattr(getattr(llm, "model", None), "config", None), "hidden_size", None)
        try:
            for dataset in datasets:
                band = reuse.emergence_band_for(config, short, dataset)
                if not band:
                    print(f"  {short}/{dataset}: no reusable band; skipping.")
                    continue
                n_total = (len(band) * hidden) if hidden else ""
                for lang in langs:
                    pairs = M.build_pairs(config, dataset, lang, max_pairs=cs["max_pairs"])
                    if not pairs:
                        continue
                    for s in seeds:
                        if (short, dataset, lang, str(s)) in done:
                            continue
                        set_seed(s)
                        mask = M.learn_minimal_cut(llm, pairs, band, {**cut_cfg, "seed": s})
                        units = M.cut_units(mask)
                        units_str = "|".join(f"{a}:{b}" for a, b in units)
                        append_row(OUT, {
                            "subject_model": short, "dataset": dataset, "language": lang,
                            "seed": s, "n_units": len(units), "n_total": n_total,
                            "band": "|".join(str(x) for x in band),
                            "units": units_str, "status": "ok"})
                    print(f"  {short}/{dataset}/{lang}: {len(seeds)} seeds done.")
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
    print("=== STEP 05  multi-seed cut stability ===")
    run(cfg(), resume=args.resume)
    print(f"  written: {OUT}")


if __name__ == "__main__":
    main()
