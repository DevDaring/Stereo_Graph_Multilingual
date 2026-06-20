"""STEP 04  (GPU_Only_04) - E3: KG-bridged cross-lingual repair transfer (headline).

Reuses the Paper 5 emergence band and the Paper 5 per-language learned cut. The only
new GPU work is: learn the English cut once per (decoder, dataset), then evaluate
that English cut on Hindi/Bengali (the failing 'direct transfer'). The KG-bridged
result is the per-language cut, which Paper 5 already computed (loaded, not
recomputed). Bridge Transfer Gain BTG = fbr(kg_bridged) - fbr(direct).

Outputs: results/e3_bridge.csv, results/e3_btg.csv
Usage:   python GPU_Only_04/e3_bridge.py [--resume]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from Common_00.common import append_row, done_keys, load_config, resolve, set_seed
from Common_00 import reuse
from Common_00 import model_utils as M

E3 = None
BTG = None


def _p5_cut(rows, model, dataset, language, variant="learned_min_cut"):
    for r in rows:
        if (r.get("subject_model") == model and r.get("dataset") == dataset
                and r.get("language") == language and r.get("variant") == variant):
            return r
    return None


def run(config, resume=True):
    global E3, BTG
    E3 = resolve(os.path.join(config["paths"]["results"], "e3_bridge.csv"))
    BTG = resolve(os.path.join(config["paths"]["results"], "e3_btg.csv"))
    set_seed(config["experiment"]["seeds"][0])

    p5_cut = reuse.cut_results(config)
    gate = reuse.competence_gate(config)
    done = done_keys(E3, ["subject_model", "dataset", "target_lang", "condition"]) if resume else set()
    btg_done = done_keys(BTG, ["subject_model", "dataset", "target_lang"]) if resume else set()

    decoders = [m for m in config["white_box"]["llms"]]
    datasets = ["crows_pairs", "indian_bias"]
    cut_cfg = {"steps": config["e3_bridge"]["cut_steps"], "max_pairs": 64,
               "lr": 0.05, "lambda_utility": 1.0, "gamma_sparsity": 0.01, "mask_init": 3.0}

    for m in decoders:
        short, hf_id = m["short"], m["hf_id"]
        # only load the model if there is work left for it
        pending = [(d, l) for d in datasets for l in ("hi", "bn")
                   if (short, d, l, "direct") not in done]
        if not pending:
            print(f"  {short}: all done, skipping load.")
            continue
        try:
            llm = M.load_llm(config, hf_id)
        except Exception as e:
            print(f"  {short}: model load failed ({type(e).__name__}: {e}); skipping.")
            continue
        try:
            for dataset in datasets:
                band = reuse.emergence_band_for(config, short, dataset)
                if not band:
                    print(f"  {short}/{dataset}: no reusable band; skipping.")
                    continue
                pairs_en = M.build_pairs(config, dataset, "en", max_pairs=64)
                if not pairs_en:
                    continue
                en_mask = M.learn_minimal_cut(llm, pairs_en, band, cut_cfg)
                for lang in ("hi", "bn"):
                    pairs = M.build_pairs(config, dataset, lang,
                                          max_pairs=config["e4_kgrag"]["max_pairs_per_lang"])
                    if not pairs:
                        continue
                    p5 = _p5_cut(p5_cut, short, dataset, lang)
                    base = float(p5["bias_baseline"]) if p5 else M.bias_score(llm, pairs)
                    base_dev = M.deviation(base)

                    # (A) direct transfer: English cut units applied to this language
                    if (short, dataset, lang, "direct") not in done:
                        after = M.eval_with_mask(llm, pairs, en_mask)
                        fbr_direct = base_dev - M.deviation(after)
                        append_row(E3, {
                            "subject_model": short, "dataset": dataset, "target_lang": lang,
                            "bias_type": "all", "condition": "direct",
                            "cut_size": len(M.cut_units(en_mask)),
                            "bias_baseline": round(base, 4), "bias_after": round(after, 4),
                            "fbr": round(fbr_direct, 4), "n_pairs": len(pairs),
                            "seed": cut_cfg.get("seed", 42),
                            "competence_ok": gate.get((short, lang), False), "status": "ok"})
                    else:
                        fbr_direct = None

                    # (B/C) KG-bridged = the per-language cut Paper 5 already learned
                    fbr_bridge = float(p5["cut_bias_reduction"]) if p5 else None
                    if (short, dataset, lang, "kg_bridged") not in done:
                        append_row(E3, {
                            "subject_model": short, "dataset": dataset, "target_lang": lang,
                            "bias_type": "all", "condition": "kg_bridged",
                            "cut_size": (p5.get("cut_size") if p5 else ""),
                            "bias_baseline": round(base, 4),
                            "bias_after": (round(base - 0, 4) if p5 is None else
                                           round(float(p5["bias_after_cut"]), 4)),
                            "fbr": (round(fbr_bridge, 4) if fbr_bridge is not None else "NaN"),
                            "n_pairs": (p5.get("n_pairs") if p5 else len(pairs)),
                            "seed": 42,
                            "competence_ok": gate.get((short, lang), False),
                            "status": "ok" if p5 else "nan"})

                    # Bridge Transfer Gain
                    if (fbr_direct is not None and fbr_bridge is not None
                            and (short, dataset, lang) not in btg_done):
                        append_row(BTG, {
                            "subject_model": short, "dataset": dataset, "target_lang": lang,
                            "fbr_direct": round(fbr_direct, 4),
                            "fbr_kg_bridged": round(fbr_bridge, 4),
                            "btg": round(fbr_bridge - fbr_direct, 4),
                            "competence_ok": gate.get((short, lang), False), "status": "ok"})
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
    print("=== STEP 04  E3 KG-bridged transfer ===")
    run(load_config(), resume=args.resume)
    print(f"  written: {E3}  and  {BTG}")


if __name__ == "__main__":
    main()
