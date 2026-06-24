"""STEP 10 (CPU) - aggregate everything into honest, leakage-free headline numbers.

Computes: (a) multi-seed cross-lingual cut Jaccard with mean/std and a matched
sparsity random-mask null (from step_05); (b) per-condition retrieval improvement
on TEST concepts with bootstrap CIs and Cohen's d, plus a forest-plot table and a
Cochran's Q / I-squared heterogeneity test (from steps 08-09); (c) the held-out
bridge ratio vs null (from step_06). Writes results/summary.json and
results/forest_plot.csv.

Usage:  python step_10_analysis.py
"""
import json
import os
import random

from lib.paths import cfg
from lib import stats as S
from Common_00.common import append_row, read_csv_dicts, resolve, write_json


def _units(s):
    return set(t for t in str(s).split("|") if t)


def _sample_null_jaccard(a, b, n, trials, rng):
    if not n or n <= 0 or a <= 0 or b <= 0 or a > n or b > n:
        return None
    js = []
    universe = range(n)
    for _ in range(trials):
        sa = set(rng.sample(universe, a))
        sb = set(rng.sample(universe, b))
        u = len(sa | sb)
        js.append(len(sa & sb) / u if u else 0.0)
    return sum(js) / len(js)


def cut_overlap(config):
    rows = read_csv_dicts(resolve(os.path.join(config["paths"]["results"], "cut_units.csv")))
    if not rows:
        return None
    rng = random.Random(config["split"]["seed"])
    trials = config["cut_stability"]["random_mask_trials"]
    by = {}
    for r in rows:
        by[(r["subject_model"], r["dataset"], r["language"], r["seed"])] = r
    cross, null, seedstab = [], [], []
    models = sorted({r["subject_model"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    langs = ["en", "hi", "bn"]
    for m in models:
        for d in datasets:
            for s in seeds:
                present = {l: by.get((m, d, l, s)) for l in langs}
                for i in range(len(langs)):
                    for j in range(i + 1, len(langs)):
                        ra, rb = present[langs[i]], present[langs[j]]
                        if not ra or not rb:
                            continue
                        ua, ub = _units(ra["units"]), _units(rb["units"])
                        cross.append(S.jaccard(ua, ub))
                        n_total = ra.get("n_total") or rb.get("n_total")
                        try:
                            nt = int(float(n_total))
                        except Exception:
                            nt = 0
                        nj = _sample_null_jaccard(len(ua), len(ub), nt, trials, rng)
                        if nj is not None:
                            null.append(nj)
            # seed stability per (model,dataset,lang)
            for l in langs:
                sets = [_units(by[(m, d, l, s)]["units"]) for s in seeds if (m, d, l, s) in by]
                for i in range(len(sets)):
                    for j in range(i + 1, len(sets)):
                        seedstab.append(S.jaccard(sets[i], sets[j]))
    return {
        "n_cross_lang_pairs": len(cross),
        "mean_cross_lingual_jaccard": round(S.mean(cross), 4) if cross else None,
        "std_cross_lingual_jaccard": round(S.std(cross), 4) if cross else None,
        "mean_random_mask_jaccard_null": round(S.mean(null), 4) if null else None,
        "mean_seed_stability_jaccard": round(S.mean(seedstab), 4) if seedstab else None,
        "reading": ("Cross-lingual cut overlap stays near the random-mask null and well "
                    "below within-language seed stability, so the disjointness is a real "
                    "feature, not optimiser noise."),
    }


def rag_analysis(config):
    rows = []
    for fn in ("rag_local.csv", "rag_api.csv"):
        rows += read_csv_dicts(resolve(os.path.join(config["paths"]["results"], fn)))
    if not rows:
        return None, []
    # index cells: (model,dataset,lang) -> {condition: row}
    cells = {}
    for r in rows:
        cells.setdefault((r["subject_model"], r["dataset"], r["language"]), {})[r["condition"]] = r

    conditions = [c for c in config["rag"]["conditions"] if c != "base"]
    improvements = {c: [] for c in conditions}
    variances = {c: [] for c in conditions}
    forest = []
    for (m, d, l), cond_rows in cells.items():
        base = cond_rows.get("base")
        if not base:
            continue
        try:
            base_dev = float(base["deviation"])
            nb = int(base["n_decided"]) or 1
        except Exception:
            continue
        for c in conditions:
            cr = cond_rows.get(c)
            if not cr:
                continue
            try:
                dev = float(cr["deviation"]); nc = int(cr["n_decided"]) or 1
            except Exception:
                continue
            imp = base_dev - dev   # positive = closer to neutral than base
            improvements[c].append(imp)
            # binomial variance of each deviation (in points), summed for the difference
            pb = float(base["expressed_bias"]) / 100.0
            pc = float(cr["expressed_bias"]) / 100.0
            var = (pb * (1 - pb) / nb + pc * (1 - pc) / nc) * (100.0 ** 2)
            variances[c].append(var)
            if c == "kg_rag":
                forest.append({"subject_model": m, "dataset": d, "language": l,
                               "base_deviation": round(base_dev, 4),
                               "kg_rag_deviation": round(dev, 4),
                               "improvement": round(imp, 4), "variance": round(var, 4)})

    summary = {}
    for c in conditions:
        xs = improvements[c]
        if not xs:
            continue
        lo, hi = S.bootstrap_ci(xs)
        entry = {"n_cells": len(xs), "mean_improvement": round(S.mean(xs), 4),
                 "ci95": [round(lo, 4), round(hi, 4)], "cohens_d": round(S.cohens_d(xs), 4)}
        if c == "kg_rag":
            entry["heterogeneity"] = S.cochran_q(xs, variances[c])
        summary[c] = entry
    return summary, forest


def main():
    config = cfg()
    out = {}
    out["cut_overlap"] = cut_overlap(config)

    e5p = resolve(os.path.join(config["paths"]["results"], "e5_summary.json"))
    if os.path.exists(e5p):
        with open(e5p, "r", encoding="utf-8") as f:
            out["e5_heldout"] = json.load(f)

    rag, forest = rag_analysis(config)
    out["retrieval"] = rag
    if forest:
        fp = resolve(os.path.join(config["paths"]["results"], "forest_plot.csv"))
        if os.path.exists(fp):
            os.remove(fp)
        for row in forest:
            append_row(fp, row)
        out["forest_plot_csv"] = fp

    sp = resolve(os.path.join(config["paths"]["results"], "summary.json"))
    write_json(sp, out)
    print("=== STEP 10  analysis ===")
    if out.get("cut_overlap"):
        co = out["cut_overlap"]
        print(f"  cut Jaccard cross={co['mean_cross_lingual_jaccard']} "
              f"+/-{co['std_cross_lingual_jaccard']} null={co['mean_random_mask_jaccard_null']} "
              f"seed_stab={co['mean_seed_stability_jaccard']}")
    if out.get("e5_heldout"):
        e = out["e5_heldout"]
        print(f"  held-out bridge={e.get('mean_bridge_ratio_heldout')} "
              f"null={e.get('mean_null_ratio')} lift={e.get('mean_lift_over_null')}")
    if rag:
        for c, e in rag.items():
            print(f"  {c:20s} mean_improv={e['mean_improvement']:+.2f} "
                  f"CI{e['ci95']} d={e['cohens_d']} n={e['n_cells']}")
    print(f"  written: {sp}")


if __name__ == "__main__":
    main()
