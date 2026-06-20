"""STEP 06  (CPU_Only_06) - E4: KG-retrieval-augmented inference for API subject models.

Black-box decoders (gpt-oss-20b, deepseek-chat, llama-3.3-70b) answer each masked
pair with and without an MS-SKG counter-stereotype context, and the configured judge
(gemini primary) extracts the choice when the subject's reply is not a clean A/B.
Expressed bias = % stereotypical wins; KG-RAG gain = no_rag - kg_rag deviation.
No GPU. Subject calls round-robin over the subject provider's keys (no fallback).

Reference baseline (optional): Paper 3 llm_clti.csv expressed-bias rows.

Output: results/e4_kgrag_api.csv
Usage:  python CPU_Only_06/e4_kgrag_api.py [--resume]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from Common_00.common import append_row, done_keys, load_config, resolve, set_seed
from Common_00.providers import get_judge
from Common_00.api_subjects import SUBJECTS, build_subject
from Common_00 import e4_core, reuse
from Common_00.kg_io import load_graph

OUT = None


def _p3_baseline(p3_rows, short, dataset, lang):
    """Optional Paper 3 expressed-bias reference (column expressed_baseline_{lang})."""
    col = f"expressed_baseline_{lang}"
    for r in p3_rows:
        model = r.get("subject_model") or r.get("model") or ""
        if short in str(model) and r.get("dataset") in (dataset, None, "") and _isnum(r.get(col)):
            return float(r[col])
    return ""


def _isnum(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def run(config, resume=True):
    global OUT
    OUT = resolve(os.path.join(config["paths"]["results"], "e4_kgrag_api.csv"))
    set_seed(config["experiment"]["seeds"][0])
    judge = get_judge(config)
    graph = load_graph(config)
    p3 = reuse.llm_clti(config)
    done = done_keys(OUT, ["subject_model", "dataset", "language", "condition"]) if resume else set()
    max_pairs = config["e4_kgrag"]["max_pairs_per_lang"]
    langs = config["data"]["crows_pairs"]["languages"]

    for short in config["api_predict_only"]:
        if short not in SUBJECTS:
            print(f"  {short}: no provider route configured; skipping.")
            continue
        subject = build_subject(short)
        if subject is None:
            print(f"  {short}: no API keys in .env; skipping.")
            continue
        chat = lambda system, user: subject.chat(system, user)  # noqa: E731
        for dataset in ("crows_pairs", "indian_bias"):
            for lang in langs:
                if all((short, dataset, lang, c) in done for c in ("no_rag", "kg_rag")):
                    continue
                pairs = e4_core.build_pairs(config, dataset, lang, max_pairs=max_pairs)
                if not pairs:
                    continue
                base_ref = _p3_baseline(p3, short, dataset, lang)
                res_no, res_kg = {}, {}
                for cond, use_rag in (("no_rag", False), ("kg_rag", True)):
                    if (short, dataset, lang, cond) in done:
                        continue
                    res = e4_core.evaluate(chat, pairs, graph, use_rag, judge=judge, dataset=dataset)
                    (res_kg if use_rag else res_no).update(res)
                    append_row(OUT, {
                        "subject_model": short, "dataset": dataset, "language": lang,
                        "condition": cond, "expressed_bias": round(res["expressed_bias"], 4),
                        "deviation": round(abs(res["expressed_bias"] - 50.0), 4),
                        "n_decided": res["n_decided"], "refusals": res["refusals"],
                        "paper3_baseline": base_ref, "status": "ok"})
                if res_no and res_kg:
                    gain = abs(res_no["expressed_bias"] - 50.0) - abs(res_kg["expressed_bias"] - 50.0)
                    print(f"  {short}/{dataset}/{lang}: KG-RAG gain={gain:+.2f}")
        print(f"  {short}: done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    print("=== STEP 06  E4 KG-RAG (API subjects) ===")
    run(load_config(), resume=args.resume)
    print(f"  written: {OUT}")


if __name__ == "__main__":
    main()
