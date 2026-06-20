"""STEP 07  (GPU_Only_07) - E4: KG-retrieval-augmented inference for LOCAL decoders.

Same expressed-bias protocol as step 06, but the subject models are the two local
decoders (qwen2.5-7b, aya-23-8b) generated on-GPU. Each masked pair is answered with
and without the MS-SKG counter-stereotype context; the judge (gemini primary) extracts
the choice only when the generated reply is not a clean A/B. GPU work = one short
greedy generation per pair per condition.

Output: results/e4_kgrag_local.csv
Usage:  python GPU_Only_07/e4_kgrag_local.py [--resume]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from Common_00.common import append_row, done_keys, load_config, resolve, set_seed
from Common_00.providers import get_judge
from Common_00 import e4_core
from Common_00.kg_io import load_graph
from Common_00 import model_utils as M

OUT = None


def run(config, resume=True):
    global OUT
    OUT = resolve(os.path.join(config["paths"]["results"], "e4_kgrag_local.csv"))
    set_seed(config["experiment"]["seeds"][0])
    judge = get_judge(config)
    graph = load_graph(config)
    done = done_keys(OUT, ["subject_model", "dataset", "language", "condition"]) if resume else set()
    max_pairs = config["e4_kgrag"]["max_pairs_per_lang"]
    langs = config["data"]["crows_pairs"]["languages"]

    for m in config["white_box"]["llms"]:
        short, hf_id = m["short"], m["hf_id"]
        pending = [(d, l, c) for d in ("crows_pairs", "indian_bias") for l in langs
                   for c in ("no_rag", "kg_rag") if (short, d, l, c) not in done]
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
            for dataset in ("crows_pairs", "indian_bias"):
                for lang in langs:
                    if all((short, dataset, lang, c) in done for c in ("no_rag", "kg_rag")):
                        continue
                    pairs = e4_core.build_pairs(config, dataset, lang, max_pairs=max_pairs)
                    if not pairs:
                        continue
                    res_no, res_kg = {}, {}
                    for cond, use_rag in (("no_rag", False), ("kg_rag", True)):
                        if (short, dataset, lang, cond) in done:
                            continue
                        res = e4_core.evaluate(chat, pairs, graph, use_rag, judge=judge,
                                               dataset=dataset)
                        (res_kg if use_rag else res_no).update(res)
                        append_row(OUT, {
                            "subject_model": short, "dataset": dataset, "language": lang,
                            "condition": cond, "expressed_bias": round(res["expressed_bias"], 4),
                            "deviation": round(abs(res["expressed_bias"] - 50.0), 4),
                            "n_decided": res["n_decided"], "refusals": res["refusals"],
                            "status": "ok"})
                    if res_no and res_kg:
                        gain = (abs(res_no["expressed_bias"] - 50.0)
                                - abs(res_kg["expressed_bias"] - 50.0))
                        print(f"  {short}/{dataset}/{lang}: KG-RAG gain={gain:+.2f}")
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
    print("=== STEP 07  E4 KG-RAG (local decoders) ===")
    run(load_config(), resume=args.resume)
    print(f"  written: {OUT}")


if __name__ == "__main__":
    main()
