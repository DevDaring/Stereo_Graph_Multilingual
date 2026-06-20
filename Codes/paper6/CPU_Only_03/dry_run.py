"""STEP 03  (CPU_Only_03) - Dry run: test every API provider, model id, and key.

For each judge/extraction provider (gemini primary, deepseek, mistral, openrouter)
it pings the configured model id once PER KEY (round-robin keys tested individually)
and reports ok/fail. It also reports the local model HF ids and whether
HUGGINGFACE_TOKEN is present. No key value is ever printed; keys are referenced by
index only. Writes results/dry_run_report.json.

Usage:  python CPU_Only_03/dry_run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paper6/

from Common_00.common import get_env, get_keys, load_config, resolve, write_json
from Common_00.providers import JudgeProvider, _provider_base_url
from Common_00.api_subjects import SUBJECTS, build_subject


def ping_provider(name, spec, jcfg):
    base = _provider_base_url(spec)
    model = spec.get("model")
    keys = get_keys(spec.get("key_env_vars", []))
    result = {"provider": name, "model": model, "base_url": base,
              "n_keys_found": len(keys), "key_env_vars": spec.get("key_env_vars", []),
              "keys": []}
    for i, key in enumerate(keys):
        entry = {"key_index": i + 1, "status": "fail", "detail": ""}
        try:
            p = JudgeProvider(name, base, model, [key], temperature=0.0,
                              max_tokens=jcfg.get("max_tokens", 24),
                              timeout_s=jcfg.get("timeout_s", 60),
                              extra_params=spec.get("extra_params"))
            txt = p.chat("You are a test endpoint.", "Reply with the single word: ok")
            entry["status"] = "ok" if txt else "empty"
            entry["detail"] = (txt or "")[:40]
        except Exception as e:
            entry["detail"] = type(e).__name__ + ": " + str(e)[:160]
        result["keys"].append(entry)
    result["any_ok"] = any(k["status"] == "ok" for k in result["keys"])
    return result


def main():
    config = load_config()
    jcfg = config["judge"]
    print("=== STEP 03  dry run (API providers / keys / models) ===")

    report = {"providers": [], "local_models": [], "active_judge": jcfg["provider"]}
    for name, spec in jcfg["providers"].items():
        r = ping_provider(name, spec, jcfg)
        report["providers"].append(r)
        ok = sum(1 for k in r["keys"] if k["status"] == "ok")
        print(f"  {name:11s} model={r['model']:<38s} keys_ok={ok}/{r['n_keys_found']}")
        for k in r["keys"]:
            if k["status"] != "ok":
                print(f"      key#{k['key_index']} {k['status']}: {k['detail']}")

    # E4 API subject models: ping each served model id once (per the requirement to
    # test all model ids in the dry run).
    report["api_subject_models"] = []
    print("  E4 API subject models:")
    for short in config["api_predict_only"]:
        entry = {"short": short, "configured": short in SUBJECTS,
                 "model": SUBJECTS.get(short, {}).get("model"), "status": "fail", "detail": ""}
        subj = build_subject(short) if short in SUBJECTS else None
        if subj is None:
            entry["status"] = "no_keys_or_route"
        else:
            try:
                txt = subj.chat("You are a test endpoint.", "Reply with the single word: ok")
                entry["status"] = "ok" if txt else "empty"
                entry["detail"] = (txt or "")[:40]
            except Exception as e:
                entry["detail"] = type(e).__name__ + ": " + str(e)[:160]
        report["api_subject_models"].append(entry)
        print(f"      {short:16s} model={str(entry['model']):37s} -> {entry['status']}")

    # local subject/encoder model ids (loadability is verified on GPU later)
    for grp in ("encoders", "llms"):
        for m in config["white_box"][grp]:
            report["local_models"].append({"group": grp, "hf_id": m["hf_id"], "short": m["short"]})
    report["huggingface_token_present"] = bool(get_env("HUGGINGFACE_TOKEN"))
    print(f"  local models: {len(report['local_models'])} | "
          f"HF token present: {report['huggingface_token_present']}")

    out = resolve(os.path.join(config["paths"]["results"], "dry_run_report.json"))
    write_json(out, report)
    print(f"  report: {out}")

    active = next((p for p in report["providers"] if p["provider"] == jcfg["provider"]), None)
    if not active or not active["any_ok"]:
        print(f"  WARNING: active judge provider '{jcfg['provider']}' has no working key.")
        sys.exit(1)
    print("  active judge provider OK.")


if __name__ == "__main__":
    main()
