"""STEP 04 (CPU/API) - dry run: test EVERY provider, key, and model id.

Pings each judge provider (linkapi primary, deepseek, mistral) once PER KEY
(round-robin keys tested individually), pings each black-box subject model id
(deepseek-chat, llama-3.3-70b, gpt-oss-20b), checks the HUGGINGFACE_TOKEN, and
reports the local torch/CUDA/flash-attention state. No key value is ever printed;
keys are referenced by index only. Writes results/dry_run_report.json.

Usage:  python step_04_dry_run.py
"""
import os
import sys

from lib.paths import cfg
from Common_00.common import get_env, get_keys, resolve, write_json
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
                              max_tokens=jcfg.get("max_tokens", 512),
                              timeout_s=jcfg.get("timeout_s", 90),
                              extra_params=spec.get("extra_params"))
            txt = p.chat("You are a test endpoint.", "Reply with the single word: ok")
            entry["status"] = "ok" if txt else "empty"
            entry["detail"] = (txt or "")[:60]
        except Exception as e:
            entry["detail"] = type(e).__name__ + ": " + str(e)[:160]
        result["keys"].append(entry)
    result["any_ok"] = any(k["status"] == "ok" for k in result["keys"])
    return result


def flash_state():
    info = {"torch": None, "cuda": None, "cuda_available": False, "flash_attn": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:
        info["torch_error"] = type(e).__name__
    try:
        import flash_attn
        info["flash_attn"] = getattr(flash_attn, "__version__", "present")
    except Exception:
        info["flash_attn"] = "not_installed"
    return info


def main():
    config = cfg()
    jcfg = config["judge"]
    print("=== STEP 04  dry run (providers / keys / model ids) ===")

    report = {"providers": [], "active_judge": jcfg["provider"], "local_models": []}
    for name, spec in jcfg["providers"].items():
        r = ping_provider(name, spec, jcfg)
        report["providers"].append(r)
        ok = sum(1 for k in r["keys"] if k["status"] == "ok")
        print(f"  {name:9s} model={str(r['model']):<22s} keys_ok={ok}/{r['n_keys_found']}")
        for k in r["keys"]:
            if k["status"] != "ok":
                print(f"      key#{k['key_index']} {k['status']}: {k['detail']}")

    report["api_subject_models"] = []
    print("  API subject models:")
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
                entry["detail"] = (txt or "")[:60]
            except Exception as e:
                entry["detail"] = type(e).__name__ + ": " + str(e)[:160]
        report["api_subject_models"].append(entry)
        print(f"      {short:16s} model={str(entry['model']):37s} -> {entry['status']}")

    for grp in ("encoders", "llms"):
        for m in config["white_box"][grp]:
            report["local_models"].append({"group": grp, "hf_id": m["hf_id"], "short": m["short"]})
    report["huggingface_token_present"] = bool(get_env("HUGGINGFACE_TOKEN"))
    report["flash_attention"] = flash_state()
    print(f"  local models: {len(report['local_models'])} | "
          f"HF token present: {report['huggingface_token_present']} | "
          f"flash_attn: {report['flash_attention']['flash_attn']}")

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
