"""Paper 6 single entry point - runs the numbered pipeline in execution order.

The execution order is the numeric SUFFIX on each folder name (Common_00 is the
shared library, imported by every step, not an execution step):
  01  Dataset_Prep_01/check_data.py     integrity + dedup + corruption check   [CPU]
  02  Dataset_Prep_02/build_kg.py       E1: build the MS-SKG                    [CPU]
  03  CPU_Only_03/dry_run.py            test every provider key + model id      [CPU]
  04  GPU_Only_04/e3_bridge.py          E3: KG-bridged transfer (headline)      [GPU]
  05  GPU_Only_05/e2_cda.py             E2: KG-guided counterfactual augment    [GPU]
  06  CPU_Only_06/e4_kgrag_api.py       E4: KG-RAG, API subjects                [CPU/API]
  07  GPU_Only_07/e4_kgrag_local.py     E4: KG-RAG, local decoders              [GPU]
  08  CPU_Only_08/e5_propagation.py     E5: graph propagation analysis          [CPU]

Resume-capable: every step skips already-written rows, so re-running continues
where a crash left off. Reuses Papers 2-5 result CSVs as baselines (no recompute).

Usage:
  python run_all.py                      # all steps in order
  python run_all.py --experiment e3      # only the E3 step (plus its prerequisites 01,02)
  python run_all.py --cpu-only           # skip GPU steps (04,05,07)
  python run_all.py --from 04            # resume from step 04 onward
  python run_all.py --no-resume          # recompute from scratch
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (order, folder/script, kind, experiment-tag, needs_api)
#   kind     : cpu | gpu        (drives --cpu-only / --gpu-only filtering)
#   tag      : prep | e1..e5    ('prep' steps are infrastructure, not experiments)
#   needs_api: True if the step calls a judge/subject provider (gates the dry run)
STEPS = [
    ("01", "Dataset_Prep_01/check_data.py", "cpu", "prep", False),
    ("02", "Dataset_Prep_02/build_kg.py", "cpu", "e1", False),
    ("03", "CPU_Only_03/dry_run.py", "cpu", "prep", False),
    ("04", "GPU_Only_04/e3_bridge.py", "gpu", "e3", False),
    ("05", "GPU_Only_05/e2_cda.py", "gpu", "e2", False),
    ("06", "CPU_Only_06/e4_kgrag_api.py", "cpu", "e4", True),
    ("07", "GPU_Only_07/e4_kgrag_local.py", "gpu", "e4", True),
    ("08", "CPU_Only_08/e5_propagation.py", "cpu", "e5", False),
]
# 01 (data check) + 02 (KG build) precede every experiment; 03 (dry run) precedes
# only the API-using steps. These are added automatically to any plan.
DATA_KG_PREREQ = {"01", "02"}
DRYRUN = "03"
# steps whose script accepts --resume/--no-resume (the others have nothing to resume)
RESUMABLE = {"04", "05", "06", "07"}


def reuse_audit():
    sys.path.insert(0, HERE)
    from Common_00.common import load_config
    from Common_00 import reuse
    print("--- reuse audit (Papers 2-5 baselines; loaded, not recomputed) ---")
    for name, info in reuse.reuse_audit(load_config()).items():
        flag = "OK " if info["found"] else "MISSING"
        print(f"  [{flag}] {name:42s} rows={info['rows']}")
    print("------------------------------------------------------------------")


def select(args):
    # 1) pick the experiment target steps (exclude the 'prep' infrastructure steps).
    targets = []
    for order, script, kind, tag, needs_api in STEPS:
        if tag == "prep":
            continue
        if args.cpu_only and kind == "gpu":
            continue
        if args.gpu_only and kind != "gpu":
            continue
        if args.experiment != "all" and tag != args.experiment:
            continue
        targets.append(order)

    # 2) add prerequisites: data check + KG build always; dry run only if a target
    #    needs an API/judge call.
    needed = set(targets) | set(DATA_KG_PREREQ)
    if any(napi for o, _, _, _, napi in STEPS if o in targets):
        needed.add(DRYRUN)

    # 3) materialise in execution order, honouring --from.
    out = []
    for order, script, kind, _, _ in STEPS:
        if order in needed and not (args.from_step and order < args.from_step):
            out.append((order, script, kind))
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper 6 pipeline runner.")
    ap.add_argument("--experiment", default="all",
                    choices=["all", "e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--cpu-only", action="store_true", help="skip GPU steps (04,05,07)")
    ap.add_argument("--gpu-only", action="store_true", help="run only GPU steps (+ prereqs)")
    ap.add_argument("--from", dest="from_step", default="", help="start at this step, e.g. 04")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    reuse_audit()
    steps = select(args)
    print(f"Plan: {', '.join(o for o, _, _ in steps)}")
    for order, script, kind in steps:
        path = os.path.join(HERE, script)
        cmd = [sys.executable, path]
        if not args.resume and order in RESUMABLE:
            cmd.append("--no-resume")
        print(f"\n===== STEP {order}  ({kind})  {script} =====")
        rc = subprocess.call(cmd, cwd=HERE)
        if rc != 0:
            print(f"STEP {order} failed (exit {rc}). Fix and re-run; completed steps are "
                  f"checkpointed and will be skipped.")
            sys.exit(rc)
    print("\nAll selected steps complete.")


if __name__ == "__main__":
    main()
