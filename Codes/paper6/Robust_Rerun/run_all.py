"""Robust_Rerun single entry point.

Runs the leakage-free pipeline in order, resume-capable. Step 01 (integrity)
runs on EVERY invocation and stops the pipeline on any duplicate/corrupted data
before any later step executes.

  python run_all.py                  # everything in order (resume)
  python run_all.py --from 05        # resume from step 05
  python run_all.py --only 08 09     # just these steps (plus mandatory 01)
  python run_all.py --cpu-only       # 01,02,03,06,10
  python run_all.py --gpu-only       # 01,02,03,05,08 (+ prereqs)
  python run_all.py --api-only       # 01,02,03,04,07,09
  python run_all.py --no-resume      # recompute (passed to steps that support it)

Hardware tags: cpu (local), gpu (24 GB), api (network). The GPU steps need
flash-attention for speed: run install_flash_attention.py once on the GPU VM first.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (num, filename, hardware, supports_resume)
STEPS = [
    ("01", "step_01_integrity.py",     "cpu", False),
    ("02", "step_02_split.py",         "cpu", False),
    ("03", "step_03_build_kg.py",      "cpu", False),
    ("04", "step_04_dry_run.py",       "api", False),
    ("05", "step_05_cut_stability.py", "gpu", True),
    ("06", "step_06_propagation.py",   "cpu", False),
    ("07", "step_07_calibrate.py",     "api", True),
    ("08", "step_08_rag_local.py",     "gpu", True),
    ("09", "step_09_rag_api.py",       "api", True),
    ("10", "step_10_analysis.py",      "cpu", False),
]
PREREQ = {"01", "02", "03"}   # data + split + graph are needed by everything


def select(args):
    nums = {s[0] for s in STEPS}
    if args.only:
        chosen = set(args.only) | PREREQ
    elif args.cpu_only:
        chosen = {n for n, _, hw, _ in STEPS if hw == "cpu"}
    elif args.gpu_only:
        chosen = {n for n, _, hw, _ in STEPS if hw == "gpu"} | PREREQ
    elif args.api_only:
        chosen = {n for n, _, hw, _ in STEPS if hw == "api"} | PREREQ
    else:
        chosen = set(nums)
    if args.start:
        chosen = {n for n in chosen if n >= args.start} | PREREQ
    return [s for s in STEPS if s[0] in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, help="resume from step NN")
    ap.add_argument("--only", nargs="*", default=None, help="run only these step numbers")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--api-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    plan = select(args)
    print("=== Robust_Rerun plan ===")
    for n, fn, hw, _ in plan:
        print(f"  {n} [{hw}] {fn}")
    for n, fn, hw, supports_resume in plan:
        cmd = [sys.executable, os.path.join(HERE, fn)]
        if supports_resume and args.no_resume:
            cmd.append("--no-resume")
        print(f"\n>>> step {n} ({hw}): {fn}")
        rc = subprocess.call(cmd)
        if rc != 0:
            if n in PREREQ or n == "01":
                print(f"  step {n} failed (rc={rc}); stopping (prerequisite).")
                sys.exit(rc)
            print(f"  step {n} failed (rc={rc}); continuing with the rest.")
        if n == "03":
            # Hard leakage gate: prove on the freshly built graph that no test/val
            # group is in it and retrieval cannot return a query's own answer.
            print("\n>>> leakage gate: tests/test_leakage.py")
            grc = subprocess.call([sys.executable, os.path.join(HERE, "tests", "test_leakage.py")])
            if grc != 0:
                print(f"  LEAKAGE GATE FAILED (rc={grc}); stopping.")
                sys.exit(grc)
    print("\n=== Robust_Rerun complete ===")


if __name__ == "__main__":
    main()
