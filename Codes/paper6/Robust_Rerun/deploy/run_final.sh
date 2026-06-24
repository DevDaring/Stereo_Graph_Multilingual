#!/usr/bin/env bash
# Final ordered pipeline for the GPU VM. GPU-bound steps (05, 08) run FIRST so the
# expensive results are computed and pushed before the long API steps (07, 09),
# protecting against any mid-run credit exhaustion. Resume-capable throughout.
cd /root/repo/Codes/paper6/Robust_Rerun || exit 1
mkdir -p results
LOG=results/run_all.out
{
  echo "=== START $(date -u) ==="
  python step_01_integrity.py || { echo "INTEGRITY FAILED"; exit 2; }
  python step_02_split.py
  python step_03_build_kg.py
  python tests/test_leakage.py || { echo "LEAKAGE GATE FAILED"; exit 3; }
  python step_04_dry_run.py || echo "dry run reported issues; continuing"
  echo "=== [GPU] step 05 cut_stability $(date -u) ==="; python step_05_cut_stability.py
  echo "=== [GPU] step 08 rag_local   $(date -u) ==="; python step_08_rag_local.py
  echo "=== [CPU] step 06 propagation $(date -u) ==="; python step_06_propagation.py
  echo "=== [API] step 07 calibrate   $(date -u) ==="; python step_07_calibrate.py
  echo "=== [API] step 09 rag_api     $(date -u) ==="; python step_09_rag_api.py
  echo "=== [CPU] step 10 analysis    $(date -u) ==="; python step_10_analysis.py
  echo "=== DONE $(date -u) ==="
  touch results/PIPELINE_DONE
} 2>&1 | tee -a "$LOG"
