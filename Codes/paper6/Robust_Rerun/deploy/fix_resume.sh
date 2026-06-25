#!/usr/bin/env bash
# One-off fix: restore the correct calibrated n_facts (=2), drop any rag_api rows
# written after the restart with the wrong n_facts, then resume step 09 + step 10
# DIRECTLY (not via run_final) so step 07 cannot overwrite calibration.json again.
set -uo pipefail
R=/root/repo/Codes/paper6/Robust_Rerun/results
RR=/root/repo/Codes/paper6/Robust_Rerun

tmux kill-session -t run 2>/dev/null || true
sleep 1

# keep header + the first 81 verified rows (deepseek 36 + llama 36 + gpt-oss 9, all n_facts=2)
head -82 "$R/rag_api.csv" > "$R/rag_api.tmp" && mv "$R/rag_api.tmp" "$R/rag_api.csv"

# restore the correct calibration result
printf '%s' '{"recommended_n_facts": 2, "mean_deviation_by_n_facts": {"1": 12.5801, "2": 10.0393, "3": 15.5268, "4": 15.6717}}' > "$R/calibration.json"

rm -f "$R/PIPELINE_DONE"

# resume step 09 (uses deepseek extractor + n_facts=2), then step 10, then flag done
tmux new-session -d -s run "cd $RR && python step_09_rag_api.py 2>&1 | tee -a $R/run_all.out && python step_10_analysis.py 2>&1 | tee -a $R/run_all.out && touch $R/PIPELINE_DONE"
sleep 2
echo DONE_FIX
echo "calibration.json:"; cat "$R/calibration.json"; echo
echo "rag_api rows:"; tail -n +2 "$R/rag_api.csv" | wc -l
tmux ls
