#!/usr/bin/env bash
# Push results + logs to Stereo_Graph_Multilingual every 15 minutes. The clone
# remote already carries the token, so push needs no secret here. .env is
# gitignored and never added. Runs until the instance is destroyed.
cd /root/repo || exit 1
git config user.email "koushikdeb2009@gmail.com"
git config user.name "Koushik Deb"
while true; do
  git add -A Codes/paper6/Robust_Rerun/results Codes/paper6/Code_Progress.html 2>/dev/null
  git add -f Codes/paper6/Robust_Rerun/results/run_all.out 2>/dev/null
  git commit -m "checkpoint: robust_rerun results $(date -u +%FT%TZ)" >/dev/null 2>&1
  git push origin HEAD >/dev/null 2>&1 && echo "pushed $(date -u +%FT%TZ)" || echo "push skipped/failed $(date -u +%FT%TZ)"
  sleep 900
done
