#!/bin/bash
# Fixed auto-checkpoint: git add -A never aborts on a missing path (the old loop
# listed GPU_DONE.txt explicitly, which did not exist, so git add failed every cycle
# and nothing was ever pushed). Pushes results+logs to GitHub every 10 minutes.
cd /workspace/repo || exit 1
while true; do
  sleep 600
  git add -A
  git -c user.email=gpu@vast.ai -c user.name=vast-gpu commit -q -m "checkpoint $(date -u +%FT%TZ)" >/dev/null 2>&1 && git push origin main >/dev/null 2>&1
done
