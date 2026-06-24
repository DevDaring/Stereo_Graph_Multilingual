#!/usr/bin/env bash
# Install pre-compiled flash-attention (trying versions until a matching wheel is
# found for the image's torch), run the dry run, then launch the pipeline and the
# 15-minute push loop in tmux. Idempotent: safe to re-run.
cd /root/repo/Codes/paper6/Robust_Rerun || exit 1

echo "=== pre-compiled flash-attention ==="
for v in 2.7.4.post1 2.7.3 2.6.3 2.8.3; do
  echo "-- trying flash-attn $v --"
  if python install_flash_attention.py --version "$v"; then break; fi
done
python -c 'import flash_attn; print("flash_attn", flash_attn.__version__)' \
  || echo "FLASH-ATTN NOT INSTALLED (pipeline still runs on SDPA, slower)"

echo "=== launch tmux sessions: run + push ==="
chmod +x deploy/run_final.sh deploy/push_loop.sh
tmux kill-session -t run  2>/dev/null
tmux kill-session -t push 2>/dev/null
tmux new-session -d -s push "bash /root/repo/Codes/paper6/Robust_Rerun/deploy/push_loop.sh"
tmux new-session -d -s run  "bash /root/repo/Codes/paper6/Robust_Rerun/deploy/run_final.sh"
sleep 2
echo "=== tmux sessions ==="; tmux ls
echo "LAUNCHED"
