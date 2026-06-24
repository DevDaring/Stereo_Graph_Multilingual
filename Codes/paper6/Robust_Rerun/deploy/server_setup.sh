#!/usr/bin/env bash
# Robust_Rerun server bootstrap for the GPU VM (Ubuntu, NVIDIA L4 24 GB).
# Installs deps + PRE-COMPILED flash-attention, runs the dry run, then launches
# the full pipeline and a 15-minute GitHub push loop in tmux. No venv (global).
# Secrets are never committed: .env must be scp'd to Codes/paper6/.env first.
#
# Usage (driven over ssh from the workstation):
#   GHTOK=<token> REPO_URL=https://github.com/DevDaring/<repo>.git bash server_setup.sh
set -uo pipefail

WORK="${WORK:-$HOME/robust}"
: "${REPO_URL:?set REPO_URL}"
: "${GHTOK:?set GHTOK}"
REPO_DIR="$WORK/repo"
RR="$REPO_DIR/Codes/paper6/Robust_Rerun"
ENVFILE="$REPO_DIR/Codes/paper6/.env"

echo "=== 1. system packages ==="
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip git tmux

echo "=== 2. clone repo (token in remote only, never committed) ==="
mkdir -p "$WORK"; cd "$WORK"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "https://${GHTOK}@${REPO_URL#https://}" repo
fi
cd "$REPO_DIR"
git remote set-url origin "https://${GHTOK}@${REPO_URL#https://}"
git config user.email "koushikdeb2009@gmail.com"
git config user.name "Koushik Deb"
git pull --rebase origin main || true

echo "=== 3. python deps (global, no venv) ==="
cd "$RR"
pip3 install -r requirements.txt

echo "=== 4. pre-compiled flash-attention (necessity) ==="
python3 install_flash_attention.py || echo "flash-attn: not installed; pipeline still runs on SDPA (slower)."

echo "=== 5. verify .env present ==="
if [ ! -f "$ENVFILE" ]; then
  echo "ERROR: .env missing at $ENVFILE  -> scp it before running the pipeline."; exit 1
fi

echo "=== 6. dry run (verify keys + model ids + flash-attn before GPU spend) ==="
python3 step_04_dry_run.py || { echo "dry run failed; fix before launching GPU."; exit 1; }

echo "=== 7. launch pipeline + 15-min push loop in tmux ==="
cat > "$WORK/push_loop.sh" <<'EOF'
#!/usr/bin/env bash
# Commit + push results every 15 minutes. Logs use .out (not .log) so they are
# not blocked by the repo .gitignore (*.log). .env is gitignored and never added.
REPO="$1"
cd "$REPO" || exit 1
while true; do
  git add -A Codes/paper6/Robust_Rerun/results Codes/paper6/Code_Progress.html 2>/dev/null || true
  git commit -m "checkpoint: robust_rerun results $(date -u +%FT%TZ)" >/dev/null 2>&1 || true
  git push origin HEAD >/dev/null 2>&1 || true
  sleep 900
done
EOF
chmod +x "$WORK/push_loop.sh"

mkdir -p "$RR/results"
tmux kill-session -t push 2>/dev/null || true
tmux kill-session -t run  2>/dev/null || true
tmux new-session -d -s push "bash $WORK/push_loop.sh $REPO_DIR"
tmux new-session -d -s run  "cd $RR && python3 run_all.py 2>&1 | tee $RR/results/run_all.out"

echo "=== launched. tmux sessions: 'run' (pipeline) and 'push' (15-min checkpoint) ==="
echo "watch:  tmux attach -t run     status: nvidia-smi; tail -f $RR/results/run_all.out"
