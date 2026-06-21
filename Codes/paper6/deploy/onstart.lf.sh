#!/bin/bash
# Vast.ai onstart - fully autonomous GPU run for Paper 6.
# Injected env vars (via vastai create --env): GHTOK, HUGGINGFACE_TOKEN,
# DEEPSEEK_API_KEY_1, DEEPSEEK_API_KEY_2. Writes its own log to /workspace/onstart.log.
exec > /workspace/onstart.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y git tmux curl

cd /workspace
rm -rf repo
git clone https://${GHTOK}@github.com/DevDaring/Stereo_Graph_Multilingual.git repo
cd repo/Codes/paper6

# --- .env from injected secrets (never committed; .gitignore covers it) ---
cat > .env <<EOF
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
DEEPSEEK_API_KEY_1=${DEEPSEEK_API_KEY_1}
DEEPSEEK_API_KEY_2=${DEEPSEEK_API_KEY_2}
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1
EOF

# --- dependencies: keep the image's torch, install everything else ---
pip install --no-input transformers accelerate bitsandbytes sentencepiece \
    "openai>=1.0" python-dotenv pandas numpy networkx PyYAML requests scikit-learn tqdm

# --- pre-compiled flash-attention (verified wheel exists for both ABIs) ---
python Common_00/install_flash_attention.py --version 2.7.4.post1 2>&1 | tee flash_install.log

# --- git identity + tokenised remote for checkpoint pushes ---
git config user.email "gpu@vast.ai"
git config user.name  "vast-gpu"
git remote set-url origin https://${GHTOK}@github.com/DevDaring/Stereo_Graph_Multilingual.git

# --- 15-minute checkpoint loop: push results + logs so preemption loses nothing ---
cat > /workspace/checkpoint.sh <<'CKPT'
#!/bin/bash
cd /workspace/repo/Codes/paper6
while true; do
  sleep 900
  git add -A results *.log GPU_DONE.txt 2>/dev/null
  git commit -m "checkpoint $(date -u +%FT%TZ)" >/dev/null 2>&1 && git push origin main >/dev/null 2>&1
done
CKPT
chmod +x /workspace/checkpoint.sh
nohup /workspace/checkpoint.sh > /workspace/checkpoint.log 2>&1 &

# --- run the pipeline: prereqs + GPU steps, each independent so one crash cannot
#     block the others (every step is resume-capable). CPU steps 06/08 run off-GPU. ---
cat > /workspace/run_pipeline.sh <<'RUN'
#!/bin/bash
cd /workspace/repo/Codes/paper6
{
  echo "=== pipeline start $(date -u +%FT%TZ) ==="
  nvidia-smi
  python -c "import torch,flash_attn;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'flash_attn',flash_attn.__version__)" || echo "flash_attn import check failed (will fall back to sdpa)"
  python Dataset_Prep_01/check_data.py   || echo "[warn] step 01 failed"
  python Dataset_Prep_02/build_kg.py     || echo "[warn] step 02 failed"
  python CPU_Only_03/dry_run.py          || echo "[warn] step 03 failed"
  python GPU_Only_04/e3_bridge.py        || echo "[warn] step 04 failed"
  python GPU_Only_05/e2_cda.py           || echo "[warn] step 05 failed"
  python GPU_Only_07/e4_kgrag_local.py   || echo "[warn] step 07 failed"
  echo "=== pipeline end $(date -u +%FT%TZ) ==="
} > run.log 2>&1
echo "EXIT=$?" > GPU_DONE.txt
date -u >> GPU_DONE.txt
git add -A results *.log GPU_DONE.txt 2>/dev/null
git commit -m "GPU run finished $(date -u +%FT%TZ)" >/dev/null 2>&1
git push origin main >/dev/null 2>&1
RUN
chmod +x /workspace/run_pipeline.sh
nohup /workspace/run_pipeline.sh > /workspace/run_pipeline.log 2>&1 &
echo "onstart complete $(date -u +%FT%TZ)"
