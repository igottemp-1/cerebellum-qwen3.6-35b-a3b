#!/bin/bash
set -e

BINARY_DIR="/tmp/beellama"
MODEL_DIR="/persistent-storage/models"
MAIN_MODEL="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
DFLASH_MODEL="$MODEL_DIR/qwen36-35b-a3b-dflash-Q6_K.gguf"

mkdir -p "$BINARY_DIR" "$MODEL_DIR"

# ── 1. Pull BeeLlama binaries from Kaggle ──────────────────────────────────
echo "[$(date +%T)] Downloading BeeLlama binaries..."
python3 - <<'EOF'
import kagglehub, shutil, os

path = kagglehub.dataset_download("igottempmail/beellama-l4-sw89")
dest = "/tmp/beellama"
for f in os.listdir(path):
    shutil.copy2(os.path.join(path, f), os.path.join(dest, f))
print(f"Copied binaries from {path} to {dest}")
EOF

chmod +x "$BINARY_DIR/llama-server"
export LD_LIBRARY_PATH="$BINARY_DIR:${LD_LIBRARY_PATH:-}"
echo "[$(date +%T)] BeeLlama binaries ready"

# ── 2. Download main model if not cached ───────────────────────────────────
if [ ! -f "$MAIN_MODEL" ]; then
    echo "[$(date +%T)] Main model not found, downloading (21.4 GB)..."
    python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

hf_hub_download(
    repo_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    filename="Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
    local_dir="$MODEL_DIR",
    token=os.environ["HF_TOKEN"],
)
print("Main model downloaded")
EOF
    echo "[$(date +%T)] Main model cached to persistent storage"
else
    echo "[$(date +%T)] Main model found in persistent storage, skipping download"
fi

# ── 3. Download DFlash model if not cached ─────────────────────────────────
if [ ! -f "$DFLASH_MODEL" ]; then
    echo "[$(date +%T)] DFlash model not found, downloading (1 GB)..."
    python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

hf_hub_download(
    repo_id="Anbeeld/Qwen3.6-35B-A3B-DFlash-GGUF",
    filename="qwen36-35b-a3b-dflash-Q6_K.gguf",
    local_dir="$MODEL_DIR",
    token=os.environ["HF_TOKEN"],
)
print("DFlash model downloaded")
EOF
    echo "[$(date +%T)] DFlash model cached to persistent storage"
else
    echo "[$(date +%T)] DFlash model found in persistent storage, skipping download"
fi

# ── 4. Start Cloudflare tunnel ─────────────────────────────────────────────
echo "[$(date +%T)] Starting Cloudflare tunnel..."
cloudflared tunnel --no-autoupdate run \
    --token "$CLOUDFLARE_TUNNEL_TOKEN" &
CF_PID=$!
echo "[$(date +%T)] Cloudflare tunnel PID: $CF_PID"

# ── 5. Start llama-server ──────────────────────────────────────────────────
echo "[$(date +%T)] Starting llama-server..."
exec "$BINARY_DIR/llama-server" \
    --model "$MAIN_MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
    --alias qwen3.6-35b-a3b \
    --api-key "$API_TOKEN" \
    \
    `# GPU offload` \
    -ngl 999 \
    \
    `# Context` \
    --ctx-size 128000 \
    --parallel 1 \
    --kv-unified \
    \
    `# KV cache` \
    --cache-type-k turbo3_tcq \
    --cache-type-v turbo3_tcq \
    --cache-ram 4096 \
    \
    `# Batch` \
    --batch-size 2048 \
    --ubatch-size 512 \
    \
    `# Flash attention` \
    --flash-attn on \
    \
    `# Sampling defaults` \
    --temp 0.3 \
    --top-k 20 \
    --top-p 0.95 \
    --min-p 0.0 \
    --repeat-penalty 1.0 \
    \
    `# Reasoning` \
    --reasoning on \
    --jinja \
    --chat-template-kwargs '{"preserve_thinking":true}' \
    \
    `# DFlash speculative decoding` \
    --spec-type dflash \
    --spec-draft-model "$DFLASH_MODEL" \
    --spec-draft-ngl all \
    --spec-dflash-cross-ctx 1024 \
    --spec-draft-n-max 16 \
    --spec-branch-budget 2 \
    --spec-draft-temp 0 \
    --spec-dm-adaptive \
    \
    `# MTP on top of DFlash` \
    --spec-draft-n-max 2 \
    \
    `# Logging` \
    --no-display-prompt \
    --log-colors off \
    --log-timestamps \
    --log-prefix \
    -v 0
