#!/usr/bin/env bash
# SwAV-REINFORCE DDP 训练启动脚本（sinkhorn reward 版）
# 用法：bash scripts/run_reinforce_ddp.sh [--config <yaml>]
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/yaner/kris/torch/bin/python"
CONFIG="${1:-${REPO}/configs/vlm/swav_reinforce_ddp.yaml}"
LOG_DIR="${REPO}/logs"
mkdir -p "$LOG_DIR"

LOG_NAME="$(basename "${CONFIG%.yaml}").log"

export CUDA_VISIBLE_DEVICES="0,1"

echo "[run_reinforce_ddp] GPUs : $CUDA_VISIBLE_DEVICES"
echo "[run_reinforce_ddp] config: $CONFIG"
echo "[run_reinforce_ddp] log  : $LOG_DIR/$LOG_NAME"

cd "$REPO"
exec "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    --master_port=29501 \
    -m vlc.train.swav_reinforce_ddp_trainer \
    --config "$CONFIG" 2>&1 | tee "$LOG_DIR/$LOG_NAME"
