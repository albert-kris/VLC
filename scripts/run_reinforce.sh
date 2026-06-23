#!/usr/bin/env bash
# 跑真正的 REINFORCE/PPO 训练器（不改动原 swav_rl_trainer.py）
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/yaner/kris/torch/bin/python"
CONFIG="${REPO}/configs/vlm/swav_reinforce.yaml"
LOG_DIR="${REPO}/logs"
mkdir -p "$LOG_DIR"

# kris 默认用物理 GPU 1；3B+LoRA 单卡 24G 放得下（device_map=auto 会全放到可见卡）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

echo "[run_reinforce] GPU=$CUDA_VISIBLE_DEVICES"
echo "[run_reinforce] config=$CONFIG"
echo "[run_reinforce] log=$LOG_DIR/swav_reinforce.log"

cd "$REPO"
exec "$PYTHON" -m vlc.train.swav_reinforce_trainer --config "$CONFIG" 2>&1 | tee "$LOG_DIR/swav_reinforce.log"
