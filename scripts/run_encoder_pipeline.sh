#!/usr/bin/env bash
# 两阶段课程训练 pipeline
# Stage 1: CIFAR-10 (从零)  ->  Stage 2: CIFAR-100 (warm-start)

set -e
PYTHON=".venv/bin/python"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export CURL_CA_BUNDLE=""

cd "$(dirname "$0")/.."

log "=== Stage 1: CIFAR-10 开始 ==="
$PYTHON -m vlc train-encoder --config configs/vlm/encoder_cifar10.yaml \
    2>&1 | tee logs/stage1_cifar10.log
log "=== Stage 1 完成 ==="

log "=== Stage 2: CIFAR-100 开始 ==="
$PYTHON -m vlc train-encoder --config configs/vlm/encoder_cifar100.yaml \
    2>&1 | tee logs/stage2_cifar100.log
log "=== Stage 2 完成 ==="

log "全部 pipeline 训练完成！"
log "  Stage 1 best: artifacts/vlm/encoder_cifar10/best/"
log "  Stage 2 best: artifacts/vlm/encoder_cifar100/best/"
