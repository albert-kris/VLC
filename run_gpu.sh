#!/bin/bash
# ============================================================
#  VLC GPU 启动脚本
#  用法：bash run_gpu.sh [阶段]
#  阶段：setup | preextract | smoke | overfit | train | eval
#  默认不带参数时：按顺序执行 preextract → train
# ============================================================

set -e

# ─────────── 路径配置（按需修改）───────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${REPO_DIR}/data"
SHAPES3D_H5="${DATA_DIR}/3dshapes.h5"           # 3DShapes HDF5 文件路径
PYTHON="${REPO_DIR}/.venv/bin/python"            # venv python；没有 venv 就改成 python3
TRAIN_CONFIG="configs/vlm/shapes3d_sft.yaml"    # 训练配置文件

# ─────────── 颜色输出 ──────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

# ─────────── 各阶段函数 ────────────────────────────────────

phase_setup() {
    info "==== [setup] 安装依赖 ===="
    # 建 venv（如果不存在）
    if [ ! -f "${PYTHON}" ]; then
        python3 -m venv .venv
        info "venv 创建完成"
    fi
    "${PYTHON}" -m pip install --upgrade pip -q
    "${PYTHON}" -m pip install -r requirements.txt -q
    info "依赖安装完成"

    # 检查 GPU
    "${PYTHON}" -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3}GB')" \
        || warn "未检测到 CUDA GPU，将使用 CPU（训练会极慢）"
}

phase_preextract() {
    info "==== [preextract] 预提取 3DShapes 图像到 .npy ===="
    if [ ! -f "artifacts/shapes3d/images_train.npy" ]; then
        [ -f "${SHAPES3D_H5}" ] || error "找不到 3DShapes HDF5: ${SHAPES3D_H5}，请先下载"
        "${PYTHON}" scripts/preextract_shapes3d.py --h5 "${SHAPES3D_H5}" --out artifacts/shapes3d
        info "预提取完成 → artifacts/shapes3d/"
    else
        info "已存在预提取文件，跳过"
    fi
}

phase_smoke() {
    info "==== [smoke] Smoke test：检查 VLM 加载 + LoRA + loss ===="
    "${PYTHON}" scripts/smoke_test_sft.py
    info "Smoke test 通过"
}

phase_overfit() {
    info "==== [overfit] 过拟合验证（5 episodes × 30 epoch，~5min）===="
    "${PYTHON}" scripts/overfit_check.py \
        --epochs 30 --episodes 5 --lr 5e-4
    info "过拟合验证完成（loss 应降至 < 0.05 说明管线正常）"
}

phase_train() {
    info "==== [train] 正式训练（~70h on 40GB A100）===="
    info "配置文件: ${TRAIN_CONFIG}"
    info "输出目录: artifacts/vlm/shapes3d/"
    info "日志实时写入: artifacts/vlm/shapes3d/train.log"

    mkdir -p artifacts/vlm/shapes3d
    "${PYTHON}" -m vlc train-sft --config "${TRAIN_CONFIG}" \
        2>&1 | tee artifacts/vlm/shapes3d/train.log

    info "训练完成。最佳 checkpoint: artifacts/vlm/shapes3d/best_lora/"
}

phase_eval() {
    info "==== [eval] 零样本迁移评测 ===="
    "${PYTHON}" -m vlc eval --config configs/vlm/eval_shapes3d_full.yaml \
        2>&1 | tee artifacts/vlm/shapes3d/eval.log

    info "评测完成。结果: artifacts/vlm/shapes3d/eval_results.json"
    info "导出论文表格:"
    "${PYTHON}" -m vlc export-tables --results artifacts/vlm/shapes3d/eval_results.json
}

# ─────────── 参数解析 ──────────────────────────────────────
STAGE="${1:-all}"

case "${STAGE}" in
    setup)      phase_setup ;;
    preextract) phase_preextract ;;
    smoke)      phase_smoke ;;
    overfit)    phase_overfit ;;
    train)      phase_train ;;
    eval)       phase_eval ;;
    all)
        phase_setup
        phase_preextract
        phase_smoke
        phase_overfit
        phase_train
        ;;
    *)
        echo "用法: bash run_gpu.sh [setup|preextract|smoke|overfit|train|eval|all]"
        echo ""
        echo "  setup       安装依赖，检查 GPU"
        echo "  preextract  将 3DShapes HDF5 预提取为 .npy（已存在则跳过）"
        echo "  smoke       快速验证 VLM 加载 + LoRA + loss 计算（<2min）"
        echo "  overfit     小集过拟合验证，确认模型能学（~5min）"
        echo "  train       正式训练（~70h on 40GB A100）"
        echo "  eval        零样本迁移评测 + 导出论文表格"
        echo "  all         依次执行 setup→preextract→smoke→overfit→train"
        exit 1
        ;;
esac
