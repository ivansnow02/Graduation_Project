#!/usr/bin/env bash
# =============================================================
# Benchmark 流水线脚本
# 用法:
#   ./scripts/run_benchmark.sh prepare <experiment_name> [model]
#   ./scripts/run_benchmark.sh eval <experiment_name>
#   ./scripts/run_benchmark.sh all <experiment_name> [model]
#
# 示例 (dpow1):
#   # 阶段1: 生成 batch 请求文件
#   ./scripts/run_benchmark.sh prepare dpow1
#
#   # (手动) 上传 data/batch/dpow1/batch.jsonl，等待处理完成后
#   #        下载结果到 data/batchoutput/dpow1/
#
#   # 阶段2: 合并 + 评测
#   ./scripts/run_benchmark.sh eval dpow1
#
#   # 一次性全跑 (prepare → 暂停等待手动操作 → eval)
#   ./scripts/run_benchmark.sh all dpow1
# =============================================================

set -euo pipefail

# ─── 参数解析 ───────────────────────────────────────────────
ACTION="${1:?用法: $0 <prepare|eval|all> <experiment_name> [model]}"
EXPERIMENT="${2:?请指定实验名称，例如: dpow1}"
MODEL="${3:-qwen-flash}"

# ─── 路径定义 ───────────────────────────────────────────────
DIALOG_DIR="data/dialog/${EXPERIMENT}"
BATCH_DIR="data/batch/${EXPERIMENT}"
BATCH_FILE="${BATCH_DIR}/batch.jsonl"
BATCH_OUTPUT_DIR="data/batchoutput/${EXPERIMENT}"
FINAL_DIR="data/final/${EXPERIMENT}"
METRICS_DIR="data/metrics/${EXPERIMENT}_per_file_metrics"
METRICS_FILE="data/metrics/${EXPERIMENT}.json"

# ─── 阶段1: prepare ────────────────────────────────────────
do_prepare() {
    echo "═══════════════════════════════════════════════════"
    echo "  [阶段1] Prepare: ${EXPERIMENT}"
    echo "═══════════════════════════════════════════════════"

    if [ ! -d "${DIALOG_DIR}" ]; then
        echo "❌ 错误: 对话数据目录不存在: ${DIALOG_DIR}"
        exit 1
    fi

    mkdir -p "${BATCH_DIR}"
    uv run scripts/benchmark/annotation_batch.py prepare \
        -i "${DIALOG_DIR}" \
        -m "${MODEL}" \
        -o "${BATCH_FILE}"

    echo ""
    echo "✅ Batch 请求文件已生成:"
    echo "   📁 ${BATCH_FILE}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📤 请手动上传 ${BATCH_FILE}"
    echo "  📥 处理完成后，将结果下载到 ${BATCH_OUTPUT_DIR}/"
    echo "  🔄 然后运行: $0 eval ${EXPERIMENT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ─── 阶段2: merge + eval ───────────────────────────────────
do_eval() {
    echo "═══════════════════════════════════════════════════"
    echo "  [阶段2] Merge + Eval: ${EXPERIMENT}"
    echo "═══════════════════════════════════════════════════"

    # 检查 batchoutput 是否存在
    if [ ! -d "${BATCH_OUTPUT_DIR}" ] || [ -z "$(ls -A ${BATCH_OUTPUT_DIR}/*.jsonl 2>/dev/null)" ]; then
        echo "❌ 错误: Batch 输出目录为空或不存在: ${BATCH_OUTPUT_DIR}"
        echo "   请先上传 batch 并下载结果到该目录。"
        exit 1
    fi

    # Merge
    echo "🔀 合并标注结果..."
    mkdir -p "${FINAL_DIR}"
    uv run scripts/benchmark/annotation_batch.py merge \
        -i "${DIALOG_DIR}/" \
        -b "${BATCH_OUTPUT_DIR}"/*.jsonl \
        -o "${FINAL_DIR}/"

    # Eval
    echo "📊 运行客观评测..."
    mkdir -p "${METRICS_DIR}"
    uv run scripts/benchmark/objective_eval.py \
        "${FINAL_DIR}"/*.jsonl \
        --summary-file "${METRICS_FILE}" \
        --output-dir "${METRICS_DIR}"

    echo ""
    echo "✅ 评测完成!"
    echo "   📄 汇总指标: ${METRICS_FILE}"
    echo "   📁 逐文件指标: ${METRICS_DIR}/"
}

# ─── all: 两阶段合一 ───────────────────────────────────────
do_all() {
    do_prepare

    echo ""
    read -p "⏸️  完成手动上传/下载后，按 Enter 继续评测..." _
    echo ""

    do_eval
}

# ─── 主逻辑 ─────────────────────────────────────────────────
case "${ACTION}" in
    prepare)
        do_prepare
        ;;
    eval)
        do_eval
        ;;
    all)
        do_all
        ;;
    *)
        echo "❌ 未知操作: ${ACTION}"
        echo "用法: $0 <prepare|eval|all> <experiment_name> [model]"
        exit 1
        ;;
esac
