#!/usr/bin/env bash
# scripts/02_train.sh
# 一键启动 SFT 训练（推荐在 AutoDL / 远程 GPU 上后台跑）
# 用法:
#   bash scripts/02_train.sh                # 正式训练
#   bash scripts/02_train.sh --dry-run      # Dry-run 验证流水线（~3 min）
#   bash scripts/02_train.sh --resume-from outputs/lora_adapter/checkpoint-500
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 自动加载 .env（已 export 的变量优先）
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

# Dry-run 短路
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "=========================================="
    echo "  DRY-RUN: 用 200 条数据跑 10 步"
    echo "  基座: ${BASE_MODEL_PATH:-"(未设置)"}"
    echo "=========================================="
    if [ ! -f "data/processed/_dryrun.jsonl" ]; then
        echo "  准备 dryrun 数据 ..."
        python src/build_dataset.py --dryrun 200
    fi
    python src/train.py --dry-run
    exit 0
fi

# 正式训练：检查数据 + 后台跑
if [ ! -f "data/processed/train.jsonl" ]; then
    echo "[ERROR] 找不到 data/processed/train.jsonl，请先跑 bash scripts/01_build_data.sh"
    exit 1
fi

LOG_FILE="outputs/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "=========================================="
echo "  正式训练: 2 万条数据/每科室 × 1 epoch"
echo "  日志: $LOG_FILE"
echo "  基座: ${BASE_MODEL_PATH:-"(未设置)"}"
echo "=========================================="

# 把任意参数透传给 train.py
python src/train.py "$@" 2>&1 | tee "$LOG_FILE"
