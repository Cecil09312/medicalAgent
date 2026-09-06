#!/usr/bin/env bash
# scripts/01_build_data.sh
# 一键构建训练数据集
# 用法: bash scripts/01_build_data.sh [N_PER_DEPT]
set -euo pipefail

# 项目根目录
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 自动加载 .env（已 export 的变量优先）
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

# 默认参数
N_PER_DEPT="${1:-${N_PER_DEPT:-20000}}"
MAX_CHAR_LEN="${MAX_CHAR_LEN:-600}"
SEED="${SEED:-42}"
RAW_DIR="${RAW_DATA_DIR:-./datasets}"
OUT_DIR="${PROCESSED_DATA_DIR:-./data/processed}"

echo "=========================================="
echo "  Step 1: 构建数据集"
echo "=========================================="
echo "  raw:  $RAW_DIR"
echo "  out:  $OUT_DIR"
echo "  n/dept: $N_PER_DEPT"
echo "  max_char_len: $MAX_CHAR_LEN"
echo "  seed: $SEED"
echo

python src/build_dataset.py \
    --raw "$RAW_DIR" \
    --out "$OUT_DIR" \
    --n-per-dept "$N_PER_DEPT" \
    --max-char-len "$MAX_CHAR_LEN" \
    --seed "$SEED"

echo
echo "=========================================="
echo "  ✓ 数据集已生成"
echo "=========================================="
ls -lh "$OUT_DIR"
