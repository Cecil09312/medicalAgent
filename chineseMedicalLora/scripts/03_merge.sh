#!/usr/bin/env bash
# scripts/03_merge.sh
# 一键合并 LoRA 到 HF 模型
# 用法: bash scripts/03_merge.sh
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

if [ ! -d "outputs/lora_adapter" ]; then
    echo "[ERROR] 找不到 outputs/lora_adapter，请先完成训练"
    exit 1
fi

echo "=========================================="
echo "  Step 3: 合并 LoRA → HF 模型"
echo "=========================================="
python src/merge_lora.py "$@"

echo
echo "=========================================="
echo "  ✓ 合并完成"
echo "=========================================="
ls -lh outputs/merged_model | head -20
