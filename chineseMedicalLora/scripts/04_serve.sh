#!/usr/bin/env bash
# scripts/04_serve.sh
# 一键启动推理服务（支持前端切换「微调前 / 微调后」）
# 用法:
#   bash scripts/04_serve.sh                # 默认 0.0.0.0:8000
#   PORT=9000 bash scripts/04_serve.sh      # 自定义端口
#   DEVICE=cpu bash scripts/04_serve.sh     # 强制 CPU（训练占 GPU 时）
#   DEVICE=cuda bash scripts/04_serve.sh    # 强制 GPU
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

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-./outputs/merged_model}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-./outputs/lora_adapter}"
DEVICE="${DEVICE:-auto}"

has_base=0
has_merged=0
has_lora=0
[ -n "$BASE_MODEL_PATH" ] && [ -f "$BASE_MODEL_PATH/config.json" ] && has_base=1
[ -f "$MERGED_MODEL_PATH/config.json" ] && has_merged=1
[ -f "$LORA_ADAPTER_PATH/adapter_config.json" ] && has_lora=1

if [ "$has_base" -eq 0 ] && [ "$has_merged" -eq 0 ]; then
    echo "[ERROR] 找不到可用模型:"
    echo "        BASE_MODEL_PATH=${BASE_MODEL_PATH:-"(未设置)"}"
    echo "        MERGED_MODEL_PATH=$MERGED_MODEL_PATH"
    echo "        请配置基座，或先执行 bash scripts/03_merge.sh"
    exit 1
fi

# GPU 已被占用时，auto → cpu，避免与训练抢显存
if [ "$DEVICE" = "auto" ]; then
    gpu_used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)"
    if [ "${gpu_used_mb:-0}" -gt 2000 ] 2>/dev/null; then
        DEVICE="cpu"
        echo "[WARN] GPU 显存已占用约 ${gpu_used_mb}MiB，DEVICE=auto → cpu"
    fi
fi

export BASE_MODEL_PATH MERGED_MODEL_PATH LORA_ADAPTER_PATH DEVICE
echo "=========================================="
echo "  Step 4: 启动推理服务"
echo "=========================================="
echo "  host:   $HOST"
echo "  port:   $PORT"
echo "  device: $DEVICE"
echo "  微调前: $BASE_MODEL_PATH $([ "$has_base" -eq 1 ] && echo '[OK]' || echo '[缺失]')"
if [ "$has_merged" -eq 1 ]; then
    echo "  微调后: $MERGED_MODEL_PATH [合并模型]"
elif [ "$has_lora" -eq 1 ] && [ "$has_base" -eq 1 ]; then
    echo "  微调后: $LORA_ADAPTER_PATH [基座+LoRA]"
else
    echo "  微调后: (暂不可用，训练/合并完成后可切换)"
fi
echo

# workers=1：模型常驻内存/显存，多 worker 会重复加载
exec uvicorn src.serve.app:app --host "$HOST" --port "$PORT" --workers 1
