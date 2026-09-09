"""
统一配置校验与摘要（启动期检查）

设计原则（spec.md 第 10.6 节，渐进式收敛第一步）：
- 只读 .env / 环境变量做启动校验与摘要输出，不改变各模块的读取逻辑
- 校验失败快速失败：main.py 启动入口调用 validate_config()，
  有错误时以可读清单报错退出，替代运行期的静默回退
- 摘要输出脱敏：API Key 仅显示前 4 位 + ***

用法：
    from core.config import validate_config, print_config_summary

    errors = validate_config()
    if errors:
        for e in errors: print(e); sys.exit(1)
    print_config_summary()
"""
import os
import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from loguru import logger


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _is_true(name: str, default: str = "false") -> bool:
    return _env(name, default).strip().lower() in ("1", "true", "yes", "on")


def _mask_secret(value: str) -> str:
    """脱敏：仅显示前 4 位 + ***；过短或为空时返回占位符"""
    if not value:
        return "(未配置)"
    if len(value) <= 4:
        return "****"
    return f"{value[:4]}***"


# 数值型配置项：环境变量名 -> (说明, 默认值)
_NUMERIC_ENVS = {
    "CIRCUIT_FAILURE_THRESHOLD": ("熔断失败阈值", 3),
    "CIRCUIT_COOLDOWN_SECONDS": ("熔断冷却秒数", 60),
    "ROUTE_SIMPLE_MAX_LEN": ("简单问题长度阈值", 30),
    "ROUTE_LLM_BOUNDARY_MIN": ("路由临界区间下限", 15),
    "ROUTE_LLM_BOUNDARY_MAX": ("路由临界区间上限", 60),
    "KB_EMBEDDING_DIM": ("嵌入向量维度", 512),
    "KB_CHUNK_SIZE": ("知识库分块大小", 500),
    "KB_CHUNK_OVERLAP": ("知识库分块重叠", 50),
    "KB_TOP_K": ("知识库检索条数", 5),
    "REVIEW_REALTIME_TIMEOUT": ("实时审核超时秒数", 15),
    "STM_SESSION_TTL_SECONDS": ("短期记忆过期秒数", 86400),
    "STM_MAX_MESSAGES": ("短期记忆单会话消息上限", 50),
    "STM_MAX_MESSAGE_CHARS": ("短期记忆单条消息字符上限", 32000),
    "STM_COMPRESS_THRESHOLD": ("短期记忆消息压缩阈值(字节)", 8192),
    "REDIS_PORT": ("Redis 端口", 6379),
    "REDIS_DB": ("Redis DB 编号", 0),
    "KB_SEARCH_CACHE_TTL": ("知识库检索缓存秒数", 300),
}

# 布尔型配置项：环境变量名 -> (说明, 默认值)
_BOOL_ENVS = {
    "REVIEW_REALTIME_ENABLED": ("实时专家审核", "false"),
    "ROUTE_LLM_CLASSIFY_ENABLED": ("二级 LLM 路由分类", "false"),
    "RERANK_ENABLED": ("CrossEncoder 重排序", "false"),
    "KB_SEARCH_CACHE_ENABLED": ("知识库检索缓存", "true"),
}

_TRUE_SET = ("1", "true", "yes", "on")
_FALSE_SET = ("0", "false", "no", "off")


def validate_config() -> List[str]:
    """
    启动期统一校验配置。

    Returns:
        错误消息列表（空列表表示校验通过）。
        每条消息为人话描述，可直接打印给用户
    """
    errors: List[str] = []

    # 1. 必填项：LLM API Key（系统核心依赖，缺失时所有功能不可用）
    if not _env("LLM_API_KEY"):
        errors.append("LLM_API_KEY 未配置：请在 .env 中填写大模型 API Key（必填项）")

    # 2. 数值格式
    for name, (label, default) in _NUMERIC_ENVS.items():
        raw = _env(name)
        if not raw:
            continue
        try:
            int(raw)
        except ValueError:
            errors.append(f"{name}={raw!r} 不是有效整数（{label}，默认值 {default}）")

    # 3. 布尔格式
    for name, (label, default) in _BOOL_ENVS.items():
        raw = _env(name)
        if not raw:
            continue
        if raw.lower() not in (set(_TRUE_SET) | set(_FALSE_SET)):
            errors.append(
                f"{name}={raw!r} 不是有效布尔值（{label}，可选 true/false，默认 {default}）"
            )

    # 4. 逻辑冲突
    if _is_true("ROUTE_LLM_CLASSIFY_ENABLED") and not _env("LLM_SMALL_MODEL_NAME"):
        errors.append(
            "ROUTE_LLM_CLASSIFY_ENABLED=true 但 LLM_SMALL_MODEL_NAME 为空："
            "二级路由分类需要配置小模型（或关闭该开关）"
        )
    if _is_true("RERANK_ENABLED") and _env("RERANK_CANDIDATE_K"):
        try:
            if int(_env("RERANK_CANDIDATE_K")) <= 0:
                errors.append("RERANK_CANDIDATE_K 必须为正整数（重排序候选数量）")
        except ValueError:
            pass  # 已由数值校验覆盖（未列入 _NUMERIC_ENVS，这里显式补充）

    # 4.1 短期记忆后端（memory / redis）
    memory_backend = _env("MEMORY_BACKEND", "memory").lower()
    if memory_backend and memory_backend not in ("memory", "redis"):
        errors.append(
            f"MEMORY_BACKEND={memory_backend!r} 无效（短期记忆后端，可选 memory/redis，默认 memory）"
        )
    if memory_backend == "redis" and _env("STM_SESSION_TTL_SECONDS"):
        try:
            if int(_env("STM_SESSION_TTL_SECONDS")) < 0:
                errors.append("STM_SESSION_TTL_SECONDS 不能为负数（0=永不过期）")
        except ValueError:
            pass  # 格式错误已由数值校验报告

    # 4.2 Redis 防护参数（浮点范围）
    for name, label, lo, hi in (
        ("STM_TTL_JITTER_RATIO", "短期记忆 TTL 抖动比例", 0.0, 1.0),
        ("REDIS_SOCKET_TIMEOUT", "Redis socket 超时秒数", 0.1, 60.0),
    ):
        raw = _env(name)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{name}={raw!r} 不是有效数字（{label}，范围 [{lo}, {hi}]）")
            continue
        if not (lo <= value <= hi):
            errors.append(f"{name}={raw!r} 超出范围 [{lo}, {hi}]（{label}）")

    # 4.3 检索缓存 TTL 必须为正（0 会导致写入非法的 ex 参数）
    if _env("KB_SEARCH_CACHE_TTL"):
        try:
            if int(_env("KB_SEARCH_CACHE_TTL")) <= 0:
                errors.append("KB_SEARCH_CACHE_TTL 必须为正整数（知识库检索缓存秒数）")
        except ValueError:
            pass  # 格式错误已由数值校验报告

    # 5. 边界合理性
    try:
        low = int(_env("ROUTE_LLM_BOUNDARY_MIN", "15"))
        high = int(_env("ROUTE_LLM_BOUNDARY_MAX", "60"))
        if low >= high:
            errors.append(
                f"ROUTE_LLM_BOUNDARY_MIN({low}) 必须小于 ROUTE_LLM_BOUNDARY_MAX({high})"
            )
    except ValueError:
        pass  # 格式错误已由数值校验报告

    return errors


def build_config_summary() -> str:
    """
    构建脱敏的生效配置摘要（多行文本）。
    仅包含影响运行行为的关键配置项，全部显示实际生效值（含默认值）
    """
    small_model = _env("LLM_SMALL_MODEL_NAME") or "(未配置，小模型路由/降级关闭)"
    lines = [
        "生效配置摘要（脱敏）:",
        f"  主模型:          {_env('LLM_MODEL_NAME', 'gpt-4o-mini') or '(默认 gpt-4o-mini)'}"
        f"  @ {_env('LLM_BASE_URL', 'https://api.openai.com/v1')}",
        f"  主模型 Key:      {_mask_secret(_env('LLM_API_KEY'))}",
        f"  小模型:          {small_model}",
        f"  小模型 Key:      {_mask_secret(_env('LLM_SMALL_API_KEY')) if _env('LLM_SMALL_API_KEY') else '(复用主 Key)'}",
        f"  熔断器:          阈值 {_env('CIRCUIT_FAILURE_THRESHOLD', '3')} 次 / "
        f"冷却 {_env('CIRCUIT_COOLDOWN_SECONDS', '60')} 秒",
        f"  简单问题路由:    长度阈值 {_env('ROUTE_SIMPLE_MAX_LEN', '30')}，"
        f"二级分类 {'开启' if _is_true('ROUTE_LLM_CLASSIFY_ENABLED') else '关闭'}",
        f"  实时专家审核:    {'开启' if _is_true('REVIEW_REALTIME_ENABLED', 'false') else '关闭'}，"
        f"超时 {_env('REVIEW_REALTIME_TIMEOUT', '15')} 秒",
        f"  短期记忆后端:    {_env('MEMORY_BACKEND', 'memory') or 'memory'}，"
        f"会话 TTL {_env('STM_SESSION_TTL_SECONDS', '86400')} 秒（仅 redis 生效，0=永不过期）",
        f"  Redis 防护:       TTL 抖动 {_env('STM_TTL_JITTER_RATIO', '0.1')}，"
        f"单会话上限 {_env('STM_MAX_MESSAGES', '50')} 条 / "
        f"单条 {_env('STM_MAX_MESSAGE_CHARS', '32000')} 字符",
        f"  检索缓存:        {'开启' if _is_true('KB_SEARCH_CACHE_ENABLED', 'true') else '关闭'}，"
        f"TTL {_env('KB_SEARCH_CACHE_TTL', '300')} 秒（Redis 不可用时自动直通）",
        f"  重排序:          {'开启' if _is_true('RERANK_ENABLED') else '关闭'}",
        f"  嵌入模型:        {_env('KB_EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')}",
    ]
    return "\n".join(lines)


def print_config_summary():
    """打印脱敏配置摘要到日志（INFO 级别）"""
    logger.info(build_config_summary())
