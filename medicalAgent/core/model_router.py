"""
模型路由模块（Model Router）

提供请求级的"大模型 / 小模型"路由开关：调用方在请求入口根据问题复杂度
（is_simple_question）或显式设置（set_route / route_scope）决定本次请求
使用主模型还是小模型；深层调用链通过 get_route 读取当前路由值，无需改动
函数签名。

路由取值：
- auto：默认值，由调用方结合 is_simple_question 自行判断。
- small：强制使用小模型（大模型熔断降级等场景）。
- big：强制使用大模型。
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from loguru import logger

# 请求级路由开关：asyncio 子任务自动继承上下文变量，
# 避免改动深层调用链的函数签名
_route_var: ContextVar[str] = ContextVar("llm_model_route", default="auto")

# 路由合法取值
_ROUTE_CHOICES = ("auto", "small", "big")

# 复杂问题关键词：包含任一关键词即认为需要大模型处理
COMPLEX_KEYWORDS = ("研究", "分析", "比较", "鉴别", "原理", "为什么", "机制", "方案", "综述", "设计")

# 简单问题的最大长度阈值（环境变量 ROUTE_SIMPLE_MAX_LEN，默认 30，解析失败用默认值）
_DEFAULT_SIMPLE_MAX_LEN = 30
_raw_max_len = os.getenv("ROUTE_SIMPLE_MAX_LEN", str(_DEFAULT_SIMPLE_MAX_LEN))
try:
    ROUTE_SIMPLE_MAX_LEN: int = int(_raw_max_len)
except (TypeError, ValueError):
    ROUTE_SIMPLE_MAX_LEN = _DEFAULT_SIMPLE_MAX_LEN
    logger.warning(f"环境变量 ROUTE_SIMPLE_MAX_LEN 取值无效：{_raw_max_len!r}，使用默认值 {_DEFAULT_SIMPLE_MAX_LEN}")


# ---------------- 二级路由：临界问题的 LLM 意图分类（可选，默认关闭） ----------------

# 未启用分类 / 分类失败时退化为纯启发式（is_simple_question），行为与现状完全一致
ROUTE_LLM_CLASSIFY_ENABLED = os.getenv("ROUTE_LLM_CLASSIFY_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")

_DEFAULT_BOUNDARY_MIN, _DEFAULT_BOUNDARY_MAX = 15, 60


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"环境变量 {name} 取值无效：{raw!r}，使用默认值 {default}")
        return default


# 临界区间：长度落在 [min, max) 且不含复杂关键词的问题视为"临界"，需二级分类
ROUTE_LLM_BOUNDARY_MIN = _parse_int_env("ROUTE_LLM_BOUNDARY_MIN", _DEFAULT_BOUNDARY_MIN)
ROUTE_LLM_BOUNDARY_MAX = _parse_int_env("ROUTE_LLM_BOUNDARY_MAX", _DEFAULT_BOUNDARY_MAX)

# 分类调用的超时秒数（超时即回退启发式，不阻塞主流程）
_CLASSIFY_TIMEOUT_SECONDS = 5.0


def is_boundary_question(question: str) -> bool:
    """
    判断问题是否处于路由临界区间（长度在 [min, max) 且不含复杂关键词）。

    临界问题既可能是简单问题也可能是复杂问题，启发式把握不大，
    启用二级分类时交给小模型判定。
    """
    text = question.strip()
    if not text:
        return False
    if any(kw in text for kw in COMPLEX_KEYWORDS):
        return False
    return ROUTE_LLM_BOUNDARY_MIN <= len(text) < ROUTE_LLM_BOUNDARY_MAX


async def _classify_by_llm(question: str) -> bool:
    """
    用小模型对临界问题做一次轻量意图分类（simple / complex）。

    直接使用小模型环境变量构建独立客户端，不经过 LLMClient（避免循环依赖，
    也不计入熔断器计数）。未配置小模型、调用失败或超时时抛出/返回异常，
    由调用方回退启发式结果。

    Returns:
        True 表示简单问题
    """
    import asyncio
    from openai import AsyncOpenAI

    small_model = (os.getenv("LLM_SMALL_MODEL_NAME") or "").strip()
    if not small_model:
        raise ValueError("未配置 LLM_SMALL_MODEL_NAME，无法执行二级路由分类")

    client = AsyncOpenAI(
        api_key=(os.getenv("LLM_SMALL_API_KEY") or "").strip() or os.getenv("LLM_API_KEY", ""),
        base_url=(os.getenv("LLM_SMALL_BASE_URL") or "").strip()
        or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    )

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=small_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是问题复杂度分类器。判断用户问题属于简单健康咨询"
                        "（如感冒怎么办、吃什么水果好）还是复杂医学问题"
                        "（需要鉴别诊断、机制解释、方案比较、多症状综合分析）。"
                        '只输出一个词：simple 或 complex'
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=8,
        ),
        timeout=_CLASSIFY_TIMEOUT_SECONDS,
    )

    answer = (response.choices[0].message.content or "").strip().lower()
    if "simple" in answer:
        return True
    if "complex" in answer:
        return False
    raise ValueError(f"分类结果无法解析: {answer!r}")


async def classify_question(question: str) -> bool:
    """
    判断问题是否为简单问题（入口路由判定，is_simple_question 的增强版）。

    分级策略：
    - 第一级：现有启发式（长度 + 复杂关键词）快速判定明确场景，零额外成本
    - 第二级（可选）：启用 ROUTE_LLM_CLASSIFY_ENABLED 且问题处于临界区间时，
      用小模型做一次轻量意图分类；分类失败/超时/未配置小模型时回退第一级结果

    Args:
        question: 用户问题文本

    Returns:
        True 表示简单问题（可路由到小模型）
    """
    heuristic_result = is_simple_question(question)

    if not ROUTE_LLM_CLASSIFY_ENABLED:
        return heuristic_result

    # 明确简单（启发式判简单且不在临界区间）或明确复杂（含复杂关键词/超长/为空）时直接用启发式
    if not is_boundary_question(question):
        logger.debug(f"路由判定（一级启发式）: simple={heuristic_result}")
        return heuristic_result

    # 临界问题：二级 LLM 分类，失败回退启发式
    try:
        llm_result = await _classify_by_llm(question)
        logger.info(f"路由判定（二级 LLM 分类）: simple={llm_result}（启发式结果: {heuristic_result}）")
        return llm_result
    except Exception as e:
        logger.warning(f"二级路由分类失败，回退启发式结果（simple={heuristic_result}）: {e}")
        return heuristic_result


def is_simple_question(question: str) -> bool:
    """
    判断是否为简单问题（简单问题可路由到小模型处理）。

    规则：
    - 去除首尾空白后为空：返回 False（空问题保守走大模型）。
    - 长度 >= ROUTE_SIMPLE_MAX_LEN：返回 False（长问题走大模型）。
    - 包含任一 COMPLEX_KEYWORDS：返回 False（复杂问题走大模型）。
    - 其余情况返回 True。

    Args:
        question: 用户问题文本

    Returns:
        True 表示简单问题，False 表示应使用大模型
    """
    text = question.strip()
    if not text:
        return False
    if len(text) >= ROUTE_SIMPLE_MAX_LEN:
        return False
    for keyword in COMPLEX_KEYWORDS:
        if keyword in text:
            return False
    return True


def set_route(route: str):
    """
    设置当前请求级路由值。

    Args:
        route: 路由取值，仅允许 auto / small / big

    Returns:
        ContextVar.set 返回的 token，用于 reset_route 恢复

    Raises:
        ValueError: 路由取值不合法时抛出
    """
    if route not in _ROUTE_CHOICES:
        raise ValueError(f"路由取值必须是 auto/small/big 之一，当前为：{route!r}")
    return _route_var.set(route)


def reset_route(token) -> None:
    """
    恢复路由上下文到 set_route 之前的状态。

    Args:
        token: set_route 返回的 token
    """
    _route_var.reset(token)


def get_route() -> str:
    """
    获取当前路由值。

    Returns:
        当前路由取值（auto / small / big）
    """
    return _route_var.get()


@contextmanager
def route_scope(route: str) -> Iterator[None]:
    """
    路由作用域上下文管理器：进入时设置路由，退出时自动恢复。

    用法示例：
        with route_scope("small"):
            ...  # 该作用域内 get_route() 返回 "small"

    Args:
        route: 路由取值，仅允许 auto / small / big
    """
    token = set_route(route)
    try:
        yield
    finally:
        reset_route(token)
