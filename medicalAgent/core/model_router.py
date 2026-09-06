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
