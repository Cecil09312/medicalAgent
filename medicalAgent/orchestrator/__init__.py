"""
Orchestrator - 基于 LangGraph 的多Agent编排

图结构: 记忆检索 → 任务分解 → Send 并行 Worker(react agent) → 综合 → 后处理
对外契约: orchestrator.process(question, context, session_id) -> 结果字典
"""
from .factory import build_orchestrator

__all__ = ["build_orchestrator"]
