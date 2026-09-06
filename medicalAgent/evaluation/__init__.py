"""
评估模块
提供 RAG 知识库评估、Agent 能力评估、Swarm 协作评估、端到端性能评估
"""
from .config import EVAL_CONFIG, DATA_DIR
from .datasets import (
    RAGTestCase,
    AgentTestCase,
    SwarmTestCase,
    load_rag_test_cases,
    load_agent_test_cases,
    load_swarm_test_cases,
)
from .rag_eval import RAGEvaluator, RAGEvalResult
from .agent_eval import AgentEvaluator, AgentEvalResult
from .swarm_eval import SwarmEvaluator, SwarmEvalResult
from .performance_eval import PerformanceEvaluator, PerfEvalResult
from .run_eval import run_evaluation

__all__ = [
    "EVAL_CONFIG",
    "DATA_DIR",
    # 数据集
    "RAGTestCase",
    "AgentTestCase",
    "SwarmTestCase",
    "load_rag_test_cases",
    "load_agent_test_cases",
    "load_swarm_test_cases",
    # 评估器
    "RAGEvaluator",
    "RAGEvalResult",
    "AgentEvaluator",
    "AgentEvalResult",
    "SwarmEvaluator",
    "SwarmEvalResult",
    "PerformanceEvaluator",
    "PerfEvalResult",
    # 统一入口
    "run_evaluation",
]
