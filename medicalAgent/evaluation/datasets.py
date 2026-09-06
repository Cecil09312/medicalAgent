"""
评估数据集定义
定义各维度的测试用例数据结构，并提供从 JSON 文件加载的功能
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from .config import DATA_DIR


@dataclass
class RAGTestCase:
    """RAG 评估测试用例"""
    id: str
    question: str
    expected_answer: str
    expected_contexts: List[str]  # 相关文档片段


@dataclass
class AgentTestCase:
    """Agent 评估测试用例"""
    id: str
    question: str
    expected_tools: List[str]  # 应调用的 Skill 名称
    expected_behavior: str  # 期望行为描述
    expected_answer: Optional[str] = None  # 可选的参考答案


@dataclass
class SwarmTestCase:
    """Swarm 评估测试用例"""
    id: str
    question: str
    expected_subtask_count: int  # 期望分解的子任务数
    expected_agents: List[str]  # 应参与的 Agent ID
    expected_coordination_mode: str  # "single" 或 "swarm"
    expected_behavior: str = ""  # 期望行为描述


def load_rag_test_cases(filename: str = "rag_test_cases.json") -> List[RAGTestCase]:
    """加载 RAG 测试用例"""
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        RAGTestCase(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_contexts=item["expected_contexts"],
        )
        for item in data
    ]


def load_agent_test_cases(filename: str = "agent_test_cases.json") -> List[AgentTestCase]:
    """加载 Agent 测试用例"""
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        AgentTestCase(
            id=item["id"],
            question=item["question"],
            expected_tools=item["expected_tools"],
            expected_behavior=item["expected_behavior"],
            expected_answer=item.get("expected_answer"),
        )
        for item in data
    ]


def load_swarm_test_cases(filename: str = "swarm_test_cases.json") -> List[SwarmTestCase]:
    """加载 Swarm 测试用例"""
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        SwarmTestCase(
            id=item["id"],
            question=item["question"],
            expected_subtask_count=item["expected_subtask_count"],
            expected_agents=item["expected_agents"],
            expected_coordination_mode=item["expected_coordination_mode"],
            expected_behavior=item.get("expected_behavior", ""),
        )
        for item in data
    ]
