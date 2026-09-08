"""
编排器工厂
构建 LangGraph 编排器（orchestrator/coordinator.LangGraphOrchestrator），
供 web/chat_tab、main.py CLI、评估模块等统一入口使用，
并保证 Worker Agent 集合按统一方式组装。
"""
from typing import Any, Dict, Optional

from loguru import logger


def build_default_worker_agents(llm_client) -> Dict[str, Any]:
    """创建 3 个默认 Worker Agent(consultation/diagnostic/research)"""
    from agents.consultation_agent import ConsultationAgent
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.research_agent import ResearchAgent

    return {
        "consultation_agent": ConsultationAgent(llm_client=llm_client),
        "diagnostic_agent": DiagnosticAgent(llm_client=llm_client),
        "research_agent": ResearchAgent(llm_client=llm_client),
    }


def build_orchestrator(
    llm_client: Optional[Any] = None,
    short_term_memory: Optional[Any] = None,
    long_term_memory: Optional[Any] = None,
    worker_agents: Optional[Dict[str, Any]] = None,
):
    """
    构建编排器

    Args:
        llm_client: LLM 客户端(缺省时创建)
        short_term_memory: 短期记忆管理器
        long_term_memory: 长期记忆管理器
        worker_agents: Worker Agent 字典(缺省时创建 3 个默认 Worker)

    Returns:
        LangGraphOrchestrator 实例
    """
    from .coordinator import LangGraphOrchestrator

    if llm_client is None:
        from core.llm_client import LLMClient
        llm_client = LLMClient()

    worker_agents = worker_agents or build_default_worker_agents(llm_client)
    logger.info("编排框架: langgraph (LangGraphOrchestrator)")
    return LangGraphOrchestrator(
        worker_agents=worker_agents,
        llm_client=llm_client,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory,
    )
