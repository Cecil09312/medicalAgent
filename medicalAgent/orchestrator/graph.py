"""
医疗编排图(LangGraph StateGraph)
替代 swarm/swarm_coordinator.py 的中心化编排流程。

图结构:
    START → load_context → decompose → (Send fan-out) worker
    worker → finalize_single / finalize_failed / synthesize → postprocess → save_memory → END

节点均为薄壳,实际逻辑委托给 LangGraphOrchestrator 的实例方法
(_decompose/_execute_worker/_synthesize/_postprocess 等),
保持与 swarm 时期相同的可测试接缝(测试可直接 monkeypatch 这些方法)。
"""
import re
from typing import Any, Dict, List
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger

from .state import MedicalGraphState


def extract_suggestions(answer: str) -> List[str]:
    """从最终答案提取建议(建议/可以/应该/推荐 模式,最多 10 条)"""
    suggestions: List[str] = []
    suggestion_patterns = [
        r"建议[：:]\s*(.+)",
        r"可以\s*(.+)",
        r"应该\s*(.+)",
        r"推荐\s*(.+)",
    ]
    for pattern in suggestion_patterns:
        matches = re.findall(pattern, answer)
        suggestions.extend(matches[:3])
    return suggestions[:10]


def build_swarm_summary(subtasks: List[Dict[str, Any]], worker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构建协作摘要(子任务数/完成数/贡献数,供结果字段 swarm_summary 使用)"""
    return {
        "total_subtasks": len(subtasks),
        "completed_subtasks": len(worker_results),
        "total_events": 0,
        "total_contributions": len(worker_results),
        "shared_data_keys": [],
    }


def _apply_decomposition_constraints(question: str, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Harness Engineering: 任务分解约束

    - 子任务数超过 max_agents_per_task:确定性裁剪到上限
    - 高风险症状问题未分配诊断 Agent:告警(不强行改分配,与工具约束同策略)
    约束模块不可用或校验异常时原样返回,不影响主流程
    """
    try:
        from constraints import ConstraintValidator

        validator = ConstraintValidator()
        validation = validator.validate_task_decomposition(question, subtasks)
        if not validation.get("valid"):
            for violation in validation.get("violations", []):
                logger.warning(f"任务分解约束违规: {violation}")

            max_agents = validator.swarm_constraints.get("max_agents_per_task", 3)
            if len(subtasks) > max_agents:
                subtasks = subtasks[:max_agents]
                logger.warning(f"子任务数已裁剪至 max_agents_per_task={max_agents}")
    except Exception as e:
        logger.debug(f"validate_task_decomposition failed: {e}")

    return subtasks


def build_medical_graph(orch) -> Any:
    """
    组装并编译医疗编排图

    Args:
        orch: LangGraphOrchestrator 实例,提供各节点委托方法

    Returns:
        编译后的 LangGraph 可执行图
    """

    async def load_context(state: MedicalGraphState) -> Dict[str, Any]:
        memories = await orch._retrieve_memories(state["question"], state.get("session_id"))
        if memories:
            context = state.get("context") or {}
            context["memories"] = memories
            return {"context": context}
        return {}

    async def decompose(state: MedicalGraphState) -> Dict[str, Any]:
        decomposition = await orch._decompose(state["question"], state.get("context"))

        subtasks = decomposition.get("decomposition", [])
        if not subtasks:
            # 分解失败/为空:默认单 Agent 处理(对齐 LeadAgent 默认分解兜底)
            subtasks = [{
                "type": "consultation",
                "description": state["question"],
                "assigned_agent": "consultation_agent",
                "dependencies": [],
            }]

        subtasks = _apply_decomposition_constraints(state["question"], subtasks)

        mode = "swarm" if len(subtasks) >= 2 else "single"
        logger.info(f"Decomposed into {len(subtasks)} subtasks (mode={mode})")
        return {"decomposition": decomposition, "subtasks": subtasks, "mode": mode}

    def route_from_decompose(state: MedicalGraphState) -> List[Send]:
        sends = []
        for subtask in state["subtasks"]:
            branch_state = dict(state)
            branch_state["current_task"] = subtask
            sends.append(Send("worker", branch_state))
        return sends

    async def worker(state: MedicalGraphState) -> Dict[str, Any]:
        task = state["current_task"]
        agent_id = task.get("assigned_agent", "consultation_agent")
        question_input = (
            state["question"] if state.get("mode") == "single" else task.get("description", "")
        )

        try:
            result = await orch._execute_worker(agent_id, question_input, state.get("session_id"))
        except TimeoutError:
            return {"worker_timeout": True}
        except Exception as e:
            logger.error(f"Worker {agent_id} failed: {e}")
            return {}

        result = dict(result)
        result.setdefault("task_id", task.get("task_id"))
        result["task_type"] = task.get("type", "unknown")
        return {"worker_results": [result]}

    def after_worker(state: MedicalGraphState) -> str:
        if state.get("mode") == "single":
            return "finalize_single"
        if not state.get("worker_results"):
            return "finalize_failed"
        return "synthesize"

    def finalize_single(state: MedicalGraphState) -> Dict[str, Any]:
        results = state.get("worker_results") or []
        if results:
            return {"result": results[0]}
        return {"result": {"answer": "没有可用的 Agent", "error": "no_agent_available"}}

    def finalize_failed(state: MedicalGraphState) -> Dict[str, Any]:
        logger.error(
            f"Swarm 所有 Worker 均失败(worker_timeout={state.get('worker_timeout')}),"
            "跳过 LLM 综合合成,直接返回兜底结果"
        )
        return {"result": {
            "answer": "抱歉，本次协作处理未能获得有效结果，请稍后重试或换个问题试试。",
            "suggestions": [],
            "swarm_summary": build_swarm_summary(state.get("subtasks", []), []),
            "timeout_occurred": bool(state.get("worker_timeout")),
            "warning": "all_workers_failed",
        }}

    async def synthesize(state: MedicalGraphState) -> Dict[str, Any]:
        worker_results = state.get("worker_results") or []
        timeout_occurred = bool(state.get("worker_timeout"))

        final_answer = await orch._synthesize(
            state["question"], worker_results, timeout_occurred
        )

        return {"result": {
            "answer": final_answer,
            "suggestions": extract_suggestions(final_answer),
            "swarm_summary": build_swarm_summary(state.get("subtasks", []), worker_results),
            "timeout_occurred": timeout_occurred,
        }}

    async def postprocess(state: MedicalGraphState) -> Dict[str, Any]:
        result = await orch._postprocess(state["question"], state.get("result") or {})
        return {"result": result}

    async def save_memory(state: MedicalGraphState) -> Dict[str, Any]:
        result = state.get("result") or {}
        await orch._save_memories(
            state["question"],
            result.get("answer", ""),
            state.get("session_id"),
            save_stm=state.get("mode") == "swarm",
        )
        return {}

    graph = StateGraph(MedicalGraphState)
    graph.add_node("load_context", load_context)
    graph.add_node("decompose", decompose)
    graph.add_node("worker", worker)
    graph.add_node("finalize_single", finalize_single)
    graph.add_node("finalize_failed", finalize_failed)
    graph.add_node("synthesize", synthesize)
    graph.add_node("postprocess", postprocess)
    graph.add_node("save_memory", save_memory)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "decompose")
    graph.add_conditional_edges("decompose", route_from_decompose, ["worker"])
    graph.add_conditional_edges(
        "worker",
        after_worker,
        ["finalize_single", "finalize_failed", "synthesize"],
    )
    graph.add_edge("finalize_single", "postprocess")
    graph.add_edge("finalize_failed", "postprocess")
    graph.add_edge("synthesize", "postprocess")
    graph.add_edge("postprocess", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
