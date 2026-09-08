"""LangGraph 编排主链路集成测试（mock LLM）：兜底路径与医疗安全硬约束

覆盖：
- 编排行为契约（全失败兜底/系统错误/紧急硬约束改写/不改写/普通透传）
- 单 Agent 跳过综合、多子任务并行派发与综合
- Worker 超时兜底、未知 Agent 回退
- WorkerRunner 端到端（LLM 适配器 + 工具预算 + react 循环）
- contextvar 模型路由在 Send 并行分支中的传播
"""
import operator
from typing import Annotated, TypedDict

import pytest

import orchestrator.coordinator as oc_module
from orchestrator.coordinator import LangGraphOrchestrator
from orchestrator.worker_runner import WorkerRunner
from core.llm_client import LLMResponse, ToolCall
from core.skill_registry import SkillRegistry, SkillParameter


class _StubLLMClient:
    """占位 LLM 客户端：所有 LLM 调用路径均被 monkeypatch，不会真正发起请求"""


@pytest.fixture
def coordinator():
    """空 Worker 集合的编排器：LLM 依赖全部由测试 monkeypatch"""
    orch = LangGraphOrchestrator(worker_agents={}, llm_client=_StubLLMClient())
    yield orch


async def _decompose(monkeypatch, orch, subtasks):
    async def fake_decompose(question, context=None):
        return {"analysis": "测试", "decomposition": subtasks}

    monkeypatch.setattr(orch, "_decompose", fake_decompose)


async def _worker_returning(monkeypatch, orch, result):
    async def fake_worker(agent_id, question_input, session_id):
        return dict(result)

    monkeypatch.setattr(orch, "_execute_worker", fake_worker)


# ========== 编排行为契约 ==========

async def test_all_workers_failed_fallback(monkeypatch, coordinator):
    """全 Worker 失败 → 跳过 LLM 合成，返回结构化兜底结果"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答问题A", "assigned_agent": "consultation_agent"},
        {"type": "研究", "description": "回答问题B", "assigned_agent": "research_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    async def failing_worker(agent_id, question_input, session_id):
        raise RuntimeError("worker down")

    monkeypatch.setattr(coordinator, "_execute_worker", failing_worker)

    result = await coordinator.process("复杂的多方面问题" * 3)

    assert result["warning"] == "all_workers_failed"
    assert "未能获得有效结果" in result["answer"]


async def test_process_unified_exception_fallback(monkeypatch, coordinator):
    """主流程未预期异常 → 返回 system_error 结构化结果，不裸抛"""

    async def broken_retrieve(question, session_id):
        raise RuntimeError("未预期异常")

    monkeypatch.setattr(coordinator, "_retrieve_memories", broken_retrieve)
    result = await coordinator.process("任何问题")

    assert result["warning"] == "system_error"
    assert "稍后重试" in result["answer"]
    assert "未预期异常" in result["error"]


async def test_emergency_hard_constraint_rewrites_answer(monkeypatch, coordinator):
    """高危问题 + 回答无就医引导 → 强制改写，原文保留在 reference_answer"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答", "assigned_agent": "consultation_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    original = "您可以先在家观察，多喝水多休息，不要剧烈运动。"
    await _worker_returning(
        monkeypatch, coordinator, {"answer": original, "agent_id": "consultation_agent"}
    )

    result = await coordinator.process("我父亲突然胸痛伴呼吸困难怎么办")

    assert result["warning"] == "emergency_hard_constraint"
    assert result["emergency_keyword"] == "胸痛"
    assert result["reference_answer"] == original
    assert "120" in result["answer"]
    assert result["answer"] != original
    # 维度信息已附加（供审核与飞轮下钻）
    assert result["review_dimensions"]["disease_category"] == "心血管"
    assert result["review_dimensions"]["agent_type"] == "consultation_agent"


async def test_emergency_answer_with_guidance_kept(monkeypatch, coordinator):
    """高危问题但回答已含就医引导 → 不改写"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答", "assigned_agent": "consultation_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    guided = "出现这种症状请立即就医或拨打 120，不要拖延。"
    await _worker_returning(
        monkeypatch, coordinator, {"answer": guided, "agent_id": "consultation_agent"}
    )

    result = await coordinator.process("我父亲突然胸痛伴呼吸困难怎么办")

    assert result["answer"] == guided
    assert "warning" not in result


async def test_normal_question_unchanged(monkeypatch, coordinator):
    """普通问题 → 回答原样返回，不触发硬约束"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答", "assigned_agent": "consultation_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    plain = "感冒期间建议多喝水、保证休息。以上内容仅供参考。"
    await _worker_returning(
        monkeypatch, coordinator, {"answer": plain, "agent_id": "consultation_agent"}
    )

    result = await coordinator.process("感冒了怎么办")

    assert result["answer"] == plain


# ========== 编排图专项行为 ==========

async def test_single_agent_skips_synthesis(monkeypatch, coordinator):
    """单子任务 → 直接返回 Worker 结果，不调用综合"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答", "assigned_agent": "consultation_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    synthesize_called = []

    async def recorder(question, worker_results, timeout_occurred=False):
        synthesize_called.append(worker_results)
        return "不应被使用"

    monkeypatch.setattr(coordinator, "_synthesize", recorder)

    await _worker_returning(
        monkeypatch, coordinator,
        {"answer": "单Agent答案", "agent_id": "consultation_agent", "skills_used": []},
    )
    result = await coordinator.process("感冒了怎么办")

    assert synthesize_called == []
    assert result["answer"] == "单Agent答案"
    assert result["agent_id"] == "consultation_agent"


async def test_swarm_fanout_and_synthesis(monkeypatch, coordinator):
    """多子任务 → 各子任务描述派发给对应 Worker，结果经综合后返回"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答问题A", "assigned_agent": "consultation_agent"},
        {"type": "研究", "description": "回答问题B", "assigned_agent": "research_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    dispatch_log = []

    async def fake_worker(agent_id, question_input, session_id):
        dispatch_log.append((agent_id, question_input))
        return {"answer": f"{agent_id}的回答", "agent_id": agent_id, "skills_used": []}

    monkeypatch.setattr(coordinator, "_execute_worker", fake_worker)

    synthesis_inputs = []

    async def fake_synthesize(question, worker_results, timeout_occurred=False):
        synthesis_inputs.append(worker_results)
        return "综合后的答案：建议多喝水。"

    monkeypatch.setattr(coordinator, "_synthesize", fake_synthesize)

    result = await coordinator.process("复杂的多方面问题" * 3)

    # 两个子任务按描述派发
    assert ("consultation_agent", "回答问题A") in dispatch_log
    assert ("research_agent", "回答问题B") in dispatch_log
    # 综合拿到两个 Worker 结果
    assert len(synthesis_inputs[0]) == 2
    assert result["answer"] == "综合后的答案：建议多喝水。"
    assert result["swarm_summary"]["total_subtasks"] == 2
    assert result["swarm_summary"]["completed_subtasks"] == 2
    assert result["timeout_occurred"] is False


async def test_worker_timeout_fallback(monkeypatch, coordinator):
    """Worker 超时 → _execute_worker 的 wait_for 触发，全部无贡献时走兜底"""
    import asyncio

    class _SlowRunner:
        async def run(self, question, session_id):
            await asyncio.sleep(10)

    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答问题A", "assigned_agent": "consultation_agent"},
        {"type": "研究", "description": "回答问题B", "assigned_agent": "research_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    coordinator.runners = {
        "consultation_agent": _SlowRunner(),
        "research_agent": _SlowRunner(),
    }
    coordinator.worker_timeout = 0.1

    result = await coordinator.process("复杂的多方面问题" * 3)

    assert result["warning"] == "all_workers_failed"
    assert result["timeout_occurred"] is True


async def test_unknown_agent_falls_back_to_consultation(monkeypatch, coordinator):
    """分配到不存在的 Agent → _execute_worker 回退 consultation_agent"""
    dispatch_log = []

    class _RecordingRunner:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        async def run(self, question, session_id):
            dispatch_log.append(self.agent_id)
            return {"answer": "回退回答", "agent_id": self.agent_id, "skills_used": []}

    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答", "assigned_agent": "ghost_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    coordinator.runners = {"consultation_agent": _RecordingRunner("consultation_agent")}

    result = await coordinator.process("感冒了怎么办")

    assert dispatch_log == ["consultation_agent"]
    assert result["agent_id"] == "consultation_agent"


# ========== WorkerRunner 端到端（脚本化 LLM） ==========

class _ScriptedLLMClient:
    """按脚本顺序返回 tool_calls / 最终文本的 LLM 桩"""

    def __init__(self, script):
        self.script = list(script)
        self.received_tools = []

    async def chat_with_tools_retry(self, messages, tools=None, tool_choice="auto",
                                    temperature=None, max_tokens=None, **kwargs):
        self.received_tools.append(tools)
        return self.script.pop(0)

    async def chat(self, messages, **kwargs):
        return "兜底答案"


class _FakeAgent:
    agent_id = "fake_agent"
    config = {}

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.skill_registry = SkillRegistry()
        self.executions = []

        async def echo_skill(query):
            self.executions.append(query)
            return {"success": True, "echo": query}

        self.skill_registry.register(
            name="echo_skill",
            function=echo_skill,
            description="回显技能",
            parameters=[SkillParameter(name="query", type="string", description="查询内容", required=True)],
        )

    def get_system_prompt(self):
        return "你是测试Agent"

    async def execute_tool(self, tool_name, arguments):
        return await self.skill_registry.execute(tool_name, **arguments)

    async def post_process_result(self, result, final_response):
        return result


def _tool_call_response(call_id, name, args):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        finish_reason="tool_calls",
    )


async def test_worker_runner_tool_loop():
    """react 循环：一次工具调用后产出最终答案，skills_used 记录正确"""
    llm = _ScriptedLLMClient([
        _tool_call_response("call_1", "echo_skill", {"query": "你好"}),
        LLMResponse(content="最终答案：多喝水", tool_calls=[], finish_reason="stop"),
    ])
    agent = _FakeAgent(llm)
    runner = WorkerRunner(agent, max_tool_calls=2)

    result = await runner.run("测试问题")

    assert result["answer"] == "最终答案：多喝水"
    assert result["agent_id"] == "fake_agent"
    assert result["skills_used"] == ["echo_skill"]
    assert agent.executions == ["你好"]
    # 工具 schema 以 OpenAI dict 格式透传给底层 LLM 客户端
    first_tools = llm.received_tools[0]
    assert first_tools[0]["function"]["name"] == "echo_skill"
    assert first_tools[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"


async def test_worker_runner_max_tool_calls_budget():
    """工具调用预算：技能真实执行不超过 max_tool_calls 次，超限返回引导文本"""
    # LLM 连续 4 次请求工具，最后才给最终答案
    llm = _ScriptedLLMClient([
        _tool_call_response("call_1", "echo_skill", {"query": "q1"}),
        _tool_call_response("call_2", "echo_skill", {"query": "q2"}),
        _tool_call_response("call_3", "echo_skill", {"query": "q3"}),
        LLMResponse(content="基于已有信息的最终答案", tool_calls=[], finish_reason="stop"),
    ])
    agent = _FakeAgent(llm)
    runner = WorkerRunner(agent, max_tool_calls=2)

    result = await runner.run("测试问题")

    # 技能只真实执行了 2 次（第 3 次被预算拦截）
    assert agent.executions == ["q1", "q2"]
    assert result["answer"] == "基于已有信息的最终答案"
    assert runner.budget.call_count == 2


# ========== Harness Engineering: 任务分解约束 ==========

async def test_decomposition_clipped_to_max_agents(monkeypatch, coordinator):
    """LLM 分解出超过 max_agents_per_task 的子任务 → 裁剪到上限(3)"""
    await _decompose(monkeypatch, coordinator, [
        {"type": f"任务{i}", "description": f"子任务{i}", "assigned_agent": "consultation_agent"}
        for i in range(5)
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    dispatch_log = []

    async def fake_worker(agent_id, question_input, session_id):
        dispatch_log.append(agent_id)
        return {"answer": "回答", "agent_id": agent_id, "skills_used": []}

    monkeypatch.setattr(coordinator, "_execute_worker", fake_worker)

    result = await coordinator.process("复杂的多方面问题" * 3)

    # 5 个子任务被裁剪为 3 个,只派发 3 个 Worker
    assert len(dispatch_log) == 3
    assert result["swarm_summary"]["total_subtasks"] == 3


async def test_high_risk_decomposition_without_diagnostic_warns_not_blocks(monkeypatch, coordinator):
    """高风险问题未分配诊断 Agent → 仅告警,不阻断流程"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答", "assigned_agent": "consultation_agent"},
    ])
    monkeypatch.setattr(oc_module, "REVIEW_ENABLED", False)

    await _worker_returning(
        monkeypatch, coordinator,
        {"answer": "请立即就医或拨打 120。", "agent_id": "consultation_agent", "skills_used": []},
    )

    result = await coordinator.process("我父亲突然胸痛怎么办")

    # 告警不阻断:正常返回回答
    assert "120" in result["answer"]


# ========== contextvar 路由传播 ==========

async def test_route_contextvar_propagates_through_send():
    """set_route 设置的请求级路由在 Send 并行分支中可见（模型路由不因并行失效）"""
    from langgraph.graph import START, StateGraph
    from langgraph.types import Send
    from core.model_router import get_route, reset_route, set_route

    class _FanState(TypedDict, total=False):
        items: Annotated[list, operator.add]

    observed = []

    async def worker_branch(state):
        observed.append(get_route())
        return {"items": [state.get("k")]}

    def fanout(state):
        return [Send("worker_branch", {"k": i}) for i in range(3)]

    graph = StateGraph(_FanState)
    graph.add_node("worker_branch", worker_branch)
    graph.add_conditional_edges(START, fanout, ["worker_branch"])
    compiled = graph.compile()

    token = set_route("small")
    try:
        final_state = await compiled.ainvoke({"items": []})
    finally:
        reset_route(token)

    assert observed == ["small", "small", "small"]
    assert sorted(final_state["items"]) == [0, 1, 2]
