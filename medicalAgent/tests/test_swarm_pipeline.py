"""Swarm 编排主链路集成测试（mock LLM）：兜底路径与医疗安全硬约束"""
import pytest

import swarm.swarm_coordinator as sc_module
from swarm.swarm_coordinator import SwarmCoordinator


class _StubLLMClient:
    """占位 LLM 客户端：所有 LLM 调用路径均被 monkeypatch，不会真正发起请求"""


@pytest.fixture
def coordinator():
    """空 Worker 集合的协调器：LeadAgent 的 LLM 依赖全部由测试 monkeypatch"""
    coord = SwarmCoordinator(worker_agents={}, llm_client=_StubLLMClient())
    yield coord


async def _decompose(monkeypatch, coord, subtasks):
    async def fake_assess(question, context=None):
        return {"analysis": "测试", "decomposition": subtasks}

    monkeypatch.setattr(coord.lead_agent, "assess_and_decompose", fake_assess)


async def test_all_workers_failed_fallback(monkeypatch, coordinator):
    """全 Worker 失败（无可用 Worker）→ 跳过 LLM 合成，返回结构化兜底结果"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答问题A", "assigned_agent": "consultation_agent"},
        {"type": "研究", "description": "回答问题B", "assigned_agent": "research_agent"},
    ])
    monkeypatch.setattr(sc_module, "REVIEW_ENABLED", False)

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
    monkeypatch.setattr(sc_module, "REVIEW_ENABLED", False)

    original = "您可以先在家观察，多喝水多休息，不要剧烈运动。"
    async def fake_single(question, subtasks, session_id):
        return {"answer": original, "agent_id": "consultation_agent"}

    monkeypatch.setattr(coordinator, "_process_single_agent", fake_single)

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
    monkeypatch.setattr(sc_module, "REVIEW_ENABLED", False)

    guided = "出现这种症状请立即就医或拨打 120，不要拖延。"
    async def fake_single(question, subtasks, session_id):
        return {"answer": guided, "agent_id": "consultation_agent"}

    monkeypatch.setattr(coordinator, "_process_single_agent", fake_single)

    result = await coordinator.process("我父亲突然胸痛伴呼吸困难怎么办")

    assert result["answer"] == guided
    assert "warning" not in result


async def test_normal_question_unchanged(monkeypatch, coordinator):
    """普通问题 → 回答原样返回，不触发硬约束"""
    await _decompose(monkeypatch, coordinator, [
        {"type": "咨询", "description": "回答", "assigned_agent": "consultation_agent"},
    ])
    monkeypatch.setattr(sc_module, "REVIEW_ENABLED", False)

    plain = "感冒期间建议多喝水、保证休息。以上内容仅供参考。"
    async def fake_single(question, subtasks, session_id):
        return {"answer": plain, "agent_id": "consultation_agent"}

    monkeypatch.setattr(coordinator, "_process_single_agent", fake_single)

    result = await coordinator.process("感冒了怎么办")

    assert result["answer"] == plain
