"""WorkerRunner 集成测试（stub LLM，无真实 API）：审核集成、skills_used、专家实时修正

安全契约:
- 高危问题强制实时审核，超时转异步待办；AutoFixer 自动补齐免责声明
- 专家在实时审核窗口内提交修正 → 返回修正后的回答
- Skill 调用记录到 skills_used
- 普通问题 + 采样率 0 → 不产生审核项
"""
import asyncio
import json

import pytest

from orchestrator.worker_runner import WorkerRunner
from core.llm_client import LLMResponse, ToolCall
from core.skill_registry import SkillRegistry, SkillParameter
from review.review_queue import get_default_queue
import review.review_queue as review_module
from flywheel.metrics_tracker import FlywheelMetrics
import flywheel.metrics_tracker as metrics_module


@pytest.fixture(autouse=True)
def _isolated_review_and_metrics(monkeypatch, tmp_path):
    """每个测试使用独立的审核队列与飞轮指标（均为进程级单例，需显式重置隔离）"""
    monkeypatch.setenv("REVIEW_DATA_PATH", str(tmp_path / "reviews.json"))
    monkeypatch.setattr(review_module, "_default_queue", None)
    monkeypatch.setattr(
        metrics_module, "_default_metrics",
        FlywheelMetrics(metrics_path=str(tmp_path / "metrics.json")),
    )
    yield


class _StubLLM:
    """按脚本返回的 LLM 客户端桩"""

    def __init__(self, scripted):
        self.scripted = list(scripted)

    async def chat_with_tools_retry(self, messages, tools=None, **kw):
        if self.scripted:
            return self.scripted.pop(0)
        return LLMResponse(content="（默认回答）仅供参考。", tool_calls=[], finish_reason="stop")


class _StubAgent:
    agent_id = "consultation_agent"
    config = {}

    def __init__(self, llm, skill_registry=None):
        self.llm_client = llm
        self.skill_registry = skill_registry or SkillRegistry()

    def get_system_prompt(self):
        return "你是医疗助手"

    async def execute_tool(self, tool_name, arguments):
        return await self.skill_registry.execute(tool_name, **arguments)


def _final(text):
    return LLMResponse(content=text, tool_calls=[], finish_reason="stop")


async def test_high_risk_question_forces_realtime_review(monkeypatch):
    """高危问题 -> 强制实时审核，超时转异步待办；Runner 层自动补齐免责声明"""
    monkeypatch.setenv("REVIEW_REALTIME_TIMEOUT", "0")

    llm = _StubLLM([_final("您注意休息即可，先在家观察看看。")])
    runner = WorkerRunner(_StubAgent(llm))
    result = await runner.run("我父亲突然胸痛伴大出汗")

    # 强制实时审核已触发并超时转异步
    items = get_default_queue().get_pending()
    assert any(i.trigger_reason == "high_risk" and i.review_mode == "async" for i in items)
    # Runner 层 AutoFixer 补齐免责声明，原文保留
    assert "您注意休息即可，先在家观察看看。" in result["answer"]
    assert "免责声明" in result["answer"]


async def test_expert_correction_replaces_answer(monkeypatch):
    """专家在实时审核窗口内提交修正 -> Runner 返回修正后的回答"""
    monkeypatch.setenv("REVIEW_REALTIME_TIMEOUT", "5")

    llm = _StubLLM([_final("可以先观察。")])
    runner = WorkerRunner(_StubAgent(llm))

    async def _run():
        return await runner.run("我父亲突然胸痛")

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.2)  # 等待审核项入队
    item = get_default_queue().get_pending()[0]

    assert get_default_queue().submit_correction(item.id, "请立即拨打 120 前往急诊。仅供参考。")
    result = await task

    assert result["answer"] == "请立即拨打 120 前往急诊。仅供参考。"
    assert result.get("expert_reviewed") is True


async def test_skills_used_tracked():
    """Skill 调用被记录到结果的 skills_used 字段"""

    registry = SkillRegistry()

    async def fake_search(query):
        return {"answer": "感冒相关知识", "sources": []}

    registry.register(
        name="search_knowledge",
        function=fake_search,
        description="搜索医学知识库",
        parameters=[SkillParameter(name="query", type="string", description="查询", required=True)],
    )

    llm = _StubLLM([
        LLMResponse(content=None,
                    tool_calls=[ToolCall(id="t1", name="search_knowledge", arguments={"query": "感冒"})],
                    finish_reason="tool_calls"),
        _final("感冒建议多喝水休息。仅供参考。"),
    ])
    runner = WorkerRunner(_StubAgent(llm, skill_registry=registry))
    result = await runner.run("感冒了怎么办")

    assert result["skills_used"] == ["search_knowledge"]
    assert result["answer"] == "感冒建议多喝水休息。仅供参考。"


async def test_normal_question_no_review(monkeypatch):
    """普通问题 + 采样率 0：不产生审核项"""
    monkeypatch.setenv("REVIEW_ASYNC_SAMPLE_RATE", "0.0")

    before = len(get_default_queue().get_pending())
    llm = _StubLLM([_final("多喝水。以上内容仅供参考。")])
    runner = WorkerRunner(_StubAgent(llm))
    await runner.run("感冒了怎么办")
    assert len(get_default_queue().get_pending()) == before
