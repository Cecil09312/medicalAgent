"""AgentLoop 集成测试（stub LLM，无真实 API）：审核集成、skills_used、专家实时修正"""
import asyncio
import json

import pytest

from core.agent_loop import AgentLoop
from core.llm_client import LLMResponse, ToolCall
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

    def create_tool_message(self, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False)}


class _StubAgent:
    agent_id = "consultation_agent"
    config = {}

    def __init__(self, llm):
        self.llm_client = llm

    def get_system_prompt(self):
        return "你是医疗助手"

    def get_tools_for_llm(self):
        return []

    def format_user_input(self, d):
        return d.get("question", str(d))

    async def execute_tool(self, tool_name, arguments):
        return {"ok": True}


def _final(text):
    return LLMResponse(content=text, tool_calls=[], finish_reason="stop")


async def test_high_risk_question_forces_realtime_review(monkeypatch):
    """高危问题 -> 强制实时审核，超时转异步待办；Agent 层自动补齐免责声明"""
    monkeypatch.setenv("REVIEW_REALTIME_TIMEOUT", "0")

    llm = _StubLLM([_final("您注意休息即可，先在家观察看看。")])
    result = await AgentLoop(max_iterations=3).run(
        _StubAgent(llm), {"question": "我父亲突然胸痛伴大出汗"}
    )

    # 强制实时审核已触发并超时转异步
    items = get_default_queue().get_pending()
    assert any(i.trigger_reason == "high_risk" and i.review_mode == "async" for i in items)
    # Agent 层 AutoFixer 补齐免责声明，原文保留
    assert "您注意休息即可，先在家观察看看。" in result["answer"]
    assert "免责声明" in result["answer"]


async def test_expert_correction_replaces_answer(monkeypatch):
    """专家在实时审核窗口内提交修正 -> AgentLoop 返回修正后的回答"""
    monkeypatch.setenv("REVIEW_REALTIME_TIMEOUT", "5")

    llm = _StubLLM([_final("可以先观察。")])
    loop = AgentLoop(max_iterations=3)

    async def _run():
        return await loop.run(_StubAgent(llm), {"question": "我父亲突然胸痛"})

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.2)  # 等待审核项入队
    item = get_default_queue().get_pending()[0]

    assert get_default_queue().submit_correction(item.id, "请立即拨打 120 前往急诊。仅供参考。")
    result = await task

    assert result["answer"] == "请立即拨打 120 前往急诊。仅供参考。"
    assert result.get("expert_reviewed") is True


async def test_skills_used_tracked():
    """Skill 调用被记录到结果的 skills_used 字段"""

    llm = _StubLLM([
        LLMResponse(content=None,
                    tool_calls=[ToolCall(id="t1", name="search_knowledge", arguments={"query": "感冒"})],
                    finish_reason="tool_calls"),
        _final("感冒建议多喝水休息。仅供参考。"),
    ])
    result = await AgentLoop(max_iterations=3).run(
        _StubAgent(llm), {"question": "感冒了怎么办"}
    )
    assert result["skills_used"] == ["search_knowledge"]
    assert result["answer"] == "感冒建议多喝水休息。仅供参考。"


async def test_normal_question_no_review(monkeypatch):
    """普通问题 + 采样率 0：不产生审核项"""
    monkeypatch.setenv("REVIEW_ASYNC_SAMPLE_RATE", "0.0")

    before = len(get_default_queue().get_pending())
    llm = _StubLLM([_final("多喝水。以上内容仅供参考。")])
    await AgentLoop(max_iterations=3).run(_StubAgent(llm), {"question": "感冒了怎么办"})
    assert len(get_default_queue().get_pending()) == before
