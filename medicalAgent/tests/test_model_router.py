"""model_router 单元测试：启发式路由、contextvar、边界区间与二级分类回退"""
import pytest

from core import model_router
from core.model_router import (
    classify_question,
    get_route,
    is_boundary_question,
    is_simple_question,
    reset_route,
    route_scope,
    set_route,
)


@pytest.fixture(autouse=True)
def _deterministic_thresholds(monkeypatch):
    """固定阈值常量，避免宿主机环境变量影响断言"""
    monkeypatch.setattr(model_router, "ROUTE_SIMPLE_MAX_LEN", 30)
    monkeypatch.setattr(model_router, "ROUTE_LLM_BOUNDARY_MIN", 15)
    monkeypatch.setattr(model_router, "ROUTE_LLM_BOUNDARY_MAX", 60)
    monkeypatch.setattr(model_router, "ROUTE_LLM_CLASSIFY_ENABLED", False)


class TestIsSimpleQuestion:
    def test_empty_is_complex(self):
        assert is_simple_question("") is False
        assert is_simple_question("   ") is False

    def test_short_plain_is_simple(self):
        assert is_simple_question("感冒了怎么办") is True

    def test_long_is_complex(self):
        assert is_simple_question("头" * 30) is False  # 长度 >= 30

    def test_complex_keyword_is_complex(self):
        for kw in ("研究", "分析", "比较", "机制"):
            assert is_simple_question(f"帮我{kw}一下这个症状") is False

    def test_boundary_length_is_simple_by_heuristic(self):
        # 长度 15~29 且无复杂关键词：启发式仍判简单（二级分类的临界区）
        assert is_simple_question("最近总是睡不好白天没精神怎么办呀") is True


class TestRouteContextVar:
    def test_default_auto(self):
        assert get_route() == "auto"

    def test_set_and_reset(self):
        token = set_route("small")
        assert get_route() == "small"
        reset_route(token)
        assert get_route() == "auto"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            set_route("huge")

    def test_route_scope_restores(self):
        with route_scope("big"):
            assert get_route() == "big"
        assert get_route() == "auto"


class TestBoundary:
    def test_boundary_range(self):
        assert is_boundary_question("最近总是睡不好白天没精神怎么办呀") is True  # 长度 16
        assert is_boundary_question("感冒了怎么办") is False  # 太短
        assert is_boundary_question("怎" * 60) is False  # 超出上限
        assert is_boundary_question("请分析一下这个胸痛的原因和应对方法是什么") is False  # 含复杂关键词
        assert is_boundary_question("") is False


class TestClassifyQuestion:
    async def test_disabled_returns_heuristic(self):
        # 未启用二级分类：行为与 is_simple_question 完全一致
        assert await classify_question("感冒了怎么办") is True
        assert await classify_question("请分析这个症状的原因") is False

    async def test_enabled_non_boundary_returns_heuristic(self, monkeypatch):
        monkeypatch.setattr(model_router, "ROUTE_LLM_CLASSIFY_ENABLED", True)
        # 非临界问题不触发 LLM 分类
        assert await classify_question("感冒了怎么办") is True

    async def test_enabled_boundary_llm_result_wins(self, monkeypatch):
        monkeypatch.setattr(model_router, "ROUTE_LLM_CLASSIFY_ENABLED", True)
        # 启发式判 simple（短于 30），LLM 判 complex → 以 LLM 为准
        async def fake_llm(question):
            return False

        monkeypatch.setattr(model_router, "_classify_by_llm", fake_llm)
        question = "最近总是睡不好白天没精神怎么办呀"
        assert await classify_question(question) is False

    async def test_classify_failure_falls_back_to_heuristic(self, monkeypatch):
        monkeypatch.setattr(model_router, "ROUTE_LLM_CLASSIFY_ENABLED", True)

        async def broken_llm(question):
            raise TimeoutError("模拟分类超时")

        monkeypatch.setattr(model_router, "_classify_by_llm", broken_llm)
        question = "最近总是睡不好白天没精神怎么办呀"
        assert await classify_question(question) is is_simple_question(question)
