"""医疗安全硬约束（emergency）单元测试：检测、就医引导判定、标准话术与强制审核触发"""
from constraints.emergency import (
    detect_emergency,
    emergency_response,
    get_emergency_keywords,
    has_urgent_care_guidance,
)
from review.review_trigger import ReviewTrigger


class TestDetectEmergency:
    def test_hits_keyword(self):
        assert detect_emergency("我父亲突然胸痛伴呼吸困难") == "胸痛"

    def test_plain_text_no_hit(self):
        assert detect_emergency("最近有点咳嗽，吃什么水果好") is None

    def test_empty_and_none_safe(self):
        assert detect_emergency("") is None
        assert detect_emergency(None) is None

    def test_keywords_configured(self):
        keywords = get_emergency_keywords()
        assert "胸痛" in keywords and "自杀" in keywords


class TestUrgentCareGuidance:
    def test_with_guidance(self):
        assert has_urgent_care_guidance("请立即就医或拨打 120") is True
        assert has_urgent_care_guidance("建议尽快前往医院就诊") is True

    def test_without_guidance(self):
        assert has_urgent_care_guidance("可以先观察，多喝水多休息") is False
        assert has_urgent_care_guidance("") is False


class TestEmergencyResponse:
    def test_contains_keyword_and_120(self):
        text = emergency_response("呼吸困难")
        assert "呼吸困难" in text
        assert "120" in text

    def test_none_keyword_still_guides(self):
        assert "120" in emergency_response(None)


class TestForcedRealtimeReview:
    """问题命中高危关键词 → 强制实时专家审核（review_trigger 条件 0）"""

    def _trigger(self):
        trigger = ReviewTrigger()
        trigger.async_sample_rate = 0.0
        return trigger

    def test_emergency_question_forces_review(self):
        should, reason = self._trigger().should_review_realtime(
            agent_output={"answer": "普通回答，无敏感内容"},
            question="我爷爷突然昏迷不醒怎么办",
        )
        assert should is True
        assert reason == "high_risk"

    def test_plain_question_no_forced_review(self):
        should, reason = self._trigger().should_review_realtime(
            agent_output={"answer": "普通回答，无敏感内容"},
            question="感冒了多喝水有用吗",
        )
        assert should is False

    def test_answer_keyword_still_triggers(self):
        # 兼容旧行为：回答本身含高风险关键词也触发
        should, reason = self._trigger().should_review_realtime(
            agent_output={"answer": "该症状可能是心肌梗死的前兆"},
        )
        assert should is True
