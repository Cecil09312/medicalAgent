"""飞轮维度统计单元测试：维度聚合、下钻排序、持久化与旧格式兼容"""
import json

from flywheel.dimensions import build_review_dimensions, classify_disease_category, top_dimensions
from flywheel.metrics_tracker import FlywheelMetrics


class TestClassifyDiseaseCategory:
    def test_categories(self):
        assert classify_disease_category("胸痛胸闷怎么办") == "心血管"
        assert classify_disease_category("孩子咳嗽发烧") == "呼吸"
        assert classify_disease_category("血糖高饮食注意什么") == "内分泌"

    def test_unknown(self):
        assert classify_disease_category("宇宙的尽头是什么") == "unknown"
        assert classify_disease_category("") == "unknown"


class TestBuildReviewDimensions:
    def test_from_result(self):
        dims = build_review_dimensions(
            "头痛得厉害",
            {"agent_id": "diagnostic_agent", "skills_used": ["analyze_symptoms"]},
        )
        assert dims == {
            "disease_category": "神经",
            "skill_name": "analyze_symptoms",
            "agent_type": "diagnostic_agent",
        }

    def test_missing_fields_use_unknown(self):
        dims = build_review_dimensions("随便问问", {})
        assert dims["skill_name"] == "unknown"
        assert dims["agent_type"] == "unknown"


class TestDimensionStats:
    def test_record_and_breakdown(self, tmp_path):
        metrics = FlywheelMetrics(metrics_path=str(tmp_path / "m.json"))
        dims_cardio = {"disease_category": "心血管"}
        dims_resp = {"disease_category": "呼吸"}

        metrics.record_review(3.0, dimensions=dims_cardio)
        metrics.record_review(5.0, dimensions=dims_cardio)
        metrics.record_correction(dims_cardio)
        metrics.record_review(2.0, dimensions=dims_resp)
        metrics.record_correction(dims_resp)
        metrics.record_correction(dims_resp)
        metrics.record_review(1.0)  # 无维度 → unknown

        breakdown = metrics.get_dimension_breakdown("disease_category")
        # 修正数降序：呼吸(2) > 心血管(1) > unknown(0)
        assert breakdown[0][0] == "呼吸"
        assert breakdown[0][1:] == (1, 2, 2.0)  # 修正率 = 2 修正 / 1 审核
        assert breakdown[1][0] == "心血管"
        assert breakdown[2][0] == "unknown"

    def test_no_dimensions_aggregates_unknown(self, tmp_path):
        metrics = FlywheelMetrics(metrics_path=str(tmp_path / "m.json"))
        metrics.record_review(1.0)
        metrics.record_correction()
        breakdown = metrics.get_dimension_breakdown("disease_category")
        assert breakdown == [("unknown", 1, 1, 1.0)]

    def test_persistence_roundtrip(self, tmp_path):
        path = str(tmp_path / "m.json")
        metrics = FlywheelMetrics(metrics_path=path)
        metrics.record_review(1.0, dimensions={"skill_name": "analyze_symptoms"})
        metrics.record_correction({"skill_name": "analyze_symptoms"})

        reloaded = FlywheelMetrics(metrics_path=path)
        breakdown = reloaded.get_dimension_breakdown("skill_name")
        assert breakdown[0][0] == "analyze_symptoms"

    def test_old_format_without_dimensions_loads(self, tmp_path):
        """旧格式文件（无 dimensions 字段）加载不报错，维度统计为空表"""
        path = tmp_path / "old.json"
        path.write_text(json.dumps([{
            "timestamp": 1700000000.0, "total_queries": 5, "total_reviews": 2,
            "total_corrections": 1, "total_approvals": 1, "total_rejections": 0,
            "intervention_rate": 0.4, "correction_rate": 0.5,
            "avg_review_time_seconds": 3.0, "kb_expert_docs": 0, "eval_cases_from_flywheel": 0,
        }], ensure_ascii=False), encoding="utf-8")

        metrics = FlywheelMetrics(metrics_path=str(path))
        assert metrics.get_current().total_queries == 5
        assert metrics.get_dimension_breakdown("disease_category") == []


def test_top_dimensions_sorting():
    stats = {
        "disease_category": {
            "a": {"reviews": 5, "corrections": 0},
            "b": {"reviews": 4, "corrections": 3},
            "c": {"reviews": 2, "corrections": 3},
            "d": {"reviews": 1, "corrections": 1},
        }
    }
    rows = top_dimensions(stats, "disease_category", top_n=3)
    assert [r[0] for r in rows] == ["b", "c", "d"]  # 修正数降序，同修正数审核数优先
