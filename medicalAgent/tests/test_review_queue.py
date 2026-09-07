"""审核队列单元测试：入队/审批/修正流程、实时唤醒、维度透传与持久化"""
import asyncio
import json

import pytest

from review.review_queue import ReviewQueue

import flywheel.metrics_tracker as metrics_module
from flywheel.metrics_tracker import FlywheelMetrics


@pytest.fixture(autouse=True)
def _isolated_metrics(tmp_path, monkeypatch):
    """
    隔离飞轮指标单例（autouse）：

    专家批准/修正会通过 _record_review_metrics 写入进程级指标单例，
    不隔离会把测试数据写进真实的 flywheel/data/flywheel_metrics.json
    """
    monkeypatch.setattr(
        metrics_module, "_default_metrics",
        FlywheelMetrics(metrics_path=str(tmp_path / "metrics.json")),
    )
    yield


async def test_enqueue_and_stats(tmp_path):
    queue = ReviewQueue(data_path=tmp_path / "reviews.json")
    review_id = queue.enqueue_async("感冒怎么办", "多喝水", "random_sample")
    assert queue.get_item(review_id).status == "pending"
    stats = queue.get_stats()
    assert stats["pending"] == 1 and stats["total"] == 1


async def test_correction_flow_records_metrics(tmp_path):
    metrics = metrics_module._default_metrics
    queue = ReviewQueue(data_path=tmp_path / "reviews.json")
    review_id = queue.enqueue_async(
        "胸痛怎么办", "立即就医", "high_risk",
        dimensions={"disease_category": "心血管", "skill_name": "unknown", "agent_type": "diagnostic_agent"},
    )

    assert queue.submit_correction(review_id, "修正后的回答") is True
    assert queue.get_item(review_id).status == "corrected"

    # 飞轮指标按维度累计
    assert metrics.get_current().total_reviews == 1
    assert metrics.get_current().total_corrections == 1
    breakdown = {v: (r, c) for v, r, c, _ in metrics.get_dimension_breakdown("disease_category")}
    assert breakdown["心血管"] == (1, 1)


async def test_realtime_timeout_becomes_async(tmp_path):
    queue = ReviewQueue(data_path=tmp_path / "reviews.json")
    result = await queue.submit_realtime("问题", "回答", "high_risk", timeout=0)
    assert result.action == "timeout"
    assert queue.get_pending()[0].review_mode == "async"


async def test_realtime_approval_wakes_waiter(tmp_path):
    queue = ReviewQueue(data_path=tmp_path / "reviews.json")
    task = asyncio.create_task(
        queue.submit_realtime("高危问题", "回答", "high_risk", timeout=5)
    )
    await asyncio.sleep(0.1)  # 等待审核项创建与事件注册
    item_id = queue.get_pending()[0].id

    assert queue.submit_approval(item_id) is True
    result = await task
    assert result.action == "approved"
    assert result.review_id == item_id


async def test_persistence_roundtrip_with_dimensions(tmp_path):
    path = tmp_path / "reviews.json"
    queue = ReviewQueue(data_path=path)
    queue.enqueue_async(
        "问题", "回答", "user_flag",
        dimensions={"disease_category": "神经", "skill_name": "search_knowledge", "agent_type": "unknown"},
    )

    # 新实例从磁盘加载：维度字段与状态保持
    reloaded = ReviewQueue(data_path=path)
    items = reloaded.get_pending()
    assert len(items) == 1
    assert items[0].dimensions["disease_category"] == "神经"


async def test_old_format_without_dimensions_loads(tmp_path):
    """旧数据无 dimensions 字段：加载不报错，字段为 None"""
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps([{
        "id": "legacy01", "question": "q", "original_answer": "a",
        "trigger_reason": "random_sample", "review_mode": "async",
        "status": "pending", "expert_correction": None, "expert_id": None,
        "created_at": 1700000000.0, "reviewed_at": None,
    }], ensure_ascii=False), encoding="utf-8")

    queue = ReviewQueue(data_path=path)
    item = queue.get_item("legacy01")
    assert item is not None
    assert item.dimensions is None


async def test_atomic_write_no_tmp_leftover(tmp_path):
    path = tmp_path / "reviews.json"
    queue = ReviewQueue(data_path=path)
    queue.enqueue_async("q", "a", "random_sample")
    assert not (tmp_path / "reviews.json.tmp").exists()
