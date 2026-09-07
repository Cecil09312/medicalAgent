"""circuit_breaker 单元测试：三态状态机迁移与阈值/冷却/单例"""
import uuid

from core.circuit_breaker import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    get_circuit_breaker,
)


def _fresh_breaker(monkeypatch, threshold=3, cooldown=60) -> CircuitBreaker:
    """创建隔离的熔断器（固定阈值/冷却，避免环境变量影响）"""
    monkeypatch.setenv("CIRCUIT_FAILURE_THRESHOLD", str(threshold))
    monkeypatch.setenv("CIRCUIT_COOLDOWN_SECONDS", str(cooldown))
    return CircuitBreaker(f"test-model-{uuid.uuid4().hex[:8]}")


async def test_initial_closed_allows(monkeypatch):
    breaker = _fresh_breaker(monkeypatch)
    assert breaker.state == STATE_CLOSED
    assert await breaker.allow_request() is True


async def test_threshold_opens_circuit(monkeypatch):
    breaker = _fresh_breaker(monkeypatch, threshold=3)
    for _ in range(2):
        await breaker.record_failure()
    assert breaker.state == STATE_CLOSED  # 未达阈值不熔断
    await breaker.record_failure()
    assert breaker.state == STATE_OPEN
    assert await breaker.allow_request() is False


async def test_success_resets_failure_count(monkeypatch):
    breaker = _fresh_breaker(monkeypatch, threshold=3)
    await breaker.record_failure()
    await breaker.record_failure()
    await breaker.record_success()  # 瞬时抖动恢复：计数清零
    await breaker.record_failure()
    assert breaker.state == STATE_CLOSED  # 单次失败不再熔断


async def test_cooldown_expiry_enters_half_open(monkeypatch):
    breaker = _fresh_breaker(monkeypatch, threshold=1, cooldown=60)
    await breaker.record_failure()
    assert breaker.state == STATE_OPEN
    assert await breaker.allow_request() is False  # 冷却期内拒绝

    # 模拟冷却期已过
    breaker.opened_at -= 61
    assert await breaker.allow_request() is True  # 放行探测
    assert breaker.state == STATE_HALF_OPEN
    # 半开状态同一时刻只放行一个探测请求
    assert await breaker.allow_request() is False


async def test_half_open_probe_success_closes(monkeypatch):
    breaker = _fresh_breaker(monkeypatch, threshold=1, cooldown=60)
    await breaker.record_failure()
    breaker.opened_at -= 61
    await breaker.allow_request()  # 进入半开并占用探测位
    await breaker.record_success()  # 探测成功
    assert breaker.state == STATE_CLOSED
    assert await breaker.allow_request() is True


async def test_half_open_probe_failure_reopens(monkeypatch):
    breaker = _fresh_breaker(monkeypatch, threshold=1, cooldown=60)
    await breaker.record_failure()
    breaker.opened_at -= 61
    await breaker.allow_request()
    await breaker.record_failure()  # 探测失败
    assert breaker.state == STATE_OPEN
    assert await breaker.allow_request() is False


def test_singleton_per_model_name():
    name = f"singleton-model-{uuid.uuid4().hex[:8]}"
    first = get_circuit_breaker(name)
    second = get_circuit_breaker(name)
    assert first is second
