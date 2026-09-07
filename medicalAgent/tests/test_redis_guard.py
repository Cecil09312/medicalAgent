"""Redis 防护组件单元测试：分布式锁 / singleflight 互斥重建 / TTL 抖动 / 共享客户端降级"""
import threading
import time

import pytest

from core.redis_guard import (
    RedisLock,
    get_shared_client,
    jittered_ttl,
    reset_shared_client,
    singleflight_get_or_build,
)
from fake_redis import FakeRedis


@pytest.fixture(autouse=True)
def _reset_shared_client():
    """每个用例前后重置共享客户端缓存，避免跨用例污染"""
    reset_shared_client()
    yield
    reset_shared_client()


class TestJitteredTtl:
    def test_within_bounds(self):
        for _ in range(200):
            ttl = jittered_ttl(100, 0.1)
            assert 100 <= ttl <= 110

    def test_zero_ratio_returns_base(self):
        assert jittered_ttl(3600, 0.0) == 3600

    def test_non_positive_base_passthrough(self):
        assert jittered_ttl(0, 0.1) == 0
        assert jittered_ttl(-1, 0.1) == -1

    def test_none_base(self):
        assert jittered_ttl(None, 0.1) == 0


class TestRedisLock:
    def test_acquire_and_release(self):
        fake = FakeRedis()
        lock = RedisLock(fake, "lock:k1")
        assert lock.acquire() is True
        assert fake.store["lock:k1"] == lock._token
        assert lock.release() is True
        assert "lock:k1" not in fake.store

    def test_mutual_exclusion_and_wrong_token_release(self):
        fake = FakeRedis()
        lock1 = RedisLock(fake, "lock:k1")
        lock2 = RedisLock(fake, "lock:k1")
        assert lock1.acquire() is True
        # 互斥：第二个锁立即抢失败
        assert lock2.acquire(wait_timeout=0.05, retry_interval=0.01) is False
        # token 安全：锁值被换成他人 token 后，原持有者无法误删
        fake.store["lock:k1"] = "someone-else-token"
        assert lock1.release() is False
        assert fake.store["lock:k1"] == "someone-else-token"

    def test_acquire_waits_until_released(self):
        fake = FakeRedis()
        lock1 = RedisLock(fake, "lock:k1")
        lock2 = RedisLock(fake, "lock:k1")
        assert lock1.acquire() is True
        result = {}

        def waiter():
            result["got"] = lock2.acquire(wait_timeout=2.0, retry_interval=0.02)

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.1)
        lock1.release()
        thread.join(timeout=3)
        assert result["got"] is True


class TestSingleflight:
    def test_hit_after_build(self):
        fake = FakeRedis()
        calls = []

        def builder():
            calls.append(1)
            return [{"text": "结果"}]

        v1 = singleflight_get_or_build(fake, "c:k", builder, ttl_seconds=60)
        v2 = singleflight_get_or_build(fake, "c:k", builder, ttl_seconds=60)
        assert v1 == v2 == [{"text": "结果"}]
        assert len(calls) == 1
        # TTL 含抖动：实际写入值 >= 基础 TTL
        assert 60 <= fake.ex_seen["c:k"] <= 66

    def test_concurrent_calls_build_only_once(self):
        fake = FakeRedis()
        calls = []
        calls_lock = threading.Lock()

        def builder():
            with calls_lock:
                calls.append(1)
            time.sleep(0.1)  # 模拟昂贵的下游查询
            return {"v": 42}

        barrier = threading.Barrier(6)
        results = [None] * 6

        def worker(i):
            barrier.wait()
            results[i] = singleflight_get_or_build(fake, "c:k", builder, ttl_seconds=60)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert all(r == {"v": 42} for r in results)
        assert len(calls) == 1  # 击穿保护：并发 miss 只触发一次重建

    def test_lock_held_falls_back_to_direct_build(self):
        fake = FakeRedis()
        holder = RedisLock(fake, "lock:c:k")
        assert holder.acquire() is True
        calls = []

        def builder():
            calls.append(1)
            return {"direct": True}

        value = singleflight_get_or_build(
            fake, "c:k", builder, ttl_seconds=60,
            wait_timeout=0.1, poll_interval=0.02,
        )
        assert value == {"direct": True}
        assert len(calls) == 1
        assert "c:k" not in fake.store  # 兜底路径不回写缓存，避免与赢家竞争
        holder.release()

    def test_none_client_passes_through(self):
        calls = []

        def builder():
            calls.append(1)
            return "raw"

        assert singleflight_get_or_build(None, "c:k", builder, ttl_seconds=60) == "raw"
        assert len(calls) == 1


class TestSharedClient:
    def test_unreachable_redis_returns_none(self, monkeypatch):
        """Redis 不可用时共享客户端为 None（防护降级为直通），且结果被缓存"""
        pytest.importorskip("redis")
        monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
        monkeypatch.setenv("REDIS_PORT", "1")
        monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "0.2")
        assert get_shared_client() is None
        assert get_shared_client() is None
