"""
Redis 防护组件：并发（击穿）/ 雪崩两类问题的通用防护工具

提供三件套：
- RedisLock：基于 SET NX PX + Lua 脚本安全释放的分布式锁（跨进程互斥）
- singleflight_get_or_build：缓存 miss 时互斥重建，防止热点 key 失效瞬间
  并发请求全部打到下游（击穿保护）
- jittered_ttl：TTL 随机抖动，避免大量键在同一时刻集体过期（雪崩保护）
- get_shared_client：带 socket 超时与健康检查的共享客户端；Redis 不可用
  时返回 None，所有防护自动退化为直通（不阻断业务主链路）

组件可在无 Redis 的环境中安全 import 与调用（懒连接）。
"""
import json
import os
import random
import time
import uuid
from typing import Any, Callable, Optional

from loguru import logger


# ----------------------------------------------------------------------
# 共享客户端
# ----------------------------------------------------------------------

_shared_client: Any = None
_shared_client_resolved: bool = False


def _redis_settings() -> dict:
    """从环境变量解析连接参数（调用时读取，便于测试覆盖）"""
    try:
        timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
    except ValueError:
        timeout = 2.0
    try:
        port = int(os.getenv("REDIS_PORT", "6379"))
    except ValueError:
        port = 6379
    try:
        db = int(os.getenv("REDIS_DB", "0"))
    except ValueError:
        db = 0
    return {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": port,
        "db": db,
        # 超时下限 0.1s：无超时的客户端在 Redis 挂起时会永久阻塞请求
        "timeout": max(0.1, timeout),
    }


def get_shared_client() -> Optional[Any]:
    """
    返回进程级共享 Redis 客户端；不可用时返回 None（防护退化为直通）。

    首次调用时连接并 ping 验证，结果缓存；与短期记忆的后端选择无关，
    仅用于检索缓存 / 分布式锁等增强能力。
    """
    global _shared_client, _shared_client_resolved
    if _shared_client_resolved:
        return _shared_client
    _shared_client_resolved = True
    try:
        import redis
        settings = _redis_settings()
        client = redis.Redis(
            host=settings["host"],
            port=settings["port"],
            db=settings["db"],
            decode_responses=True,
            socket_connect_timeout=settings["timeout"],
            socket_timeout=settings["timeout"],
            health_check_interval=30,
        )
        client.ping()
        _shared_client = client
        logger.info("Redis 防护组件连接成功（检索缓存/分布式锁可用）")
    except Exception as e:
        _shared_client = None
        logger.info(f"Redis 不可用，击穿/雪崩防护降级为直通: {e}")
    return _shared_client


def reset_shared_client() -> None:
    """重置共享客户端缓存状态（测试用）"""
    global _shared_client, _shared_client_resolved
    _shared_client = None
    _shared_client_resolved = False


# ----------------------------------------------------------------------
# TTL 抖动（雪崩保护）
# ----------------------------------------------------------------------

def jittered_ttl(base_ttl: int, jitter_ratio: float = 0.1) -> int:
    """
    在基础 TTL 上叠加 [0, base*jitter_ratio] 的随机抖动。

    批量写入的键即使同时创建也不会同时过期，避免过期瞬间的
    集中回源/回收压力（缓存雪崩）。base<=0 或 ratio<=0 时原样返回。
    """
    if base_ttl is None:
        return 0
    if base_ttl <= 0 or jitter_ratio <= 0:
        return base_ttl
    return int(base_ttl + random.uniform(0, base_ttl * jitter_ratio))


# ----------------------------------------------------------------------
# 分布式锁（并发保护）
# ----------------------------------------------------------------------

class RedisLock:
    """
    轻量分布式锁：SET key token NX PX + Lua 脚本 compare-and-delete 释放。

    - token 为随机值，只有持锁者能释放，防止误解删他人的锁
    - 锁自带 TTL，持锁进程崩溃后自动过期，不会死锁
    - 非可重入；release 失败无需处理（等 TTL 自然过期）
    """

    _RELEASE_SCRIPT = (
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) "
        "else return 0 end"
    )

    def __init__(self, client: Any, key: str, ttl_ms: int = 10000):
        self._client = client
        self._key = key
        self._ttl_ms = max(1000, int(ttl_ms))
        self._token = uuid.uuid4().hex

    def acquire(self, wait_timeout: float = 0.0, retry_interval: float = 0.05) -> bool:
        """
        尝试获取锁。

        Args:
            wait_timeout: 等待时长（秒）；0 表示只尝试一次立即返回
            retry_interval: 等待期间的重试间隔

        Returns:
            是否成功持有锁
        """
        deadline = time.monotonic() + max(0.0, wait_timeout)
        while True:
            if self._client.set(self._key, self._token, nx=True, px=self._ttl_ms):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.01, retry_interval))

    def release(self) -> bool:
        """释放锁（仅当锁仍属于自己时删除）；失败靠 TTL 兜底"""
        try:
            return bool(self._client.eval(self._RELEASE_SCRIPT, 1, self._key, self._token))
        except Exception as e:
            logger.debug(f"释放分布式锁失败（将等待 TTL 过期）: {self._key}, {e}")
            return False


# ----------------------------------------------------------------------
# singleflight 互斥重建（击穿保护）
# ----------------------------------------------------------------------

def singleflight_get_or_build(
    client: Any,
    key: str,
    builder: Callable[[], Any],
    ttl_seconds: int,
    jitter_ratio: float = 0.1,
    wait_timeout: float = 2.0,
    poll_interval: float = 0.05,
    lock_ttl_ms: int = 30000,
) -> Any:
    """
    带互斥重建的缓存读取：

    1. 命中缓存直接返回；
    2. 未命中时通过分布式锁保证同一 key 只有一个调用方执行 builder，
       其余调用方短暂等待后读取重建结果；
    3. 等待超时的调用方直接执行 builder 兜底（不回写，避免与赢家竞争）。

    Args:
        client: Redis 客户端；None 时直通 builder（防护关闭）
        key: 缓存键
        builder: 重建函数，返回 JSON 可序列化的值；返回 None 视为不可缓存
        ttl_seconds: 缓存 TTL（会附加随机抖动）
        wait_timeout: 未抢到重建权时的等待时长

    Returns:
        缓存值或 builder 的结果
    """
    if client is None:
        return builder()

    cached = client.get(key)
    if cached is not None:
        try:
            return json.loads(cached)
        except (TypeError, ValueError):
            logger.warning(f"缓存值损坏，忽略并重建: {key}")

    lock = RedisLock(client, f"lock:{key}", ttl_ms=lock_ttl_ms)
    if lock.acquire():
        try:
            cached = client.get(key)  # 双重检查：排队期间可能已被重建
            if cached is not None:
                try:
                    return json.loads(cached)
                except (TypeError, ValueError):
                    pass
            value = builder()
            if value is not None:
                client.set(
                    key,
                    json.dumps(value, ensure_ascii=False),
                    ex=jittered_ttl(max(1, int(ttl_seconds)), jitter_ratio),
                )
            return value
        finally:
            lock.release()

    # 未抢到重建权：轮询等待赢家写入
    deadline = time.monotonic() + max(0.0, wait_timeout)
    while time.monotonic() < deadline:
        time.sleep(max(0.01, poll_interval))
        cached = client.get(key)
        if cached is not None:
            try:
                return json.loads(cached)
            except (TypeError, ValueError):
                break
    logger.debug(f"singleflight 等待超时，直接构建兜底（不回写缓存）: {key}")
    return builder()
