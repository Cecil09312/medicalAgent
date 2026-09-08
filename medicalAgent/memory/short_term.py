"""
短期记忆模块
支持内存和 Redis 两种存储后端
管理会话级别的对话历史

Redis 后端键结构（stm2:*，HASH，替代旧版单 STRING 大 key）：
- meta:  会话元数据 JSON（session_id/created_at/updated_at）
- seq:   消息序号计数器（HINCRBY 原子递增，只增不减）
- first: 当前最旧消息的序号（裁剪/重写后前移）
- msg:000001 ...: 单条消息 JSON（超过阈值时 zlib 压缩，z: 前缀标记）

三类防护：
- 大 key：按消息拆 field（追加 O(1)，不再全量重写整个会话）、单条长度
  上限、超阈值压缩、删除用 UNLINK 异步释放、遍历用 SCAN 替代 KEYS
- 并发：追加走 HINCRBY 原子取号（天然并发安全，无读-改-写丢更新）；
  重写类操作（自动清理/旧格式迁移）持进程内会话锁 + Redis 分布式锁
- 雪崩：每次 EXPIRE 附加随机抖动（批量键不同时过期）；客户端带 socket
  超时；运行期连续失败进入熔断窗口降级进程内存，窗口结束探活恢复并回写
"""
import sys
import os
import json
import time
import zlib
import base64
import functools
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

# 加载 .env
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

from core.redis_guard import RedisLock, jittered_ttl, safe_unlink
from .entropy_manager import MemoryEntropyManager

# redis 包可选：未安装时后端只能为 memory，运行期熔断守卫也不生效
try:
    import redis as _redis_module
    _REDIS_CONN_ERRORS: Optional[tuple] = (
        _redis_module.ConnectionError, _redis_module.TimeoutError
    )
except ImportError:
    _redis_module = None
    _REDIS_CONN_ERRORS = None

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# 存储后端选择（.env 的 MEMORY_BACKEND，缺省 memory）：redis 时短期记忆持久化到 Redis。
# 之前的版本 Redis 模式只能通过构造函数 backend="redis" 启用，主链路（chat_tab 等）
# 均为无参构造，导致 .env 里配了 Redis 也切不过去；现改为无参构造读取该环境变量
_DEFAULT_BACKEND = (os.getenv("MEMORY_BACKEND", "memory").strip().lower() or "memory")

# Redis 会话键的过期秒数（.env 的 STM_SESSION_TTL_SECONDS，默认 24 小时，0=永不过期）。
# 滑动过期：每次写入（追加消息）都会刷新 TTL，活跃会话不过期、闲置会话自动回收，
# 避免 stm:* 键无限累积（Web 端"重置会话"仅更换 session_id，不清理旧键）
_DEFAULT_SESSION_TTL_SECONDS = 86400
try:
    STM_SESSION_TTL_SECONDS: int = int(os.getenv("STM_SESSION_TTL_SECONDS", str(_DEFAULT_SESSION_TTL_SECONDS)))
except (TypeError, ValueError):
    STM_SESSION_TTL_SECONDS = _DEFAULT_SESSION_TTL_SECONDS

# 大 key 上限（调用时读取模块常量，可被测试/运维按需覆盖）
STM_MAX_MESSAGES = _env_int("STM_MAX_MESSAGES", 50)          # 单会话保留的最大消息条数（0=不限）
STM_MAX_MESSAGE_CHARS = _env_int("STM_MAX_MESSAGE_CHARS", 32000)  # 单条消息字符上限（0=不限）
STM_COMPRESS_THRESHOLD = _env_int("STM_COMPRESS_THRESHOLD", 8192)  # 单 field 压缩阈值字节（0=关闭）

# 雪崩防护：TTL 随机抖动比例（每次过期刷新时叠加 [0, base*ratio] 的随机量）
STM_TTL_JITTER_RATIO = _env_float("STM_TTL_JITTER_RATIO", 0.1)

# 连接健壮性：socket 超时（无超时的客户端在 Redis 挂起时会永久阻塞请求）
REDIS_SOCKET_TIMEOUT = _env_float("REDIS_SOCKET_TIMEOUT", 2.0)

# 运行期熔断：连续连接类失败达到阈值后进入降级窗口（走进程内存）
_REDIS_FAILURE_THRESHOLD = 3
_REDIS_OUTAGE_SECONDS = 30.0

# 新旧键前缀：stm:（旧版单 STRING）→ stm2:（HASH）
_KEY_PREFIX = "stm2:"
_LEGACY_KEY_PREFIX = "stm:"


def _session_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def _msg_field(seq: int) -> str:
    """消息 field 名：6 位零填充序号，保证字典序即时间序"""
    return f"msg:{seq:06d}"


def _sanitize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """规范化单条消息并应用单条长度上限（大 key 保护）"""
    role = str(message.get("role", "system"))
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    if STM_MAX_MESSAGE_CHARS > 0 and len(content) > STM_MAX_MESSAGE_CHARS:
        omitted = len(content) - STM_MAX_MESSAGE_CHARS
        content = content[:STM_MAX_MESSAGE_CHARS] + f"\n...[内容超长，已截断 {omitted} 字符]"
        logger.warning(
            f"单条消息超过 {STM_MAX_MESSAGE_CHARS} 字符上限，已截断 {omitted} 字符 "
            f"(role={role})"
        )
    return {
        "role": role,
        "content": content,
        "timestamp": message.get("timestamp", time.time()),
    }


def _encode_message(message: Dict[str, Any]) -> str:
    """单条消息序列化；超过压缩阈值时 zlib+base85（前缀 z: 标记）"""
    raw = json.dumps(message, ensure_ascii=False)
    if STM_COMPRESS_THRESHOLD > 0 and len(raw.encode("utf-8")) > STM_COMPRESS_THRESHOLD:
        packed = zlib.compress(raw.encode("utf-8"), 6)
        return "z:" + base64.b85encode(packed).decode("ascii")
    return raw


def _decode_message(value: str) -> Dict[str, Any]:
    if value.startswith("z:"):
        raw = zlib.decompress(base64.b85decode(value[2:].encode("ascii")))
        return json.loads(raw)
    return json.loads(value)


@dataclass
class ConversationHistory:
    """对话历史数据结构"""
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def add_message(self, role: str, content: str):
        """添加一条消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().timestamp(),
        })
        self.updated_at = datetime.now().timestamp()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """从字典反序列化"""
        history = cls(session_id=data["session_id"])
        history.messages = data.get("messages", [])
        history.created_at = data.get("created_at", datetime.now().timestamp())
        history.updated_at = data.get("updated_at", datetime.now().timestamp())
        return history


def _redis_conn_guarded(fn):
    """
    Redis 操作守卫：连接类异常记录熔断记账并返回 None（调用方走内存兜底）；
    其余异常照常抛出（数据/代码问题不应被静默吞掉）
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as exc:
            if _REDIS_CONN_ERRORS and isinstance(exc, _REDIS_CONN_ERRORS):
                self._record_redis_failure(exc)
                return None
            raise

    return wrapper


class ShortTermMemory:
    """
    短期记忆管理器（单例模式）

    支持两种存储后端：
    - memory: 内存存储（默认）
    - redis: Redis 持久化存储（stm2:* HASH 结构，见模块 docstring）

    集成 MemoryEntropyManager 进行自动清理。
    """

    _instance: Optional["ShortTermMemory"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        backend: Optional[str] = None,
        redis_config: Optional[Dict[str, Any]] = None,
        auto_clean_enabled: bool = True,
    ):
        """
        初始化短期记忆

        Args:
            backend: 存储后端 ("memory" 或 "redis")；None 时读取环境变量
                MEMORY_BACKEND（默认 "memory"），使 .env 配置可直接切换后端
            redis_config: Redis 配置（backend="redis" 时使用，缺省取 REDIS_HOST/PORT/DB）
            auto_clean_enabled: 是否启用自动清理
        """
        # 避免重复初始化（单例）
        if ShortTermMemory._initialized:
            return

        self.backend = (backend or _DEFAULT_BACKEND).strip().lower()
        self.session_ttl = STM_SESSION_TTL_SECONDS
        self.auto_clean_enabled = auto_clean_enabled
        self.entropy_manager = MemoryEntropyManager()

        # 内存存储（memory 后端的主存储；redis 后端熔断降级期间的兜底存储）
        self._sessions: Dict[str, ConversationHistory] = {}

        # 进程内会话锁注册表（追加串行化 / 重写类操作互斥）
        self._session_locks: Dict[str, threading.RLock] = {}
        self._locks_registry_guard = threading.Lock()

        # 运行期熔断状态
        self._redis_failure_streak = 0
        self._redis_in_outage = False
        self._redis_down_until = 0.0

        # Redis 存储（按解析后的后端判断，环境变量切换时同样初始化）
        self._redis_client = None
        if self.backend == "redis":
            self._init_redis(redis_config or REDIS_CONFIG)

        ShortTermMemory._initialized = True
        logger.info(f"ShortTermMemory 初始化完成 (backend={self.backend})")

    def _init_redis(self, config: Dict[str, Any]):
        """初始化 Redis 连接（带 socket 超时与健康检查）"""
        try:
            import redis
            self._redis_client = redis.Redis(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                db=config.get("db", 0),
                decode_responses=True,
                socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                health_check_interval=30,
            )
            self._redis_client.ping()
            logger.info("Redis 连接成功")
        except Exception as e:
            logger.warning(f"Redis 连接失败，回退到内存存储: {e}")
            self._redis_client = None
            self.backend = "memory"

    # ------------------------------------------------------------------
    # 可用性判断与运行期熔断
    # ------------------------------------------------------------------

    def _redis_active(self) -> bool:
        """Redis 是否可用（含熔断窗口判断；窗口结束时探活恢复并回写降级数据）"""
        if self._redis_client is None:
            return False
        if not self._redis_in_outage:
            return True
        if time.monotonic() < self._redis_down_until:
            return False
        # 熔断窗口结束：探活
        try:
            self._redis_client.ping()
        except Exception:
            self._redis_down_until = time.monotonic() + _REDIS_OUTAGE_SECONDS
            return False
        self._redis_in_outage = False
        self._redis_failure_streak = 0
        logger.info("Redis 恢复可用，短期记忆切回 Redis")
        self._flush_memory_sessions_to_redis()
        return True

    def _record_redis_failure(self, exc: Exception) -> None:
        """连接类失败记账：连续达到阈值后进入熔断窗口（降级进程内存）"""
        self._redis_failure_streak += 1
        if (not self._redis_in_outage
                and self._redis_failure_streak >= _REDIS_FAILURE_THRESHOLD):
            self._redis_in_outage = True
            self._redis_down_until = time.monotonic() + _REDIS_OUTAGE_SECONDS
            logger.warning(
                f"Redis 连续失败 {self._redis_failure_streak} 次，熔断 "
                f"{_REDIS_OUTAGE_SECONDS:.0f} 秒：短期记忆临时走进程内存"
                "（重启丢失，降级期间写入的历史可能不完整）"
            )

    def _flush_memory_sessions_to_redis(self) -> None:
        """熔断恢复后，把降级期间写入进程内存的会话追加回 Redis（尽力而为）"""
        if not self._sessions:
            return
        flushed = []
        failed = False
        for session_id, history in list(self._sessions.items()):
            ok = True
            for message in history.messages:
                if self._redis_append_raw(session_id, message) is None:
                    ok = False
                    failed = True
                    break
            if ok:
                flushed.append(session_id)
        for session_id in flushed:
            self._sessions.pop(session_id, None)
        if failed:
            logger.warning("部分降级会话回写 Redis 失败，保留在进程内存等待下次恢复")
        if flushed:
            logger.info(f"熔断降级期间的消息已回写 Redis（{len(flushed)} 个会话）")

    # ------------------------------------------------------------------
    # 会话锁（并发保护）
    # ------------------------------------------------------------------

    def _get_session_lock(self, session_id: str) -> threading.RLock:
        """获取进程内会话锁（可重入；注册表自身加锁保护）"""
        with self._locks_registry_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._session_locks[session_id] = lock
            return lock

    @contextmanager
    def _locked(self, session_id: str, redis_wait: float = 5.0):
        """
        重写类操作的双层锁：进程内会话锁 + Redis 分布式锁（跨进程互斥）。

        分布式锁繁忙时退化为仅进程内锁（操作幂等，最坏重复执行）。
        注意：内部调用的 _redis_trim/_redis_rewrite 等原语自身不加锁，
        加锁是调用方的责任，避免嵌套获取分布式锁。
        """
        with self._get_session_lock(session_id):
            redis_lock = None
            if self._redis_active():
                redis_lock = RedisLock(
                    self._redis_client, f"lock:{_session_key(session_id)}"
                )
                if not redis_lock.acquire(wait_timeout=redis_wait):
                    logger.debug(f"跨进程会话锁繁忙，退化为进程内锁: {session_id}")
                    redis_lock = None
            try:
                yield
            finally:
                if redis_lock is not None:
                    redis_lock.release()

    # ------------------------------------------------------------------
    # Redis 存储原语（连接异常→熔断记账→返回 None）
    # ------------------------------------------------------------------

    def _redis_refresh_ttl(self, key: str) -> None:
        """滑动过期 + 随机抖动（雪崩保护：批量键不同时失效）"""
        if self.session_ttl and self.session_ttl > 0:
            self._redis_client.expire(
                key, jittered_ttl(self.session_ttl, STM_TTL_JITTER_RATIO)
            )

    def _redis_range(self, key: str) -> Tuple[int, Optional[int]]:
        """返回 (first, seq)；键不存在时 seq 为 None"""
        seq_raw = self._redis_client.hget(key, "seq")
        if seq_raw is None:
            return 1, None
        first_raw = self._redis_client.hget(key, "first")
        return (int(first_raw) if first_raw else 1), int(seq_raw)

    @_redis_conn_guarded
    def _redis_touch(self, session_id: str) -> bool:
        """创建空会话（仅写 meta + TTL；seq 由首条消息的 HINCRBY 创建）"""
        key = _session_key(session_id)
        now = time.time()
        meta = {"session_id": session_id, "created_at": now, "updated_at": now}
        self._redis_client.hset(key, "meta", json.dumps(meta, ensure_ascii=False))
        self._redis_refresh_ttl(key)
        return True

    @_redis_conn_guarded
    def _redis_append_raw(self, session_id: str, message: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        原子追加一条消息：HINCRBY 取号 + HSET 单 field + 滑动 TTL。

        Returns:
            (seq, 当前消息条数)；连接失败时返回 None（调用方走内存兜底）
        """
        client = self._redis_client
        key = _session_key(session_id)
        seq = client.hincrby(key, "seq", 1)
        client.hset(key, _msg_field(seq), _encode_message(message))
        now = time.time()
        meta = {"session_id": session_id, "created_at": now, "updated_at": now}
        client.hset(key, "meta", json.dumps(meta, ensure_ascii=False))
        self._redis_refresh_ttl(key)
        first_raw = client.hget(key, "first")
        first = int(first_raw) if first_raw else 1
        return seq, seq - first + 1

    @_redis_conn_guarded
    def _redis_load(
        self, session_id: str, limit: Optional[int] = None
    ) -> Optional[ConversationHistory]:
        """
        加载会话；limit>0 时只取最近 N 条（避免全量读大 key）。

        键不存在时先尝试旧版 stm: 单 STRING 键的懒迁移。
        """
        client = self._redis_client
        key = _session_key(session_id)
        first, seq = self._redis_range(key)
        if seq is None:
            self._migrate_legacy_session(session_id)
            first, seq = self._redis_range(key)
            if seq is None:
                return None

        messages: List[Dict[str, Any]] = []
        if limit is not None and limit > 0 and (seq - first + 1) > limit:
            start = seq - limit + 1
            fields = [_msg_field(i) for i in range(start, seq + 1)]
            values = client.hmget(key, fields)
            messages = [_decode_message(v) for v in values if v is not None]
        else:
            pairs = client.hgetall(key)
            messages = [
                _decode_message(v) for k, v in sorted(pairs.items())
                if k.startswith("msg:")
            ]

        meta: Dict[str, Any] = {}
        meta_raw = client.hget(key, "meta")
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
            except ValueError:
                meta = {}

        history = ConversationHistory(session_id=session_id)
        history.messages = messages
        history.created_at = meta.get("created_at", history.created_at)
        history.updated_at = meta.get("updated_at", history.updated_at)
        return history

    @_redis_conn_guarded
    def _redis_ensure_migrated(self, session_id: str) -> None:
        """写路径前的迁移检查：新键无 seq 且旧键存在时执行懒迁移"""
        if self._redis_client.hget(_session_key(session_id), "seq") is not None:
            return
        self._migrate_legacy_session(session_id)

    @_redis_conn_guarded
    def _migrate_legacy_session(self, session_id: str) -> Optional[bool]:
        """
        旧版 stm: 单 STRING 键 → 新版 stm2: HASH（持双层锁 + 双重检查，幂等）。

        Returns:
            是否发生了迁移；旧键不存在返回 False；连接失败返回 None
        """
        client = self._redis_client
        old_key = f"{_LEGACY_KEY_PREFIX}{session_id}"
        payload = client.get(old_key)
        if not payload:
            return False

        with self._locked(session_id):
            # 双重检查：排队期间可能已被其他进程/线程迁移
            new_key = _session_key(session_id)
            if client.hget(new_key, "seq") is not None:
                safe_unlink(client, old_key)
                return True

            data = json.loads(payload)
            messages = list(data.get("messages", []))
            # 迁移时应用大 key 上限：只保留最近 N 条，超长单条截断
            if STM_MAX_MESSAGES > 0 and len(messages) > STM_MAX_MESSAGES:
                logger.info(
                    f"迁移会话 {session_id}: {len(messages)} → {STM_MAX_MESSAGES} 条（超出上限截断）"
                )
                messages = messages[-STM_MAX_MESSAGES:]
            messages = [_sanitize_message(m) for m in messages]

            now = time.time()
            meta = {
                "session_id": session_id,
                "created_at": data.get("created_at", now),
                "updated_at": data.get("updated_at", now),
            }
            # 逐字段 hset（兼容旧版 Redis：HSET 多字段形式 4.0 才引入）
            client.hset(new_key, "meta", json.dumps(meta, ensure_ascii=False))
            client.hset(new_key, "first", "1")
            for i, message in enumerate(messages, start=1):
                client.hset(new_key, _msg_field(i), _encode_message(message))
            client.hset(new_key, "seq", str(len(messages)))
            self._redis_refresh_ttl(new_key)
            safe_unlink(client, old_key)
            logger.info(
                f"会话 {session_id} 已由旧格式（stm: STRING）迁移为新格式"
                f"（stm2: HASH），共 {len(messages)} 条消息"
            )
            return True

    @_redis_conn_guarded
    def _redis_trim(self, session_id: str) -> None:
        """
        硬上限裁剪：只保留最近 STM_MAX_MESSAGES 条（HDEL 最旧 field，幂等）。

        调用方需持会话锁（见 _locked 的说明）。
        """
        client = self._redis_client
        key = _session_key(session_id)
        first, seq = self._redis_range(key)
        if seq is None or STM_MAX_MESSAGES <= 0:
            return
        excess = (seq - first + 1) - STM_MAX_MESSAGES
        if excess <= 0:
            return
        fields = [_msg_field(i) for i in range(first, first + excess)]
        client.hdel(key, *fields)
        client.hset(key, "first", str(first + excess))
        self._redis_refresh_ttl(key)
        logger.debug(
            f"会话 {session_id} 裁剪 {excess} 条旧消息（上限 {STM_MAX_MESSAGES} 条）"
        )

    @_redis_conn_guarded
    def _redis_rewrite(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        整表重写（熵清理后）：新消息以续号写入，再删除旧区间。

        并发安全性：并发追加经 HINCRBY 取得的序号必然大于重写前读到的
        old_seq，不会落入删除区间 [first, old_seq]；重写消息占用其后的
        新序号，与并发追加天然错开。调用方需持会话锁。
        """
        client = self._redis_client
        key = _session_key(session_id)
        first, old_seq = self._redis_range(key)
        if old_seq is None:
            return
        for message in messages:
            seq = client.hincrby(key, "seq", 1)
            client.hset(key, _msg_field(seq), _encode_message(message))
        if old_seq >= first:
            client.hdel(key, *[_msg_field(i) for i in range(first, old_seq + 1)])
        client.hset(key, "first", str(old_seq + 1))
        now = time.time()
        meta = {"session_id": session_id, "created_at": now, "updated_at": now}
        client.hset(key, "meta", json.dumps(meta, ensure_ascii=False))
        self._redis_refresh_ttl(key)

    @_redis_conn_guarded
    def _redis_clear(self, session_id: str) -> None:
        """异步删除大 key（UNLINK，旧版 Redis 回退 DEL）；同时清理可能的旧格式键"""
        safe_unlink(
            self._redis_client,
            _session_key(session_id), f"{_LEGACY_KEY_PREFIX}{session_id}",
        )

    @_redis_conn_guarded
    def _redis_scan_session_ids(self) -> Optional[List[str]]:
        """SCAN 游标遍历会话键（替代阻塞的 KEYS）"""
        ids: List[str] = []
        cursor = 0
        while True:
            cursor, keys = self._redis_client.scan(
                cursor=cursor, match=f"{_KEY_PREFIX}*", count=100
            )
            ids.extend(k[len(_KEY_PREFIX):] for k in keys)
            if not cursor or int(cursor) == 0:
                break
        return ids

    # ------------------------------------------------------------------
    # 内存路径原语
    # ------------------------------------------------------------------

    def _memory_append(self, session_id: str, message: Dict[str, Any]) -> None:
        """内存追加（memory 后端主路径 / redis 熔断降级兜底）"""
        history = self._sessions.get(session_id)
        if history is None:
            history = ConversationHistory(session_id=session_id)
            self._sessions[session_id] = history
        history.messages.append(message)
        history.updated_at = message.get("timestamp", time.time())

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def create_session(self, session_id: str) -> ConversationHistory:
        """
        创建新会话

        Args:
            session_id: 会话 ID

        Returns:
            创建的 ConversationHistory 对象
        """
        history = ConversationHistory(session_id=session_id)

        if self._redis_active():
            if self._redis_touch(session_id) is None:
                # Redis 写入失败（熔断记账）：内存兜底
                self._sessions[session_id] = history
        else:
            self._sessions[session_id] = history

        logger.debug(f"创建会话: {session_id}")
        return history

    def _load_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> Optional[ConversationHistory]:
        """
        加载会话历史（内部方法）。

        Redis 模式 limit>0 时仅取最近 N 条 field，避免全量读取大 key。
        """
        if self._redis_active():
            return self._redis_load(session_id, limit=limit)
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str):
        """
        清除指定会话

        Args:
            session_id: 会话 ID
        """
        if self._redis_active():
            self._redis_clear(session_id)
        self._sessions.pop(session_id, None)
        with self._locks_registry_guard:
            self._session_locks.pop(session_id, None)

        logger.info(f"清除会话: {session_id}")

    # ------------------------------------------------------------------
    # 消息操作
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str):
        """
        向会话添加消息

        如果会话不存在，会自动创建。Redis 模式下走 HINCRBY 原子追加
        （无读-改-写，并发安全），失败时熔断降级到进程内存。

        Args:
            session_id: 会话 ID
            role: 消息角色 (user / assistant / system / tool)
            content: 消息内容
        """
        message = _sanitize_message(
            {"role": role, "content": content, "timestamp": time.time()}
        )

        if self._redis_active():
            self._redis_ensure_migrated(session_id)
            with self._get_session_lock(session_id):
                appended = self._redis_append_raw(session_id, message)
            if appended is None:
                self._memory_append(session_id, message)
                return
            _, count = appended
            if STM_MAX_MESSAGES > 0 and count > STM_MAX_MESSAGES:
                if self.auto_clean_enabled:
                    self._auto_clean_session(session_id)
                with self._locked(session_id):
                    self._redis_trim(session_id)
            return

        self._memory_append(session_id, message)
        if self.auto_clean_enabled:
            history = self._sessions.get(session_id)
            if history and len(history.messages) > STM_MAX_MESSAGES:
                self._auto_clean_session(session_id)

    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        获取最近 N 条消息

        Args:
            session_id: 会话 ID
            limit: 返回消息数量上限

        Returns:
            消息列表（按时间正序）
        """
        history = self._load_history(session_id, limit=limit)
        if history is None:
            return []

        return history.messages[-limit:]

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        openai_format: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        获取会话历史（OpenAI 格式）

        Args:
            session_id: 会话 ID
            limit: 返回消息数量上限（None = 全部）
            openai_format: 是否返回 OpenAI 格式（仅保留 role 和 content）

        Returns:
            消息列表
        """
        history = self._load_history(session_id, limit=limit)
        if history is None:
            return []

        messages = history.messages
        if limit:
            messages = messages[-limit:]

        if openai_format:
            return [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]

        return messages

    # ------------------------------------------------------------------
    # 自动清理
    # ------------------------------------------------------------------

    def _auto_clean_session(self, session_id: str):
        """对指定会话执行自动清理（熵管理器去重压缩；Redis 模式持双层锁重写）"""
        if self._redis_active():
            with self._locked(session_id):
                history = self._redis_load(session_id)
                if history is None or not history.messages:
                    return
                cleaned = self.entropy_manager.auto_clean(history.messages)
                if len(cleaned) == len(history.messages):
                    return  # 无变化，跳过重写
                self._redis_rewrite(session_id, cleaned)
                logger.info(
                    f"会话 {session_id} 自动清理: {len(history.messages)} → {len(cleaned)} 条消息"
                )
            return

        history = self._sessions.get(session_id)
        if history is None:
            return

        original_count = len(history.messages)
        cleaned = self.entropy_manager.auto_clean(history.messages)
        history.messages = cleaned

        logger.info(
            f"会话 {session_id} 自动清理: {original_count} → {len(cleaned)} 条消息"
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_session_ids(self) -> List[str]:
        """获取所有会话 ID（Redis 模式 SCAN 游标遍历）"""
        if self._redis_active():
            ids = self._redis_scan_session_ids()
            if ids is not None:
                return ids
        return list(self._sessions.keys())

    def get_session_count(self) -> int:
        """获取会话总数"""
        return len(self.get_session_ids())
