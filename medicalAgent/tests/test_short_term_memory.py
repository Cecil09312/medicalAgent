"""短期记忆单元测试：后端选择（MEMORY_BACKEND）、Redis HASH 读写/清理、TTL 抖动、
并发追加不丢消息、大 key 上限/截断/压缩、旧格式懒迁移、熔断降级与恢复"""
import json
import threading

import pytest

import memory.short_term as stm_mod
from memory.short_term import ShortTermMemory
from fake_redis import FakeRedis


@pytest.fixture(autouse=True)
def _fresh_singleton(monkeypatch):
    """每个用例独立单例与默认配置（模块常量在调用时读取，可被 monkeypatch 覆盖）"""
    monkeypatch.setattr(stm_mod, "_DEFAULT_BACKEND", "memory")
    monkeypatch.setattr(stm_mod, "STM_SESSION_TTL_SECONDS", 3600)
    monkeypatch.setattr(stm_mod, "STM_MAX_MESSAGES", 50)
    monkeypatch.setattr(stm_mod, "STM_MAX_MESSAGE_CHARS", 32000)
    monkeypatch.setattr(stm_mod, "STM_COMPRESS_THRESHOLD", 8192)
    monkeypatch.setattr(stm_mod, "STM_TTL_JITTER_RATIO", 0.1)
    monkeypatch.setattr(stm_mod, "_REDIS_OUTAGE_SECONDS", 30.0)
    ShortTermMemory._instance = None
    ShortTermMemory._initialized = False
    yield
    ShortTermMemory._instance = None
    ShortTermMemory._initialized = False


def _make_redis_memory(fake: FakeRedis) -> ShortTermMemory:
    """构造使用 FakeRedis 的 redis 后端实例（绕过真实连接初始化）"""
    mem = ShortTermMemory()  # 默认 memory，跳过 _init_redis
    mem.backend = "redis"
    mem._redis_client = fake
    return mem


def _ttl_bounds(base: int = 3600) -> tuple:
    """TTL 抖动后的合法区间 [base, base*1.1]"""
    return base, int(base * 1.1)


class TestBackendSelection:
    def test_default_backend_is_memory(self):
        assert ShortTermMemory().backend == "memory"

    def test_env_backend_redis_resolved(self, monkeypatch):
        """无参构造读取 MEMORY_BACKEND=redis（主链路 chat_tab 即无参构造）"""
        monkeypatch.setattr(stm_mod, "_DEFAULT_BACKEND", "redis")
        # 跳过真实 Redis 连接（连接逻辑由 FakeRedis 用例与回退用例分别覆盖）
        monkeypatch.setattr(ShortTermMemory, "_init_redis", lambda self, cfg: None)
        mem = ShortTermMemory()
        assert mem.backend == "redis"

    def test_env_backend_redis_actually_inits_connection(self, monkeypatch):
        """回归：环境变量选 redis 时必须真正执行 _init_redis（而非仅改 backend 字符串）。

        历史缺陷：__init__ 曾用构造参数（None）判断是否初始化 Redis，导致环境变量
        切换后 client=None、数据静默落内存——用不可达端口验证 _init_redis 被调用
        且失败回退路径生效
        """
        monkeypatch.setattr(stm_mod, "_DEFAULT_BACKEND", "redis")
        monkeypatch.setattr(
            stm_mod, "REDIS_CONFIG", {"host": "127.0.0.1", "port": 1, "db": 0}
        )
        mem = ShortTermMemory()
        assert mem.backend == "memory"       # 连接失败 → 真正走过了 _init_redis 并回退
        assert mem._redis_client is None

    def test_explicit_backend_takes_precedence(self, monkeypatch):
        monkeypatch.setattr(stm_mod, "_DEFAULT_BACKEND", "redis")
        monkeypatch.setattr(ShortTermMemory, "_init_redis", lambda self, cfg: None)
        assert ShortTermMemory(backend="memory").backend == "memory"


class TestRedisBackend:
    def test_write_read_and_format(self):
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.add_message("s1", "user", "头痛怎么办")
        mem.add_message("s1", "assistant", "注意休息。仅供参考。")

        # HASH 结构：按消息拆 field，seq 原子计数
        h = fake.hashes["stm2:s1"]
        assert h["seq"] == "2"
        assert json.loads(h["msg:000001"])["content"] == "头痛怎么办"
        assert json.loads(h["msg:000002"])["content"] == "注意休息。仅供参考。"
        assert mem.get_history("s1") == [
            {"role": "user", "content": "头痛怎么办"},
            {"role": "assistant", "content": "注意休息。仅供参考。"},
        ]

    def test_ttl_applied_and_sliding(self):
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.add_message("s1", "user", "第一条")
        lo, hi = _ttl_bounds()
        assert lo <= fake.ex_seen["stm2:s1"] <= hi
        mem.add_message("s1", "user", "第二条")  # 每次写入刷新 TTL（滑动过期+抖动）
        assert lo <= fake.ex_seen["stm2:s1"] <= hi

    def test_ttl_zero_means_never_expire(self, monkeypatch):
        monkeypatch.setattr(stm_mod, "STM_SESSION_TTL_SECONDS", 0)
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.add_message("s1", "user", "内容")
        assert "stm2:s1" not in fake.ex_seen

    def test_cross_instance_persistence(self):
        """新实例（等价新进程）经同一 Redis 读到历史数据"""
        fake = FakeRedis()
        mem1 = _make_redis_memory(fake)
        mem1.add_message("s1", "user", "跨实例数据")

        ShortTermMemory._instance = None
        ShortTermMemory._initialized = False
        mem2 = _make_redis_memory(fake)
        assert mem2.get_history("s1") == [{"role": "user", "content": "跨实例数据"}]

    def test_clear_session_unlinks_key(self):
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.add_message("s1", "user", "内容")
        mem.clear_session("s1")
        assert "stm2:s1" not in fake.hashes
        assert "stm2:s1" in fake.unlinked  # 大 key 用 UNLINK 异步释放
        assert mem.get_history("s1") == []

    def test_session_listing(self):
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.add_message("a", "user", "1")
        mem.add_message("b", "user", "2")
        assert sorted(mem.get_session_ids()) == ["a", "b"]

    def test_limit_reads_only_recent_fields(self):
        """get_history(limit) 只读最近 N 个 field，不全量读大 key"""
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.auto_clean_enabled = False
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(stm_mod, "STM_MAX_MESSAGES", 0)
        try:
            for i in range(10):
                mem.add_message("s1", "user", f"m{i}")
            calls = []
            orig_hmget = fake.hmget

            def counting_hmget(key, fields):
                calls.append(list(fields))
                return orig_hmget(key, fields)

            fake.hmget = counting_hmget
            history = mem.get_history("s1", limit=3)
            assert [m["content"] for m in history] == ["m7", "m8", "m9"]
            assert calls == [["msg:000008", "msg:000009", "msg:000010"]]
        finally:
            monkeypatch.undo()


class TestConcurrency:
    def test_concurrent_adds_no_lost_messages(self, monkeypatch):
        """并发追加不丢消息：HINCRBY 原子取号，条数与序号连续无覆盖"""
        monkeypatch.setattr(stm_mod, "STM_MAX_MESSAGES", 0)  # 关闭条数上限，纯验证并发安全
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.auto_clean_enabled = False

        def worker(tid: int):
            for i in range(25):
                mem.add_message("s1", "user", f"t{tid}-m{i}")

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not any(t.is_alive() for t in threads)

        messages = mem.get_history("s1", openai_format=False)
        assert len(messages) == 200                       # 8 线程 × 25 条全部落库
        contents = {m["content"] for m in messages}
        assert len(contents) == 200                       # 无覆盖/丢失

        h = fake.hashes["stm2:s1"]
        assert h["seq"] == "200"
        field_nums = sorted(int(k[4:]) for k in h if k.startswith("msg:"))
        assert field_nums == list(range(1, 201))          # 序号连续（原子取号）


class TestBigKeyProtection:
    def test_hard_trim_caps_messages(self):
        """超过条数上限只保留最近 N 条（HDEL 最旧 field）"""
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.auto_clean_enabled = False
        for i in range(60):
            mem.add_message("s1", "user", f"m{i}")

        messages = mem.get_history("s1", openai_format=False)
        assert len(messages) == 50
        assert messages[0]["content"] == "m10"   # 最旧 10 条被裁剪
        assert messages[-1]["content"] == "m59"
        h = fake.hashes["stm2:s1"]
        assert h["first"] == "11"
        assert len([k for k in h if k.startswith("msg:")]) == 50

    def test_long_message_truncated(self, monkeypatch):
        """单条消息超过字符上限被截断并加标记"""
        monkeypatch.setattr(stm_mod, "STM_MAX_MESSAGE_CHARS", 100)
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.add_message("s1", "assistant", "x" * 500)

        content = mem.get_history("s1", openai_format=False)[0]["content"]
        assert content.startswith("x" * 100)
        assert "已截断" in content
        assert len(content) < 200

    def test_compression_roundtrip(self, monkeypatch):
        """超过压缩阈值的消息 zlib 压缩存储（z: 前缀），读取无损还原"""
        monkeypatch.setattr(stm_mod, "STM_COMPRESS_THRESHOLD", 64)
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        long_text = "长期医嘱内容监测报告" * 100

        mem.add_message("s1", "assistant", long_text)
        raw = fake.hashes["stm2:s1"]["msg:000001"]
        assert raw.startswith("z:")
        assert len(raw) < len(long_text)  # 压缩后更小
        assert mem.get_history("s1")[0]["content"] == long_text

    def test_small_message_not_compressed(self):
        fake = FakeRedis()
        mem = _make_redis_memory(fake)
        mem.add_message("s1", "user", "短消息")
        raw = fake.hashes["stm2:s1"]["msg:000001"]
        assert not raw.startswith("z:")
        assert json.loads(raw)["content"] == "短消息"


class TestLegacyMigration:
    @staticmethod
    def _legacy_payload(messages, session_id="s1"):
        return json.dumps({
            "session_id": session_id,
            "messages": messages,
            "created_at": 1.0,
            "updated_at": 2.0,
        }, ensure_ascii=False)

    def test_legacy_string_migrated_on_read(self):
        """读路径懒迁移：旧 stm: STRING → stm2: HASH，旧键 UNLINK"""
        fake = FakeRedis()
        fake.store["stm:s1"] = self._legacy_payload([
            {"role": "user", "content": "旧格式问题", "timestamp": 1.0},
            {"role": "assistant", "content": "旧格式回答", "timestamp": 2.0},
        ])
        mem = _make_redis_memory(fake)

        history = mem.get_history("s1")
        assert [m["content"] for m in history] == ["旧格式问题", "旧格式回答"]

        assert "stm:s1" not in fake.store
        assert "stm:s1" in fake.unlinked
        h = fake.hashes["stm2:s1"]
        assert h["seq"] == "2"
        lo, hi = _ttl_bounds()
        assert lo <= fake.ex_seen["stm2:s1"] <= hi  # 迁移后继承会话 TTL（含抖动）

    def test_legacy_string_migrated_before_append(self):
        """写路径同样先迁移：旧消息不会因新追加而丢失"""
        fake = FakeRedis()
        fake.store["stm:s1"] = self._legacy_payload(
            [{"role": "user", "content": "旧", "timestamp": 1.0}]
        )
        mem = _make_redis_memory(fake)

        mem.add_message("s1", "assistant", "新")
        messages = mem.get_history("s1", openai_format=False)
        assert [m["content"] for m in messages] == ["旧", "新"]
        assert fake.hashes["stm2:s1"]["seq"] == "2"

    def test_clear_removes_both_formats(self):
        fake = FakeRedis()
        fake.store["stm:s1"] = self._legacy_payload([])
        mem = _make_redis_memory(fake)
        mem.add_message("s1", "user", "新格式")

        mem.clear_session("s1")
        assert "stm:s1" not in fake.store
        assert "stm2:s1" not in fake.hashes
        assert mem.get_history("s1") == []


class TestMemoryBackendAndFallback:
    def test_memory_backend_roundtrip(self):
        mem = ShortTermMemory()
        mem.add_message("s1", "user", "内容")
        assert mem.get_history("s1") == [{"role": "user", "content": "内容"}]
        mem.clear_session("s1")
        assert mem.get_history("s1") == []

    def test_redis_unreachable_falls_back_to_memory(self):
        """连接不可达端口（port 1 本机必然拒绝）→ 回退 memory 后读写正常"""
        mem = ShortTermMemory(backend="redis", redis_config={"host": "127.0.0.1", "port": 1, "db": 0})
        assert mem.backend == "memory"
        mem.add_message("s1", "user", "回退后内容")
        assert mem.get_history("s1") == [{"role": "user", "content": "回退后内容"}]


class FlakyRedis:
    """包装 FakeRedis：fail=True 时所有命令抛连接异常（模拟运行期 Redis 故障）"""

    def __init__(self, inner: FakeRedis):
        self.inner = inner
        self.fail = False

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if callable(attr):
            def guarded(*args, **kwargs):
                if self.fail:
                    raise _redis_exceptions().ConnectionError("模拟连接断开")
                return attr(*args, **kwargs)
            return guarded
        return attr


def _redis_exceptions():
    return pytest.importorskip("redis").exceptions


class TestOutageRecovery:
    def test_outage_fallback_and_recovery(self, monkeypatch):
        """运行期连接故障：连续失败→熔断窗口走内存→恢复后回写降级数据"""
        pytest.importorskip("redis")
        monkeypatch.setattr(stm_mod, "_REDIS_OUTAGE_SECONDS", 0.0)  # 窗口 0：立即探活
        flaky = FlakyRedis(FakeRedis())
        mem = _make_redis_memory(flaky)
        mem.auto_clean_enabled = False

        mem.add_message("s1", "user", "m1")
        mem.add_message("s1", "user", "m2")
        assert len(mem.get_history("s1")) == 2

        # 连接断开：失败的追加逐次记账，达到阈值进入熔断（期间仍可读写）
        flaky.fail = True
        for i in range(3):
            mem.add_message("s1", "user", f"down{i}")
        assert mem._redis_in_outage is True

        mem.add_message("s1", "user", "during-outage")
        # 熔断期间 Redis 不可达：断前写入 Redis 的 m1/m2 暂时读不到，
        # 只能看到降级期间写入进程内存的消息（设计上的降级语义）
        assert [m["content"] for m in mem.get_history("s1", openai_format=False)] == [
            "down0", "down1", "down2", "during-outage",
        ]

        # 恢复：探活成功 → 降级期间的消息回写 Redis → 继续正常写入
        flaky.fail = False
        mem.add_message("s1", "user", "recovered")
        assert mem._redis_in_outage is False
        assert mem._sessions == {}
        messages = mem.get_history("s1", openai_format=False)
        assert [m["content"] for m in messages] == [
            "m1", "m2", "down0", "down1", "down2", "during-outage", "recovered",
        ]
