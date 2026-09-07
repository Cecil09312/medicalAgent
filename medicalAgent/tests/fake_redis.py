"""测试用 Redis 客户端桩：覆盖短期记忆/防护组件用到的命令子集，线程安全"""
import fnmatch
import threading


class FakeRedis:
    """
    内存版 Redis 桩。

    - STRING 存 self.store，HASH 存 self.hashes
    - self.ex_seen 记录每个键最近一次设置的 TTL（set 的 ex / expire）
    - self.unlinked 记录 UNLINK 过的键（验证异步删除语义）
    - eval 仅实现 RedisLock 释放脚本的 compare-and-delete 语义
    """

    def __init__(self):
        self.store = {}
        self.hashes = {}
        self.ex_seen = {}
        self.unlinked = []
        self._guard = threading.RLock()

    # ---------------- 连接 ----------------
    def ping(self):
        return True

    # ---------------- STRING ----------------
    def set(self, key, value, ex=None, nx=False, px=None):
        with self._guard:
            if nx and key in self.store:
                return None
            self.store[key] = value
            if ex is not None:
                self.ex_seen[key] = ex
            return True

    def get(self, key):
        with self._guard:
            return self.store.get(key)

    def incr(self, key, amount=1):
        with self._guard:
            value = int(self.store.get(key, "0")) + amount
            self.store[key] = str(value)
            return value

    def delete(self, *keys):
        with self._guard:
            removed = 0
            for k in keys:
                if k in self.store or k in self.hashes:
                    self.store.pop(k, None)
                    self.hashes.pop(k, None)
                    removed += 1
            return removed

    def unlink(self, *keys):
        with self._guard:
            self.unlinked.extend(k for k in keys if k in self.store or k in self.hashes)
            return self.delete(*keys)

    def keys(self, pattern):
        with self._guard:
            all_keys = list(self.store) + list(self.hashes)
            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    def scan(self, cursor=0, match=None, count=None):
        """简化实现：单批返回全部匹配键（游标语义由真实验证）"""
        with self._guard:
            return 0, self.keys(match or "*")

    def expire(self, key, seconds):
        with self._guard:
            if key in self.store or key in self.hashes:
                self.ex_seen[key] = seconds
                return True
            return False

    def exists(self, *keys):
        with self._guard:
            return sum(1 for k in keys if k in self.store or k in self.hashes)

    # ---------------- HASH ----------------
    def hset(self, key, field=None, value=None, mapping=None):
        with self._guard:
            h = self.hashes.setdefault(key, {})
            added = 0
            if mapping:
                for f, v in mapping.items():
                    if f not in h:
                        added += 1
                    h[f] = v if isinstance(v, str) else str(v)
            if field is not None:
                if field not in h:
                    added += 1
                h[field] = value
            return added

    def hget(self, key, field):
        with self._guard:
            return self.hashes.get(key, {}).get(field)

    def hmget(self, key, fields):
        with self._guard:
            h = self.hashes.get(key, {})
            return [h.get(f) for f in fields]

    def hgetall(self, key):
        with self._guard:
            return dict(self.hashes.get(key, {}))

    def hincrby(self, key, field, amount=1):
        with self._guard:
            h = self.hashes.setdefault(key, {})
            value = int(h.get(field, "0")) + amount
            h[field] = str(value)
            return value

    def hdel(self, key, *fields):
        with self._guard:
            h = self.hashes.get(key, {})
            removed = 0
            for f in fields:
                if h.pop(f, None) is not None:
                    removed += 1
            return removed

    # ---------------- 脚本 ----------------
    def eval(self, script, numkeys, *args):
        """仅支持 RedisLock 释放脚本的 compare-and-delete"""
        with self._guard:
            key, token = args[0], args[1]
            if self.store.get(key) == token:
                del self.store[key]
                return 1
            return 0
