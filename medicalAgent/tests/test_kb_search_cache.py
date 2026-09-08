"""知识库检索缓存单元测试：缓存命中、参数隔离、世代失效、Redis 不可用直通
（object.__new__ 绕过 __init__，不加载嵌入模型/Milvus；环境缺少 pymilvus /
sentence_transformers 时以桩模块代替——被测的缓存包装层不依赖它们）"""
import sys
import types

import pytest

from fake_redis import FakeRedis


def _ensure_importable(name: str, attrs: dict) -> None:
    """真实包存在则用真实包；缺失时装载桩模块（仅满足 import 语句）"""
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module


_ensure_importable("pymilvus", {
    "MilvusClient": object,
    "CollectionSchema": object,
    "FieldSchema": object,
    "DataType": types.SimpleNamespace(
        VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR", INT64="INT64"
    ),
})
_ensure_importable("sentence_transformers", {"SentenceTransformer": object})

import knowledge.milvus_kb as kb_mod
from knowledge.milvus_kb import MedicalKnowledgeBase


@pytest.fixture(autouse=True)
def _cache_defaults(monkeypatch):
    monkeypatch.setattr(kb_mod, "KB_SEARCH_CACHE_ENABLED", True)
    monkeypatch.setattr(kb_mod, "KB_SEARCH_CACHE_TTL", 300)
    monkeypatch.setattr(kb_mod, "RERANK_ENABLED", False)
    monkeypatch.setattr(kb_mod, "RERANK_CANDIDATE_K", 15)


def _make_kb(calls):
    """构造仅含缓存包装层的知识库实例：_search_uncached 记录调用次数"""
    kb = object.__new__(MedicalKnowledgeBase)

    def fake_uncached(query, top_k, doc_type=None):
        calls.append({"query": query, "top_k": top_k, "doc_type": doc_type})
        return [{
            "text": f"结果-{query}", "score": 0.9,
            "doc_type": "general", "source": "测试来源", "page": None,
        }]

    kb._search_uncached = fake_uncached
    return kb


class TestSearchCache:
    def test_second_identical_query_hits_cache(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(kb_mod, "get_shared_client", lambda: fake)
        calls = []
        kb = _make_kb(calls)

        r1 = kb.search("高血压")
        r2 = kb.search("高血压")
        assert r1 == r2 and r1[0]["text"] == "结果-高血压"
        assert len(calls) == 1  # 第二次命中缓存，不触发底层检索

    def test_different_params_cached_separately(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(kb_mod, "get_shared_client", lambda: fake)
        calls = []
        kb = _make_kb(calls)

        kb.search("高血压")
        kb.search("高血压", top_k=3)
        kb.search("高血压", doc_type="clinical_guideline")
        assert len(calls) == 3  # top_k / doc_type 不同 → 指纹不同 → 分别检索
        kb.search("高血压", top_k=3)  # 与第二次相同 → 命中
        assert len(calls) == 3

    def test_generation_bump_invalidates_cache(self, monkeypatch):
        """知识库内容变更（add_documents/delete）→ 世代号 INCR → 旧缓存失效"""
        fake = FakeRedis()
        monkeypatch.setattr(kb_mod, "get_shared_client", lambda: fake)
        calls = []
        kb = _make_kb(calls)

        kb.search("糖尿病")
        assert len(calls) == 1
        kb_mod._invalidate_search_cache()
        assert fake.get("kb:search:gen") == "1"
        kb.search("糖尿病")
        assert len(calls) == 2  # 世代变化 → 缓存键变化 → 重新检索

    def test_ttl_written_with_jitter(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(kb_mod, "get_shared_client", lambda: fake)
        kb = _make_kb([])

        kb.search("高血压")
        cache_keys = [k for k in fake.store if k.startswith("kb:search:")]
        assert len(cache_keys) == 1
        assert 300 <= fake.ex_seen[cache_keys[0]] <= 330  # 基础 TTL + 10% 抖动

    def test_no_redis_passes_through(self, monkeypatch):
        monkeypatch.setattr(kb_mod, "get_shared_client", lambda: None)
        calls = []
        kb = _make_kb(calls)

        kb.search("高血压")
        kb.search("高血压")
        assert len(calls) == 2  # Redis 不可用：每次直通检索，行为与无缓存一致

    def test_cache_disabled_passes_through(self, monkeypatch):
        fake = FakeRedis()
        monkeypatch.setattr(kb_mod, "get_shared_client", lambda: fake)
        monkeypatch.setattr(kb_mod, "KB_SEARCH_CACHE_ENABLED", False)
        calls = []
        kb = _make_kb(calls)

        kb.search("高血压")
        kb.search("高血压")
        assert len(calls) == 2
        assert not [k for k in fake.store if k.startswith("kb:search:")]

    def test_large_top_k_not_cached(self, monkeypatch):
        """top_k 超限的结果集过大，不缓存（大 key 保护）"""
        fake = FakeRedis()
        monkeypatch.setattr(kb_mod, "get_shared_client", lambda: fake)
        calls = []
        kb = _make_kb(calls)

        kb.search("高血压", top_k=50)
        kb.search("高血压", top_k=50)
        assert len(calls) == 2
        assert not [k for k in fake.store if k.startswith("kb:search:")]
