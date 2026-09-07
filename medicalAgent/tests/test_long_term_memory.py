"""长期记忆单元测试：本地中文检索（n-gram）、云端/本地结果合并去重、云端故障回退"""
import pytest

from memory.long_term import LongTermMemory, _extract_ngrams


class FakeMem0Client:
    """Mem0 云端客户端桩：返回固定搜索结果，可注入异常以测试回退"""

    def __init__(self, results=None, raise_on_search=False):
        self.results = results or []
        self.raise_on_search = raise_on_search
        self.search_calls = []

    def search(self, query, filters=None, limit=5):
        self.search_calls.append({"query": query, "filters": filters, "limit": limit})
        if self.raise_on_search:
            raise ConnectionError("mem0 cloud unreachable")
        return {"results": self.results[:limit]}

    def add(self, messages, user_id=None, metadata=None):
        return {"id": "fake-id"}


@pytest.fixture
def ltm(tmp_path):
    """本地模式长期记忆实例（无云端 Key，存储指向临时目录），预置两条记忆"""
    m = LongTermMemory(config={"api_key": "", "local_storage_dir": tmp_path / "long_term"})
    m.add_session_summary(
        summary="患者咨询血压偏高（150/95mmHg）是否需用药，属1级高血压，建议先改善生活方式。",
        session_id="s1",
    )
    m.add_session_summary(
        summary="患者咨询经常头痛怎么缓解，建议注意作息，考虑紧张性头痛。",
        session_id="s2",
    )
    return m


class TestExtractNgrams:
    def test_chinese_bigrams(self):
        # 中文按滑窗二元组切分，而非整句成一个"关键词"
        assert _extract_ngrams("我血压有点高") == {"我血", "血压", "压有", "有点", "点高"}

    def test_mixed_text(self):
        grams = _extract_ngrams("血压 150/95mmHg 偏高")
        assert "血压" in grams and "偏高" in grams
        assert "150" in grams and "95mmhg" in grams

    def test_single_char(self):
        assert _extract_ngrams("痛") == {"痛"}


class TestLocalSearch:
    def test_chinese_sentence_query(self, ltm):
        """回归：整句中文查询（无空格）必须能命中血压记忆——修复前返回 0 条"""
        results = ltm.search_similar_sessions("我血压有点高怎么办", top_k=5)
        assert any("血压" in r["summary"] for r in results)

    def test_ranking_relevant_first(self, ltm):
        results = ltm.search_similar_sessions("血压偏高要吃药吗", top_k=5)
        assert results and "血压" in results[0]["summary"]

    def test_irrelevant_query_returns_empty(self, ltm):
        assert ltm.search_similar_sessions("骨折后康复训练", top_k=5) == []

    def test_space_separated_keywords_still_work(self, ltm):
        results = ltm.search_similar_sessions("血压 偏高", top_k=5)
        assert any("血压" in r["summary"] for r in results)


class TestCloudLocalMerge:
    def test_cloud_and_local_merged(self, ltm):
        """开启云端后，本地降级期间写入的记忆仍应出现在检索结果中"""
        ltm._mem0_client = FakeMem0Client(results=[{"memory": "User consulted about glucose monitoring"}])
        ltm._use_mem0 = True
        results = ltm.search_similar_sessions("我血压有点高怎么办", top_k=5)
        texts = [r.get("memory") or r.get("summary") for r in results]
        assert any("glucose" in t for t in texts)  # 云端结果
        assert any("血压" in t for t in texts)     # 本地结果

    def test_dedupe_identical_text(self, ltm):
        """云端与本地文本相同的记忆只保留一份"""
        dup_text = "患者咨询血压偏高（150/95mmHg）是否需用药，属1级高血压，建议先改善生活方式。"
        ltm._mem0_client = FakeMem0Client(results=[{"memory": dup_text}])
        ltm._use_mem0 = True
        results = ltm.search_similar_sessions("血压偏高", top_k=5)
        texts = [(r.get("memory") or r.get("summary") or "").strip() for r in results]
        assert texts.count(dup_text) == 1

    def test_cloud_failure_falls_back_to_local(self, ltm):
        ltm._mem0_client = FakeMem0Client(raise_on_search=True)
        ltm._use_mem0 = True
        results = ltm.search_similar_sessions("我血压有点高怎么办", top_k=5)
        assert any("血压" in r["summary"] for r in results)

    def test_top_k_respected(self, ltm):
        ltm._mem0_client = FakeMem0Client(results=[{"memory": f"记忆{i}"} for i in range(5)])
        ltm._use_mem0 = True
        results = ltm.search_similar_sessions("血压偏高", top_k=3)
        assert len(results) == 3
