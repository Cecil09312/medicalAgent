"""知识库导入管道单元测试（不初始化嵌入模型/Milvus）"""
import pytest

from knowledge.importer import (
    KnowledgeImporter,
    file_hash,
    parse_document,
)


class TestParseDocument:
    def test_txt(self, tmp_path):
        path = tmp_path / "doc.txt"
        path.write_text("高血压患者应低盐饮食。", encoding="utf-8")
        assert parse_document(path) == ["高血压患者应低盐饮食。"]

    def test_md(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text("# 指南\n内容", encoding="utf-8")
        assert parse_document(path) == ["# 指南\n内容"]

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "doc.docx"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="不支持的文档格式"):
            parse_document(path)


class TestFileHash:
    def test_stable_and_sensitive(self, tmp_path):
        a1 = tmp_path / "a.txt"
        a2 = tmp_path / "b.txt"
        a1.write_text("same", encoding="utf-8")
        a2.write_text("same", encoding="utf-8")
        assert file_hash(a1) == file_hash(a2)

        a2.write_text("changed", encoding="utf-8")
        assert file_hash(a1) != file_hash(a2)


class TestSync:
    def test_all_disabled_sources_skipped_without_kb(self, tmp_path, monkeypatch):
        """清单条目全部 enabled=false → 全部跳过，且不初始化知识库（不加载嵌入模型）"""
        sources = tmp_path / "sources.yaml"
        sources.write_text(
            "sources:\n"
            "  - source_id: s1\n"
            "    path: knowledge/data/documents/20_guideline_hypertension.txt\n"
            "    enabled: false\n",
            encoding="utf-8",
        )
        importer = KnowledgeImporter(
            sources_path=sources, state_path=tmp_path / "state.json"
        )

        report = importer.sync()

        assert report == [{"source_id": "s1", "status": "skipped", "detail": "enabled=false"}]
        assert importer._kb is None

    def test_missing_file_fails(self, tmp_path):
        sources = tmp_path / "sources.yaml"
        sources.write_text(
            "sources:\n"
            "  - source_id: ghost\n"
            "    path: not/exists.txt\n"
            "    enabled: true\n",
            encoding="utf-8",
        )
        importer = KnowledgeImporter(
            sources_path=sources, state_path=tmp_path / "state.json"
        )
        report = importer.sync()
        assert report[0]["status"] == "failed"
        assert "文件不存在" in report[0]["detail"]

    def test_missing_source_id_fails(self, tmp_path):
        sources = tmp_path / "sources.yaml"
        sources.write_text("sources:\n  - path: whatever.txt\n    enabled: true\n", encoding="utf-8")
        importer = KnowledgeImporter(
            sources_path=sources, state_path=tmp_path / "state.json"
        )
        report = importer.sync()
        assert report[0]["status"] == "failed"
        assert "source_id" in report[0]["detail"]

    def test_unchanged_content_skipped(self, tmp_path, monkeypatch):
        """同一文件二次同步：hash 未变化 → 跳过（用假 KB 验证增量去重逻辑）"""
        doc = tmp_path / "doc.txt"
        doc.write_text("糖尿病饮食建议内容。", encoding="utf-8")
        sources = tmp_path / "sources.yaml"
        sources.write_text(
            "sources:\n"
            "  - source_id: s1\n"
            f"    path: {doc.as_posix()}\n"
            "    doc_type: general\n"
            "    enabled: true\n",
            encoding="utf-8",
        )
        importer = KnowledgeImporter(
            sources_path=sources, state_path=tmp_path / "state.json"
        )

        class _FakeKB:
            def __init__(self):
                self.deleted = []
                self.ingested = []

            def count_documents(self):
                return 1

            def delete_by_source(self, source):
                self.deleted.append(source)

            def add_documents(self, texts, doc_type="general", source="", pages=None):
                self.ingested.append((source, len(texts)))
                return 3

            def search_by_text(self, query, top_k=5):
                return [{"source": "s1 (v1.0)", "text": query, "score": 1.0}]

        fake_kb = _FakeKB()
        monkeypatch.setattr(
            KnowledgeImporter, "_get_kb", lambda self: fake_kb, raising=True
        )

        first = importer.sync()
        assert first[0]["status"] == "synced"
        assert first[0]["detail"].startswith("入库 3 块")

        # 第二次同步：hash 相同 → 跳过，不再删除/入库
        second = importer.sync()
        assert second[0]["status"] == "skipped"
        assert fake_kb.ingested == [("s1 (v1.0)", 1)]
        # 仅首次同步删除（覆盖原始 id 与版本标签），第二次未触发
        assert fake_kb.deleted == ["s1", "s1 (v1.0)"]

    def test_status_listing(self, tmp_path):
        sources = tmp_path / "sources.yaml"
        sources.write_text(
            "sources:\n"
            "  - source_id: s1\n"
            "    path: a.txt\n"
            "    version: '2.0'\n"
            "    enabled: false\n",
            encoding="utf-8",
        )
        state = tmp_path / "state.json"
        state.write_text('{"s1": {"hash": "x", "chunks": 7, "version": "1.0"}}', encoding="utf-8")

        importer = KnowledgeImporter(sources_path=sources, state_path=state)
        rows = importer.status()
        assert rows[0] == {
            "source_id": "s1", "path": "a.txt", "enabled": False,
            "synced": True, "chunks": 7, "version": "1.0",
        }
