"""
知识库多源导入管道（Knowledge Importer）

按 knowledge/data/sources.yaml 文档源清单将外部权威医学资料
（txt / markdown / pdf）增量导入 Milvus 知识库：

- 多格式解析：txt / md 直接读取，pdf 复用 pypdf 逐页提取
- 增量同步：按文件内容 hash 去重，未变化的条目跳过；
  内容变化时先按 source_id 删除旧文档块再重新入库
- 元数据：doc_type / source / authority / version 随文档写入，
  检索结果可通过 source 字段溯源
- 检索验证：入库后用首个分块作为查询做一次语义检索，
  确认新语料能被命中（未命中输出告警，提示人工检查）

CLI 用法：
    python -m knowledge.importer --sync          # 按清单增量同步
    python -m knowledge.importer --list          # 查看清单与状态
"""
import os
import sys
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

SOURCES_PATH = _PROJECT_ROOT / "knowledge" / "data" / "sources.yaml"
STATE_PATH = _PROJECT_ROOT / "knowledge" / "data" / "importer_state.json"

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")

# 同步报告的单条结果
_SYNC_RESULT_FIELDS = ("source_id", "status", "detail")


def load_sources(sources_path: Path = SOURCES_PATH) -> List[Dict]:
    """
    加载文档源清单。

    Returns:
        源条目列表；清单文件不存在时返回空列表
    """
    if not sources_path.exists():
        logger.warning(f"文档源清单不存在: {sources_path}")
        return []
    try:
        with open(sources_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        sources = doc.get("sources") or []
        return [s for s in sources if isinstance(s, dict)]
    except Exception as e:
        logger.error(f"文档源清单解析失败: {e}")
        return []


def file_hash(path: Path) -> str:
    """计算文件内容 MD5（增量去重的依据）"""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parse_document(path: Path) -> List[str]:
    """
    解析文档为文本列表（txt/md 整篇一个元素；pdf 逐页一个元素，页码可追溯）

    Returns:
        文本列表（PDF 场景与页码一一对应，交由 add_documents 的 pages 参数）
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [f.read()]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("解析 PDF 需要安装 pypdf（requirements.txt 已包含）")

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
        if not pages:
            raise RuntimeError(f"PDF 未提取到任何文本: {path}")
        return pages

    raise ValueError(f"不支持的文档格式 {suffix}（支持 {SUPPORTED_EXTENSIONS}）")


class KnowledgeImporter:
    """知识库增量导入器"""

    def __init__(self, sources_path: Path = SOURCES_PATH, state_path: Path = STATE_PATH):
        self._sources_path = sources_path
        self._state_path = state_path
        # 状态文件：{source_id: {"hash": ..., "chunks": n, "version": ...}}
        self._state: Dict[str, Dict] = self._load_state()
        self._kb = None  # 懒初始化（加载嵌入模型较慢，仅在真正入库时初始化）

    # ---------------- 状态管理 ----------------

    def _load_state(self) -> Dict[str, Dict]:
        import json
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception as e:
                logger.warning(f"导入状态加载失败（将视为全部未同步）: {e}")
        return {}

    def _save_state(self):
        import json
        tmp = Path(f"{self._state_path}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.error(f"导入状态保存失败: {e}")

    def _get_kb(self):
        """懒初始化知识库单例（首次调用时加载嵌入模型）"""
        if self._kb is None:
            from knowledge.milvus_kb import MedicalKnowledgeBase
            self._kb = MedicalKnowledgeBase()
        return self._kb

    # ---------------- 核心流程 ----------------

    def sync(self) -> List[Dict]:
        """
        按清单增量同步全部 enabled 的文档源。

        Returns:
            同步报告列表：[{source_id, status, detail}]，
            status 为 synced / skipped / failed
        """
        sources = load_sources(self._sources_path)
        if not sources:
            return [{"source_id": "-", "status": "failed",
                     "detail": f"清单为空或不存在: {self._sources_path}"}]

        report = []
        for spec in sources:
            source_id = str(spec.get("source_id") or "").strip()
            if not source_id:
                report.append({"source_id": "-", "status": "failed", "detail": "缺少 source_id"})
                continue
            if not spec.get("enabled", False):
                report.append({"source_id": source_id, "status": "skipped", "detail": "enabled=false"})
                continue
            report.append(self._sync_one(source_id, spec))

        self._save_state()
        return report

    def _sync_one(self, source_id: str, spec: Dict) -> Dict:
        """同步单个文档源：解析 -> 增量判断 -> 删旧 -> 入库 -> 验证"""
        rel_path = str(spec.get("path") or "").strip()
        # 相对路径统一相对项目根目录解析
        if Path(rel_path).is_absolute():
            path = Path(rel_path)
        else:
            path = _PROJECT_ROOT / rel_path

        if not path.exists():
            return {"source_id": source_id, "status": "failed", "detail": f"文件不存在: {path}"}

        try:
            digest = file_hash(path)
        except Exception as e:
            return {"source_id": source_id, "status": "failed", "detail": f"读取失败: {e}"}

        known = self._state.get(source_id) or {}
        if known.get("hash") == digest:
            return {"source_id": source_id, "status": "skipped",
                    "detail": f"内容未变化 (chunks={known.get('chunks', '?')})"}

        # 解析文档
        try:
            texts = parse_document(path)
        except Exception as e:
            return {"source_id": source_id, "status": "failed", "detail": f"解析失败: {e}"}

        try:
            kb = self._get_kb()

            doc_type = str(spec.get("doc_type") or "general")
            pages = list(range(1, len(texts) + 1)) if path.suffix.lower() == ".pdf" else None
            # source 写入"source_id (v版本)"形式，检索结果可直接展示出处与版本
            source_label = f"{source_id} (v{spec.get('version', '1.0')})"

            # 内容变化：先删除该来源的历史文档块。需同时覆盖：
            # 原始 source_id（首次同步前的历史导入）、旧版本标签与新标签，
            # 避免版本号变化后旧块残留导致重复检索
            previous_label = known.get("source_label")
            for label in dict.fromkeys(
                lbl for lbl in (previous_label, source_id, source_label) if lbl
            ):
                kb.delete_by_source(label)

            chunks = kb.add_documents(
                texts=texts,
                doc_type=doc_type,
                source=source_label,
                pages=pages,
            )

            self._state[source_id] = {
                "hash": digest,
                "chunks": chunks,
                "version": str(spec.get("version", "1.0")),
                "source_label": source_label,
            }

            # 检索验证：用首个分块前缀做语义检索，确认新语料可被命中
            verify_ok = self._verify_ingest(source_label, texts[0])

            detail = f"入库 {chunks} 块，验证{'通过' if verify_ok else '未命中（请人工检查）'}"
            return {"source_id": source_id, "status": "synced", "detail": detail}
        except Exception as e:
            logger.exception(f"入库失败: {source_id}")
            return {"source_id": source_id, "status": "failed", "detail": f"入库失败: {e}"}

    def _verify_ingest(self, source_label: str, sample_text: str, top_k: int = 5) -> bool:
        """
        入库质量验证：以首个分块前缀作为查询做语义检索，
        检索结果中应包含来自该来源的文档块。

        Returns:
            验证是否通过；失败时输出告警（不阻断同步流程）
        """
        try:
            kb = self._get_kb()
            query = sample_text.strip()[:64]
            hits = kb.search_by_text(query, top_k=top_k)
            hit_sources = {h.get("source", "") for h in hits}
            if source_label in hit_sources:
                return True
            logger.warning(
                f"入库验证未命中新语料 (source={source_label}, 命中来源: {hit_sources})，"
                "请人工检查文档内容与分块质量"
            )
            return False
        except Exception as e:
            logger.warning(f"入库验证执行失败: {e}")
            return False

    # ---------------- 状态查询 ----------------

    def status(self) -> List[Dict]:
        """查看清单条目与同步状态"""
        rows = []
        for spec in load_sources(self._sources_path):
            source_id = str(spec.get("source_id") or "-")
            state = self._state.get(source_id)
            rows.append({
                "source_id": source_id,
                "path": spec.get("path", ""),
                "enabled": bool(spec.get("enabled", False)),
                "synced": state is not None,
                "chunks": (state or {}).get("chunks"),
                "version": (state or {}).get("version") or spec.get("version", "-"),
            })
        return rows


def _print_report(rows: List[Dict], title: str):
    print(f"\n{title}")
    print("-" * 72)
    for row in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in row.items()))


def main():
    parser = argparse.ArgumentParser(description="知识库多源导入管道")
    parser.add_argument("--sync", action="store_true", help="按 sources.yaml 清单增量同步")
    parser.add_argument("--list", action="store_true", help="查看清单条目与同步状态")
    args = parser.parse_args()

    if not (args.sync or args.list):
        parser.print_help()
        return

    importer = KnowledgeImporter()
    if args.list:
        _print_report(importer.status(), "文档源清单状态")
    if args.sync:
        report = importer.sync()
        failed = [r for r in report if r["status"] == "failed"]
        _print_report(report, "同步报告")
        print()
        if failed:
            print(f"⚠️ {len(failed)} 个来源同步失败，详见上方报告")
            sys.exit(1)
        print("✅ 同步完成")


if __name__ == "__main__":
    main()
