"""
医疗知识文档导入脚本
从 knowledge/data/documents/ 目录读取 .txt 与 .pdf 文件并导入 Milvus 知识库
PDF 文档按页提取并记录页码，便于回答时标注"文件名 + 页码"出处
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from knowledge.milvus_kb import MedicalKnowledgeBase

# 文档目录
_DOCUMENTS_DIR = _PROJECT_ROOT / "knowledge" / "data" / "documents"


def _infer_doc_type(filename: str) -> str:
    """
    根据文件名前缀推断文档类型

    命名约定：
      01-09  → lifestyle（生活方式建议）
      10-19  → disease_classification（ICD-10 疾病分类编码）
      20-29  → clinical_guideline（临床指南）

    Args:
        filename: 文件名（不含路径）

    Returns:
        文档类型字符串
    """
    try:
        prefix = int(filename.split("_")[0])
    except (ValueError, IndexError):
        return "general"

    if 1 <= prefix <= 9:
        return "lifestyle"
    elif 10 <= prefix <= 19:
        return "disease_classification"
    elif 20 <= prefix <= 29:
        return "clinical_guideline"
    else:
        return "general"


def _load_pdf_pages(pdf_file: Path) -> List[Dict[str, Any]]:
    """
    逐页读取 PDF 文件

    Args:
        pdf_file: PDF 文件路径

    Returns:
        页列表，每项包含 {text, page}（page 从 1 开始）
    """
    from pypdf import PdfReader

    pages = []
    reader = PdfReader(str(pdf_file))
    for page_no, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning(f"跳过空白页: {pdf_file.name} 第{page_no}页")
            continue
        pages.append({"text": text, "page": page_no})
    return pages


def load_documents_from_directory(directory: Path = _DOCUMENTS_DIR) -> List[Dict[str, Any]]:
    """
    从指定目录读取所有 .txt 与 .pdf 文件

    Args:
        directory: 文档目录路径

    Returns:
        文档列表，每项包含 {text, doc_type, source, page}
        （txt 文档 page 为 None；pdf 每页一条，page 为页码）
    """
    if not directory.exists():
        logger.error(f"文档目录不存在: {directory}")
        return []

    documents = []

    # 读取 txt 文件（无页码概念）
    for txt_file in sorted(directory.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                logger.warning(f"跳过空文件: {txt_file.name}")
                continue

            doc_type = _infer_doc_type(txt_file.name)
            documents.append({
                "text": text,
                "doc_type": doc_type,
                "source": txt_file.name,
                "page": None,
            })
            logger.info(f"已读取: {txt_file.name} (type={doc_type}, len={len(text)})")
        except Exception as e:
            logger.error(f"读取文件失败 {txt_file.name}: {e}")

    # 读取 pdf 文件（逐页提取，保留页码）
    for pdf_file in sorted(directory.glob("*.pdf")):
        try:
            doc_type = _infer_doc_type(pdf_file.name)
            pdf_pages = _load_pdf_pages(pdf_file)
            for page_info in pdf_pages:
                documents.append({
                    "text": page_info["text"],
                    "doc_type": doc_type,
                    "source": pdf_file.name,
                    "page": page_info["page"],
                })
            logger.info(f"已读取: {pdf_file.name} (type={doc_type}, {len(pdf_pages)} 页)")
        except Exception as e:
            logger.error(f"读取 PDF 失败 {pdf_file.name}: {e}")

    if not documents:
        logger.warning(f"文档目录为空: {directory}")
        return []

    logger.info(f"共读取 {len(documents)} 个文档/页")
    return documents


def main():
    """
    主函数：加载文档 → 导入 Milvus → 测试检索
    """
    logger.info("=" * 60)
    logger.info("医疗知识库文档导入脚本")
    logger.info("=" * 60)

    # 1. 加载文档
    documents = load_documents_from_directory()
    if not documents:
        logger.error("没有可导入的文档，退出")
        return

    # 2. 初始化知识库
    kb = MedicalKnowledgeBase()

    # 3. 逐文档导入，确保每个文档块的 source 为真实文件名、page 为真实页码（如有）
    total_inserted = 0
    for doc in documents:
        inserted = kb.add_documents(
            texts=[doc["text"]],
            doc_type=doc["doc_type"],
            source=doc["source"],
            pages=[doc.get("page")],
        )
        total_inserted += inserted

    logger.info(f"导入完成，共插入 {total_inserted} 个文档块")

    # 4. 统计
    count = kb.count_documents()
    logger.info(f"知识库当前文档块总数: {count}")

    # 5. 测试检索
    logger.info("-" * 60)
    logger.info("测试检索")
    logger.info("-" * 60)

    test_queries = [
        "高血压患者应该如何饮食？",
        "糖尿病的 diagnostic 标准是什么？",
        "感冒了怎么办？",
        "胸痛应该怎么办？",
        "高血压的 ICD-10 编码是什么？",
    ]

    for query in test_queries:
        results = kb.search(query=query, top_k=3)
        logger.info(f"\n查询: {query}")
        for i, r in enumerate(results, 1):
            snippet = r["text"][:80].replace("\n", " ")
            logger.info(f"  [{i}] score={r['score']:.4f} type={r['doc_type']} | {snippet}...")

    logger.info("=" * 60)
    logger.info("导入脚本执行完毕")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
