"""
Clinical Guideline - 临床指南查询技能
查询临床指南信息
"""
from typing import Dict, Any
from loguru import logger

# 全局单例
_kb_instance = None


def get_knowledge_base():
    """获取知识库实例（懒加载）"""
    global _kb_instance
    if _kb_instance is None:
        try:
            from knowledge.milvus_kb import MedicalKnowledgeBase
            _kb_instance = MedicalKnowledgeBase()
        except Exception as e:
            logger.warning(f"MedicalKnowledgeBase init failed: {e}")
            _kb_instance = None
    return _kb_instance


async def clinical_guideline(query: str, max_results: int = 1) -> Dict[str, Any]:
    """
    查询临床指南

    Args:
        query: 查询关键词
        max_results: 最大结果数

    Returns:
        临床指南信息
    """
    logger.info(f"Querying clinical guideline: {query}")

    kb = get_knowledge_base()

    if kb is None:
        return {
            "answer": "知识库暂不可用，无法查询临床指南。",
            "sources": [],
        }

    try:
        # 搜索临床指南（doc_type 过滤，同步接口）
        results = kb.search(query, top_k=max_results, doc_type="clinical_guideline")

        if not results:
            return {
                "answer": f"未找到关于「{query}」的临床指南。",
                "sources": [],
            }

        # 拼接指南内容（逐条标注出处：文件名 + 页码，如有）
        answer_parts = [f"## 临床指南: {query}\n"]
        sources = []
        for i, result in enumerate(results, 1):
            content = result.get("text", "")
            source = result.get("source", "")
            page = result.get("page")
            if source:
                sources.append(source)
            cite = source or "未知来源"
            if page is not None:
                cite = f"{cite}，第 {page} 页"
            answer_parts.append(f"{i}. {content} 【来源：{cite}】")

        return {
            "answer": "\n".join(answer_parts),
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"Clinical guideline query failed: {e}")
        return {
            "answer": f"查询失败: {str(e)}",
            "sources": [],
        }
