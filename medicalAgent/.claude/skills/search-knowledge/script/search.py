"""
Search Knowledge - 医学知识库搜索技能
使用 Milvus 向量数据库搜索医学知识
"""
import asyncio
import threading
from typing import Dict, Any, Optional
from loguru import logger

# 全局单例，避免重复加载模型
_kb_instance = None
# 初始化锁：防止并发请求与预热线程重复加载嵌入模型
_kb_init_lock = threading.Lock()


def get_knowledge_base():
    """
    获取知识库实例（懒加载）

    Returns:
        MedicalKnowledgeBase 实例
    """
    global _kb_instance
    if _kb_instance is None:
        with _kb_init_lock:
            # 双重检查：等待锁期间可能已被其他线程初始化
            if _kb_instance is None:
                try:
                    from knowledge.milvus_kb import MedicalKnowledgeBase
                    _kb_instance = MedicalKnowledgeBase()
                    logger.info("MedicalKnowledgeBase initialized")
                except Exception as e:
                    logger.warning(f"MedicalKnowledgeBase init failed: {e}")
                    _kb_instance = None
    return _kb_instance


async def search_knowledge(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    搜索医学知识库

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果字典（含来源列表）
    """
    logger.info(f"Searching knowledge base: query='{query}', max_results={max_results}")

    # 首次初始化需加载嵌入模型（耗时较长），放入线程避免阻塞事件循环
    kb = await asyncio.to_thread(get_knowledge_base)

    if kb is None:
        # 降级处理：返回空结果
        return {
            "answer": "知识库暂不可用，请稍后重试。",
            "total_found": 0,
            "query": query,
            "sources": [],
        }

    try:
        # 搜索 Milvus（同步接口）
        results = kb.search(query, top_k=max_results)

        # 格式化结果
        formatted = format_results(results)

        return {
            "answer": formatted["answer"],
            "total_found": len(results),
            "query": query,
            "sources": formatted["sources"],
        }

    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        return {
            "answer": f"搜索失败: {str(e)}",
            "total_found": 0,
            "query": query,
            "sources": [],
        }


def _format_citation(result: Dict[str, Any]) -> str:
    """
    格式化单条结果的引用出处

    Args:
        result: 检索结果（含 source / page）

    Returns:
        引用文本，如 "01_lifestyle_hypertension.txt" 或 "指南.pdf，第 3 页"
    """
    source = result.get("source", "") or "未知来源"
    page = result.get("page")
    if page is not None:
        return f"{source}，第 {page} 页"
    return source


def format_results(results: list) -> Dict[str, Any]:
    """
    格式化搜索结果

    Args:
        results: 原始结果列表（含 text / score / doc_type / source / page）

    Returns:
        格式化后的字典
    """
    if not results:
        return {"answer": "未找到相关知识。", "sources": []}

    answer_parts = []
    for i, result in enumerate(results, 1):
        content = result.get("text", "")
        doc_type = result.get("doc_type", "")
        # 逐条标注出处：文件名 + 页码（如有）
        cite = _format_citation(result)
        answer_parts.append(f"{i}. [{doc_type}] {content[:200]}... 【来源：{cite}】")

    # 去重收集来源文档名
    sources = list(dict.fromkeys(
        r.get("source", "") for r in results if r.get("source")
    ))

    return {"answer": "\n\n".join(answer_parts), "sources": sources}
