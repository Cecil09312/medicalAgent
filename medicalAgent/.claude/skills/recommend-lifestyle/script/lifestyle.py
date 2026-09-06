"""
Recommend Lifestyle - 生活方式推荐技能
根据诊断结果推荐生活方式调整
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
            from medical_knowledge_base import MedicalKnowledgeBase
            _kb_instance = MedicalKnowledgeBase()
        except ImportError:
            _kb_instance = None
    return _kb_instance


async def recommend_lifestyle(diagnosis: str) -> Dict[str, Any]:
    """
    推荐生活方式调整

    Args:
        diagnosis: 诊断结果

    Returns:
        生活方式建议
    """
    logger.info(f"Recommending lifestyle for diagnosis: {diagnosis}")

    kb = get_knowledge_base()

    if kb is None:
        return {
            "answer": "知识库暂不可用，无法获取生活方式建议。",
            "diagnosis": diagnosis,
            "categories": [],
            "source": "unavailable"
        }

    try:
        # 搜索生活方式建议
        results = await kb.search(
            diagnosis,
            max_results=3,
            filter_type="lifestyle"
        )

        if not results:
            return {
                "answer": f"未找到关于「{diagnosis}」的生活方式建议。",
                "diagnosis": diagnosis,
                "categories": [],
                "source": "knowledge_base"
            }

        # 提取分类和内容
        categories = []
        answer_parts = [f"## 「{diagnosis}」生活方式建议\n"]

        for i, result in enumerate(results, 1):
            content = result.get('content', '')
            category = result.get('category', '一般建议')
            categories.append(category)
            answer_parts.append(f"{i}. **{category}**: {content}")

        return {
            "answer": "\n".join(answer_parts),
            "diagnosis": diagnosis,
            "categories": categories,
            "source": "knowledge_base"
        }

    except Exception as e:
        logger.error(f"Lifestyle recommendation failed: {e}")
        return {
            "answer": f"推荐失败: {str(e)}",
            "diagnosis": diagnosis,
            "categories": [],
            "source": "error"
        }
