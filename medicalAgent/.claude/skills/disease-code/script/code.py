"""
Disease Code - 疾病编码查询技能
查询疾病的 ICD-10 编码
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


async def disease_code(disease_name: str) -> Dict[str, Any]:
    """
    查询疾病 ICD-10 编码

    Args:
        disease_name: 疾病名称

    Returns:
        疾病编码信息
    """
    logger.info(f"Querying disease code for: {disease_name}")

    kb = get_knowledge_base()

    if kb is None:
        return {
            "answer": "知识库暂不可用，无法查询疾病编码。",
            "icd10_code": "N/A",
            "category": "N/A",
            "source": "unavailable"
        }

    try:
        # 搜索疾病分类
        results = await kb.search(
            disease_name,
            max_results=1,
            filter_type="disease_classification"
        )

        if not results:
            return {
                "answer": f"未找到「{disease_name}」的 ICD-10 编码。",
                "icd10_code": "N/A",
                "category": "N/A",
                "source": "knowledge_base"
            }

        result = results[0]
        icd10_code = result.get('icd10_code', 'N/A')
        category = result.get('category', 'N/A')
        content = result.get('content', '')

        answer = f"## 「{disease_name}」编码信息\n\n"
        answer += f"- **ICD-10 编码**: {icd10_code}\n"
        answer += f"- **分类**: {category}\n"
        if content:
            answer += f"- **说明**: {content}\n"

        return {
            "answer": answer,
            "icd10_code": icd10_code,
            "category": category,
            "source": "knowledge_base"
        }

    except Exception as e:
        logger.error(f"Disease code query failed: {e}")
        return {
            "answer": f"查询失败: {str(e)}",
            "icd10_code": "N/A",
            "category": "N/A",
            "source": "error"
        }
