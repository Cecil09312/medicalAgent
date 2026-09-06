"""
Assess Risk - 症状风险评估技能
使用规则引擎评估症状风险等级
"""
from typing import Dict, Any, List
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


# 高风险症状列表
HIGH_RISK_SYMPTOMS = [
    "胸痛", "胸闷", "呼吸困难", "意识丧失", "抽搐",
    "大出血", "剧烈头痛", "偏瘫", "失语", "急性腹痛",
    "高热不退", "呕血", "便血", "窒息", "中毒"
]

# 中风险关键词
MEDIUM_RISK_KEYWORDS = [
    "持续", "加重", "反复", "发热", "肿胀",
    "麻木", "无力", "恶心", "呕吐", "腹泻"
]


async def assess_risk(symptoms: str) -> Dict[str, Any]:
    """
    评估症状风险等级

    Args:
        symptoms: 症状描述

    Returns:
        风险评估结果
    """
    logger.info(f"Assessing risk for symptoms: {symptoms}")

    risk_level = "low"
    recommendation = ""

    # 检查高风险症状
    for high_risk in HIGH_RISK_SYMPTOMS:
        if high_risk in symptoms:
            risk_level = "high"
            recommendation = f"检测到高风险症状「{high_risk}」，建议立即就医。"
            break

    # 检查中风险关键词
    if risk_level != "high":
        for medium_risk in MEDIUM_RISK_KEYWORDS:
            if medium_risk in symptoms:
                risk_level = "medium"
                recommendation = f"症状包含「{medium_risk}」，建议尽快就医检查。"
                break

    # 默认低风险
    if risk_level == "low":
        recommendation = "症状较轻，建议观察并保持良好的生活习惯。"

    # 知识库增强
    kb_advice = ""
    kb = get_knowledge_base()
    if kb:
        try:
            kb_results = await kb.search(symptoms, max_results=2, filter_type="risk_assessment")
            if kb_results:
                kb_advice = kb_results[0].get('content', '')
        except Exception as e:
            logger.warning(f"KB enhancement failed: {e}")

    answer = format_assessment(risk_level, recommendation, kb_advice)

    return {
        "answer": answer,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "kb_advice": kb_advice
    }


def format_assessment(risk_level: str, recommendation: str, kb_advice: str) -> str:
    """
    格式化风险评估结果

    Args:
        risk_level: 风险等级
        recommendation: 建议
        kb_advice: 知识库建议

    Returns:
        格式化后的文本
    """
    level_map = {"high": "高", "medium": "中", "low": "低"}
    level_cn = level_map.get(risk_level, "未知")

    parts = [f"**风险等级**: {level_cn}", f"**建议**: {recommendation}"]

    if kb_advice:
        parts.append(f"\n**参考**: {kb_advice}")

    return "\n".join(parts)
