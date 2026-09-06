"""
Analyze Symptoms - 症状分析技能
分析症状模式，识别可能的疾病
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
            from knowledge.milvus_kb import MedicalKnowledgeBase
            _kb_instance = MedicalKnowledgeBase()
        except Exception as e:
            logger.warning(f"MedicalKnowledgeBase init failed: {e}")
            _kb_instance = None
    return _kb_instance


# 症状分类字典
SYMPTOM_CATEGORIES = {
    "respiratory": {
        "name": "呼吸系统",
        "keywords": ["咳嗽", "咳痰", "气喘", "呼吸困难", "胸痛", "咽痛", "流涕", "打喷嚏", "鼻塞"]
    },
    "digestive": {
        "name": "消化系统",
        "keywords": ["腹痛", "腹泻", "恶心", "呕吐", "便秘", "腹胀", "食欲不振", "反酸", "嗳气"]
    },
    "neurological": {
        "name": "神经系统",
        "keywords": ["头痛", "头晕", "失眠", "麻木", "抽搐", "意识障碍", "记忆力下降", "无力"]
    },
    "cardiovascular": {
        "name": "心血管系统",
        "keywords": ["心悸", "胸闷", "胸痛", "水肿", "发绀", "血压升高", "脉搏不齐"]
    },
    "musculoskeletal": {
        "name": "肌肉骨骼系统",
        "keywords": ["关节痛", "肌肉痛", "腰痛", "背痛", "肿胀", "活动受限", "僵硬"]
    }
}

# 症状-疾病关联
SYMPTOM_DISEASE_MAP = {
    "咳嗽": ["感冒", "支气管炎", "肺炎", "哮喘"],
    "发热": ["感冒", "流感", "感染性疾病", "炎症"],
    "头痛": ["偏头痛", "紧张性头痛", "高血压", "感冒"],
    "胸痛": ["心绞痛", "心肌梗死", "肺炎", "胸膜炎"],
    "腹痛": ["胃炎", "阑尾炎", "肠梗阻", "胆囊炎"],
    "心悸": ["心律失常", "甲亢", "贫血", "焦虑症"],
    "头晕": ["高血压", "低血压", "贫血", "颈椎病"],
    "腹泻": ["急性肠炎", "食物中毒", "肠易激综合征", "痢疾"]
}


async def analyze_symptoms(symptoms: str) -> Dict[str, Any]:
    """
    分析症状模式

    Args:
        symptoms: 症状描述

    Returns:
        分析结果
    """
    logger.info(f"Analyzing symptoms: {symptoms}")

    # 检测涉及的症状分类
    detected_categories = []
    for category_key, category_info in SYMPTOM_CATEGORIES.items():
        for keyword in category_info["keywords"]:
            if keyword in symptoms:
                if category_info["name"] not in detected_categories:
                    detected_categories.append(category_info["name"])
                break

    # 关联可能的疾病
    possible_diseases = []
    for symptom, diseases in SYMPTOM_DISEASE_MAP.items():
        if symptom in symptoms:
            for disease in diseases:
                if disease not in possible_diseases:
                    possible_diseases.append(disease)

    # 知识库增强
    kb_insights = ""
    kb_sources = []
    kb = get_knowledge_base()
    if kb:
        try:
            kb_results = kb.search(symptoms, top_k=3)
            if kb_results:
                insights = [r.get('text', '')[:100] for r in kb_results]
                kb_insights = "\n".join(insights)
                kb_sources = list(dict.fromkeys(
                    r.get('source', '') for r in kb_results if r.get('source')
                ))
        except Exception as e:
            logger.warning(f"KB enhancement failed: {e}")

    # 构建答案
    patterns = f"涉及系统: {', '.join(detected_categories) if detected_categories else '未明确分类'}"
    diseases_str = ", ".join(possible_diseases[:5]) if possible_diseases else "暂无明确关联"

    answer_parts = [
        f"**症状模式**: {patterns}",
        f"**可能疾病**: {diseases_str}"
    ]
    if kb_insights:
        answer_parts.append(f"\n**知识库洞察**:\n{kb_insights}")

    answer = "\n".join(answer_parts)

    return {
        "answer": answer,
        "patterns": detected_categories,
        "possible_diseases": possible_diseases,
        "kb_insights": kb_insights,
        "sources": kb_sources
    }
