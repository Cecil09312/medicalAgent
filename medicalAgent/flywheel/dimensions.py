"""
飞轮维度工具：基于关键词的疾病类别粗分类

用于飞轮指标维度下钻（按疾病类别统计介入率/修正率），
关键词命中即归类，未命中返回 "unknown"（旧数据兼容口径一致）。
"""
from typing import Dict, List, Tuple

# 疾病类别关键词表（粗分类，可按需扩充）：类别 -> 关键词列表
DISEASE_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "心血管": ["胸痛", "胸闷", "心悸", "心梗", "心肌梗死", "高血压", "冠心病", "心律", "中风", "脑卒中"],
    "呼吸": ["咳嗽", "感冒", "发烧", "发热", "哮喘", "呼吸困难", "肺炎", "咽喉", "流鼻涕"],
    "消化": ["胃", "腹泻", "便秘", "腹痛", "恶心", "呕吐", "肠", "肝", "食欲"],
    "神经": ["头痛", "头晕", "失眠", "抽搐", "癫痫", "麻木", "昏迷", "意识"],
    "内分泌": ["糖尿病", "血糖", "甲状腺", "肥胖", "血脂"],
    "精神心理": ["焦虑", "抑郁", "自杀", "压力", "情绪"],
    "骨骼肌肉": ["关节", "腰痛", "颈椎", "骨折", "肌肉", "扭伤"],
    "皮肤": ["皮疹", "瘙痒", "湿疹", "痤疮", "荨麻疹"],
    "泌尿生殖": ["尿", "肾", "前列腺", "月经", "妇科"],
}

# 维度取值中"未知"的统一口径：旧记录无维度字段时同样按 unknown 聚合
UNKNOWN = "unknown"


def classify_disease_category(question: str) -> str:
    """
    基于关键词的疾病类别粗分类。

    Args:
        question: 用户问题文本

    Returns:
        疾病类别名称；未命中任何关键词返回 "unknown"
    """
    if not question:
        return UNKNOWN
    for category, keywords in DISEASE_CATEGORY_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            return category
    return UNKNOWN


def build_review_dimensions(
    question: str,
    result: Dict,
) -> Dict[str, str]:
    """
    从问题与回答结果中提取审核维度信息。

    Args:
        question: 用户原始问题
        result: Swarm/Agent 处理结果（含 agent_id、skills_used 等字段）

    Returns:
        维度字典：disease_category / skill_name / agent_type
    """
    skills = result.get("skills_used") or []
    skill_name = skills[0] if len(skills) == 1 else (",".join(skills) if skills else UNKNOWN)
    return {
        "disease_category": classify_disease_category(question),
        "skill_name": skill_name or UNKNOWN,
        "agent_type": result.get("agent_id", UNKNOWN) or UNKNOWN,
    }


def top_dimensions(
    dimension_stats: Dict[str, Dict[str, Dict[str, int]]],
    dimension: str,
    top_n: int = 10,
) -> List[Tuple[str, int, int, float]]:
    """
    按修正数降序聚合某维度的统计。

    Args:
        dimension_stats: {dimension: {value: {"reviews": n, "corrections": n}}} 结构的维度统计
        dimension: 维度名（disease_category / skill_name / agent_type）
        top_n: 返回条数上限

    Returns:
        [(维度取值, 审核数, 修正数, 修正率)] 列表，按修正数降序
    """
    entries = dimension_stats.get(dimension) or {}
    rows = []
    for value, stats in entries.items():
        reviews = stats.get("reviews", 0)
        corrections = stats.get("corrections", 0)
        rate = corrections / reviews if reviews > 0 else 0.0
        rows.append((value, reviews, corrections, rate))
    rows.sort(key=lambda r: (r[2], r[1]), reverse=True)
    return rows[:top_n]
