"""
医疗安全硬约束：高危症状检测与强制就医引导

配置来源：constraints/agent_constraints.yaml 的 emergency_constraints 段
（关键词、就医引导判定模式均可配置），文件缺失或解析失败时使用内置默认值，
不阻断主流程。
"""
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

_CONSTRAINTS_PATH = Path(__file__).parent / "agent_constraints.yaml"

# 内置默认值：YAML 配置缺失/异常时的兜底，保证硬约束始终生效
_DEFAULT_KEYWORDS = (
    "胸闷", "胸痛", "呼吸困难", "咯血", "大出血", "意识障碍", "昏迷",
    "抽搐", "剧烈头痛", "药物过量", "自杀", "休克", "中毒", "窒息",
    "心脏骤停", "中风",
)
_DEFAULT_URGENT_CARE_PATTERNS = (
    r"立即.*就医", r"拨打.*120", r"急救电话", r"急诊", r"尽快.*医院", r"寻求.*医疗帮助",
)

_DEFAULT_RESPONSE = (
    "⚠️ **紧急就医提醒**\n\n"
    "您描述的症状（{keyword}）可能属于急症，存在延误就医的风险。\n\n"
    "**请立即拨打 120 或前往最近的急诊科就诊**，不要等待线上回复，就医前请勿自行用药。\n\n"
    "本系统提供的信息不能替代执业医师的诊疗。"
)


@lru_cache(maxsize=1)
def _load_emergency_config() -> dict:
    """加载硬约束配置（进程内只读一次；异常时返回空 dict，由默认值兜底）"""
    try:
        with open(_CONSTRAINTS_PATH, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        config = doc.get("emergency_constraints") or {}
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _compiled_urgent_care_patterns():
    patterns = _load_emergency_config().get("urgent_care_patterns") or list(_DEFAULT_URGENT_CARE_PATTERNS)
    return tuple(re.compile(p) for p in patterns)


def get_emergency_keywords() -> tuple:
    """获取高危症状关键词元组"""
    keywords = _load_emergency_config().get("keywords") or list(_DEFAULT_KEYWORDS)
    return tuple(keywords)


def detect_emergency(text: str) -> Optional[str]:
    """
    检测文本中是否包含高危症状关键词。

    Args:
        text: 待检测文本（用户问题或回答）

    Returns:
        命中的第一个关键词；未命中返回 None
    """
    if not text:
        return None
    for keyword in get_emergency_keywords():
        if keyword in text:
            return keyword
    return None


def has_urgent_care_guidance(text: str) -> bool:
    """判断文本是否已包含紧急就医引导"""
    if not text:
        return False
    return any(p.search(text) for p in _compiled_urgent_care_patterns())


def emergency_response(matched_keyword: Optional[str] = None) -> str:
    """
    生成标准紧急就医引导话术（用于硬约束触发时替换原回答）

    Args:
        matched_keyword: 命中的高危关键词（用于提示用户具体风险点）

    Returns:
        标准就医引导文本
    """
    template = _load_emergency_config().get("response_template") or _DEFAULT_RESPONSE
    keyword = matched_keyword or "您描述的症状"
    try:
        return template.format(keyword=keyword)
    except (KeyError, IndexError):
        return template
