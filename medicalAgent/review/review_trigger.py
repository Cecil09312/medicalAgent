"""
审核触发判断
根据风险等级、约束违规、置信度等条件判断是否需要专家审核
"""
import re
import os
import random
from typing import Dict, Any, List, Optional
from loguru import logger


class ReviewTrigger:
    """审核触发判断器"""

    def __init__(self):
        """初始化，从环境变量读取配置"""
        self.realtime_enabled = os.getenv("REVIEW_REALTIME_ENABLED", "true").lower() == "true"
        self.async_sample_rate = float(os.getenv("REVIEW_ASYNC_SAMPLE_RATE", "0.05"))

        # 高风险关键词（触发实时审核）
        self.high_risk_keywords = [
            "心肌梗死", "脑卒中", "休克", "窒息", "心脏骤停",
            "大出血", "昏迷", "呼吸困难", "自杀", "中毒",
        ]

        # 敏感操作模式（诊断/处方类）
        self.sensitive_patterns = [
            r"确诊.*为",
            r"诊断.*为",
            r"你(患|得)了.{2,10}(病|症)",
            r"建议.*服用.{2,20}(mg|克|片|粒)",
            r"处方.*药",
            r"用药.*方案.*为",
        ]

    def should_review_realtime(
        self,
        agent_output: Dict[str, Any],
        risk_level: str = "",
        violations: Optional[List[str]] = None,
        forced_stop: bool = False,
    ) -> tuple:
        """
        判断是否需要实时审核

        Args:
            agent_output: Agent 输出字典，需包含 "answer" 键
            risk_level: 风险评估等级 ("HIGH" / "MEDIUM" / "LOW" / "")
            violations: 约束违规列表
            forced_stop: 是否因达到最大迭代次数被强制终止

        Returns:
            (should_review: bool, reason: str)
        """
        if not self.realtime_enabled:
            return False, ""

        answer = agent_output.get("answer", "")
        violations = violations or []

        # 条件1: 高风险症状
        if risk_level.upper() == "HIGH":
            logger.info("实时审核触发: 高风险症状")
            return True, "high_risk"

        # 条件2: 约束不可自动修复
        non_fixable = [v for v in violations if v not in ("missing_disclaimer", "excessive_length")]
        if non_fixable:
            logger.info(f"实时审核触发: 不可自动修复的违规 {non_fixable}")
            return True, "constraint_violation"

        # 条件3: 敏感操作（诊断/处方）
        for pattern in self.sensitive_patterns:
            if re.search(pattern, answer):
                logger.info(f"实时审核触发: 敏感操作匹配 '{pattern}'")
                return True, "sensitive_operation"

        # 条件4: 低置信度（被强制终止）
        if forced_stop:
            logger.info("实时审核触发: Agent 被强制终止（低置信度）")
            return True, "low_confidence"

        # 条件5: 高风险关键词检测
        for kw in self.high_risk_keywords:
            if kw in answer:
                logger.info(f"实时审核触发: 回答包含高风险关键词 '{kw}'")
                return True, "high_risk_keyword"

        return False, ""

    def should_enqueue_async(self, agent_output: Dict[str, Any]) -> tuple:
        """
        判断是否入异步审核队列（随机抽样）

        Args:
            agent_output: Agent 输出字典

        Returns:
            (should_enqueue: bool, reason: str)
        """
        # 随机抽样
        if random.random() < self.async_sample_rate:
            logger.debug(f"异步审核触发: 随机抽样 (rate={self.async_sample_rate})")
            return True, "random_sample"

        return False, ""

    def should_enqueue_user_flag(self) -> tuple:
        """
        用户主动标记时的入队判断（总是入队）

        Returns:
            (should_enqueue: bool, reason: str)
        """
        return True, "user_flag"
