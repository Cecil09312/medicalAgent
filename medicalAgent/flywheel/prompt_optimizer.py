"""
Prompt/约束优化建议
分析专家修正模式，生成系统 prompt 和约束规则的优化建议
"""
import json
from typing import List, Dict, Any, Optional
from collections import Counter
from loguru import logger


class PromptOptimizer:
    """Prompt/约束优化建议生成器"""

    def __init__(self):
        self._suggestions_history: List[Dict] = []

    def analyze_corrections(self, corrections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        分析专家修正模式，生成优化建议

        Args:
            corrections: 修正列表，每项包含:
                - question: 问题
                - original_answer: 原始回答
                - expert_correction: 专家修正
                - trigger_reason: 触发原因

        Returns:
            优化建议列表
        """
        if not corrections:
            return []

        suggestions = []

        # 分析触发原因分布
        reason_counts = Counter(c.get("trigger_reason", "") for c in corrections)
        logger.info(f"修正触发原因分布: {dict(reason_counts)}")

        # 分析1: 高频触发原因 -> 建议加强对应约束
        for reason, count in reason_counts.most_common(3):
            if count >= 2:
                suggestions.append({
                    "type": "constraint",
                    "priority": "high" if count >= 5 else "medium",
                    "reason": f"触发原因 '{reason}' 出现 {count} 次",
                    "suggestion": self._get_constraint_suggestion(reason),
                })

        # 分析2: 修正内容模式 -> 识别系统性问题
        missing_disclaimer_count = sum(
            1 for c in corrections
            if "仅供参考" not in c.get("expert_correction", "")
            and "仅供参考" not in c.get("original_answer", "")
        )
        if missing_disclaimer_count > len(corrections) * 0.3:
            suggestions.append({
                "type": "prompt",
                "priority": "high",
                "reason": f"{missing_disclaimer_count}/{len(corrections)} 的回答缺少免责声明",
                "suggestion": "在相关 Agent 的 system prompt 中强调：每次回答必须包含免责声明",
            })

        # 分析3: 修正幅度分析（原始回答和修正的差异程度）
        large_corrections = []
        for c in corrections:
            orig = c.get("original_answer", "")
            corr = c.get("expert_correction", "")
            if orig and corr:
                # 简单的差异度量：长度变化比例
                len_ratio = len(corr) / max(len(orig), 1)
                if len_ratio > 2.0 or len_ratio < 0.3:
                    large_corrections.append(c)

        if large_corrections:
            suggestions.append({
                "type": "prompt",
                "priority": "medium",
                "reason": f"{len(large_corrections)} 个回答被大幅修正（长度变化超过 70%）",
                "suggestion": "检查相关 Agent 的回答详细程度是否合适，考虑调整 system prompt 中的回答长度指导",
            })

        # 分析4: 按 Agent 分组统计
        agent_errors = Counter()
        for c in corrections:
            # 从问题内容推断 Agent
            q = c.get("question", "")
            if any(kw in q for kw in ["饮食", "运动", "生活"]):
                agent_errors["consultation_agent"] += 1
            elif any(kw in q for kw in ["症状", "诊断", "风险"]):
                agent_errors["diagnostic_agent"] += 1
            elif any(kw in q for kw in ["指南", "研究", "进展"]):
                agent_errors["research_agent"] += 1

        for agent, count in agent_errors.most_common():
            if count >= 2:
                suggestions.append({
                    "type": "prompt",
                    "priority": "medium",
                    "reason": f"{agent} 有 {count} 次修正",
                    "suggestion": f"审查 {agent} 的 system prompt，关注回答准确性和完整性",
                })

        self._suggestions_history.extend(suggestions)
        logger.info(f"生成了 {len(suggestions)} 条优化建议")
        return suggestions

    def _get_constraint_suggestion(self, reason: str) -> str:
        """根据触发原因生成约束建议"""
        mapping = {
            "high_risk": "增加高风险症状的自动检测规则，确保所有涉及急症的问答都包含就医提醒",
            "high_risk_keyword": "扩展高风险关键词列表，或提高现有关键词的检测灵敏度",
            "constraint_violation": "审查当前约束规则，确认是否有规则定义不合理或缺失",
            "sensitive_operation": "加强对诊断和处方类内容的检测，考虑增加 LLM 辅助判断",
            "low_confidence": "优化 Agent 的 Skill 调用策略，减少因无法调用合适 Skill 导致的低质量回答",
            "random_sample": "随机抽样发现修正，说明系统存在未被实时检测捕获的质量问题",
            "user_flag": "用户标记的修正需要重点关注，可能反映了系统性的回答质量问题",
        }
        return mapping.get(reason, f"审查 '{reason}' 相关的约束规则和 Agent 行为")

    def generate_prompt_patch(self, suggestions: List[Dict]) -> str:
        """
        生成 prompt 补丁（供人工确认后应用）

        Args:
            suggestions: 优化建议列表

        Returns:
            prompt 补丁文本
        """
        if not suggestions:
            return "无需修改"

        lines = ["# 优化建议补丁（请人工确认后应用）\n"]

        prompt_suggestions = [s for s in suggestions if s["type"] == "prompt"]
        constraint_suggestions = [s for s in suggestions if s["type"] == "constraint"]

        if prompt_suggestions:
            lines.append("## Prompt 修改建议\n")
            for i, s in enumerate(prompt_suggestions, 1):
                lines.append(f"### {i}. [{s['priority'].upper()}] {s['reason']}")
                lines.append(f"   建议: {s['suggestion']}\n")

        if constraint_suggestions:
            lines.append("## 约束规则修改建议\n")
            for i, s in enumerate(constraint_suggestions, 1):
                lines.append(f"### {i}. [{s['priority'].upper()}] {s['reason']}")
                lines.append(f"   建议: {s['suggestion']}\n")

        return "\n".join(lines)
