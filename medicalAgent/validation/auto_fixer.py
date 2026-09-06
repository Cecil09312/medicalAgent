"""
自动修复器
对违反约束的输出进行自动修复
"""
import re
from typing import List
from loguru import logger


class AutoFixer:
    """自动修复器"""

    def __init__(self, max_length: int = 2000):
        """
        初始化修复器
        
        Args:
            max_length: 最大输出长度
        """
        self.max_length = max_length

    def fix_output(self, output: str, auto_fixable: List[str]) -> str:
        """
        根据可修复项列表进行修复
        
        Args:
            output: 原始输出
            auto_fixable: 可修复项列表
        
        Returns:
            修复后的输出
        """
        fixed_output = output
        
        for fix_type in auto_fixable:
            if fix_type == "missing_disclaimer":
                fixed_output = self.fix_missing_disclaimer(fixed_output)
            elif fix_type == "high_risk_warning":
                fixed_output = self.fix_high_risk_warning(fixed_output)
            elif fix_type == "excessive_length":
                fixed_output = self.fix_excessive_length(fixed_output)
            elif fix_type == "remove_diagnose_disease":
                fixed_output = self.remove_diagnosis_statements(fixed_output)
            elif fix_type == "remove_prescribe_medication":
                fixed_output = self.remove_medication_recommendations(fixed_output)
        
        if fixed_output != output:
            logger.info(f"Auto-fixed {len(auto_fixable)} issues")
        
        return fixed_output

    def fix_missing_disclaimer(self, output: str) -> str:
        """
        添加缺失的免责声明
        
        Args:
            output: 输出文本
        
        Returns:
            添加免责声明后的文本
        """
        disclaimer = "\n\n---\n**免责声明：** 以上内容仅供参考，不构成专业医疗建议。如有健康问题，请咨询专业医生。"
        
        # 检查是否已有免责声明
        disclaimer_patterns = [
            r"仅供参考",
            r"不构成医疗建议",
            r"请咨询.*医生",
            r"建议就医"
        ]
        has_disclaimer = any(re.search(p, output) for p in disclaimer_patterns)
        
        if not has_disclaimer:
            output += disclaimer
            logger.debug("Added missing disclaimer")
        
        return output

    def fix_high_risk_warning(self, output: str) -> str:
        """
        添加高风险警告
        
        Args:
            output: 输出文本
        
        Returns:
            添加警告后的文本
        """
        warning_prefix = "⚠️ **紧急提醒：** 您描述的症状可能需要紧急医疗关注，建议立即就医或拨打急救电话。\n\n"
        
        # 检查是否已有紧急提醒
        warning_patterns = [
            r"紧急提醒",
            r"立即就医",
            r"急诊",
            r"急救"
        ]
        has_warning = any(re.search(p, output) for p in warning_patterns)
        
        if not has_warning:
            output = warning_prefix + output
            logger.debug("Added high-risk warning")
        
        return output

    def fix_excessive_length(self, output: str) -> str:
        """
        截断过长的输出
        
        Args:
            output: 输出文本
        
        Returns:
            截断后的文本
        """
        if len(output) <= self.max_length:
            return output
        
        truncated = output[:self.max_length]
        notice = "\n\n---\n*（内容已截断，如需更多信息请咨询专业人士）*"
        
        output = truncated + notice
        logger.debug(f"Truncated output from {len(output)} to {self.max_length} characters")
        
        return output

    def remove_diagnosis_statements(self, output: str) -> str:
        """
        移除诊断性陈述
        
        Args:
            output: 输出文本
        
        Returns:
            移除诊断后的文本
        """
        # 替换诊断性语句
        diagnosis_patterns = [
            (r"你(患|得)了.{2,10}(病|症)", "您可能存在健康风险"),
            (r"诊断.*为", "初步分析提示"),
            (r"确诊.*为", "症状提示可能")
        ]
        
        for pattern, replacement in diagnosis_patterns:
            output = re.sub(pattern, replacement, output)
        
        logger.debug("Removed diagnosis statements")
        return output

    def remove_medication_recommendations(self, output: str) -> str:
        """
        移除药物推荐
        
        Args:
            output: 输出文本
        
        Returns:
            移除药物推荐后的文本
        """
        # 替换药物推荐语句
        medication_patterns = [
            (r"建议.*服用.{2,20}", "建议咨询医生关于用药方案"),
            (r"推荐.*药物.{2,20}", "药物治疗需由医生决定"),
            (r"处方.*药", "处方药需由医生开具")
        ]
        
        for pattern, replacement in medication_patterns:
            output = re.sub(pattern, replacement, output)
        
        logger.debug("Removed medication recommendations")
        return output
