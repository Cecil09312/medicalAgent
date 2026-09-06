"""
证据合成器
将网络搜索结果和知识库结果进行证据合成
使用 LLM 进行结构化分析和总结
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class ResearchReport:
    """研究报告数据结构"""
    query: str
    key_findings: List[str] = field(default_factory=list)
    evidence_level: str = ""
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    conflicts: List[str] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class EvidenceSynthesizer:
    """
    证据合成器
    使用 LLM 对多源信息进行结构化合成
    """

    def __init__(self, llm_client=None):
        """
        初始化证据合成器

        Args:
            llm_client: LLM 客户端实例（可选，不提供则使用默认配置）
        """
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # 延迟导入避免循环依赖
            from core.llm_client import LLMClient
            self.llm_client = LLMClient()

        logger.debug("EvidenceSynthesizer initialized")

    async def synthesize(
        self,
        query: str,
        web_results: Optional[List[Dict[str, Any]]] = None,
        kb_results: Optional[List[Dict[str, Any]]] = None,
    ) -> ResearchReport:
        """
        合成证据

        Args:
            query: 研究问题
            web_results: 网络搜索结果
            kb_results: 知识库搜索结果

        Returns:
            ResearchReport 研究报告
        """
        web_results = web_results or []
        kb_results = kb_results or []

        logger.info(f"Synthesizing evidence for: {query}")
        logger.debug(f"Web results: {len(web_results)}, KB results: {len(kb_results)}")

        # 构建合成提示
        prompt = self._build_synthesis_prompt(query, web_results, kb_results)

        try:
            # 调用 LLM
            messages = [
                {"role": "system", "content": "你是一个专业的医疗信息分析助手。请严格按照指定格式输出分析结果。"},
                {"role": "user", "content": prompt},
            ]

            response = await self.llm_client.chat(messages, temperature=0.3)

            # 解析响应
            report = self._parse_response(query, response, web_results, kb_results)

            logger.info(f"Evidence synthesis completed, confidence: {report.confidence}")
            return report

        except Exception as e:
            logger.error(f"Evidence synthesis failed: {e}")

            # 返回基础报告
            return ResearchReport(
                query=query,
                key_findings=["信息合成失败，请人工核实"],
                evidence_level="低",
                sources=[r.get("url", "") for r in web_results if r.get("url")],
                confidence=0.0,
                conflicts=[],
                summary=str(e),
                recommendations=["建议咨询专业医疗人员"],
            )

    def _build_synthesis_prompt(
        self,
        query: str,
        web_results: List[Dict[str, Any]],
        kb_results: List[Dict[str, Any]],
    ) -> str:
        """
        构建证据合成提示

        Args:
            query: 研究问题
            web_results: 网络搜索结果
            kb_results: 知识库搜索结果

        Returns:
            格式化的提示文本
        """
        prompt_parts = [
            f"## 研究问题\n{query}\n",
        ]

        # 格式化网络搜索结果
        if web_results:
            prompt_parts.append("## 网络搜索结果\n")
            for i, r in enumerate(web_results, 1):
                title = r.get("title", "无标题")
                snippet = r.get("snippet", r.get("content", "无内容"))
                url = r.get("url", "")
                prompt_parts.append(f"### 来源 {i}: {title}\n- 摘要: {snippet}\n- URL: {url}\n")

        # 格式化知识库结果
        if kb_results:
            prompt_parts.append("## 知识库搜索结果\n")
            for i, r in enumerate(kb_results, 1):
                content = r.get("content", r.get("text", "无内容"))
                score = r.get("score", 0)
                prompt_parts.append(f"### KB来源 {i} (相关度: {score:.2f})\n{content}\n")

        # 输出格式要求
        prompt_parts.append("""
## 请按以下格式输出分析结果：

**关键发现**：
1. [发现1]
2. [发现2]
...

**证据等级**：[高/中/低]

**置信度**：[0.0-1.0之间的数值]

**信息冲突**：
- [冲突1]（如无冲突写"无"）

**综合总结**：
[200字以内的综合总结]

**建议**：
1. [建议1]
2. [建议2]
""")

        return "\n".join(prompt_parts)

    def _parse_response(
        self,
        query: str,
        response: str,
        web_results: List[Dict[str, Any]],
        kb_results: List[Dict[str, Any]],
    ) -> ResearchReport:
        """
        解析 LLM 响应为结构化报告

        Args:
            query: 研究问题
            response: LLM 响应文本
            web_results: 网络搜索结果
            kb_results: 知识库搜索结果

        Returns:
            ResearchReport
        """
        report = ResearchReport(query=query)

        # 提取关键发现
        findings_match = re.search(r"\*\*关键发现\*\*[：:]\s*\n([\s\S]*?)(?=\n\*\*|$)", response)
        if findings_match:
            findings_text = findings_match.group(1)
            report.key_findings = [
                line.strip().lstrip("0123456789.-、)） ").strip()
                for line in findings_text.strip().split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]

        # 提取证据等级
        evidence_match = re.search(r"\*\*证据等级\*\*[：:]\s*(.+)", response)
        if evidence_match:
            report.evidence_level = evidence_match.group(1).strip()

        # 提取置信度
        confidence_match = re.search(r"\*\*置信度\*\*[：:]\s*([\d.]+)", response)
        if confidence_match:
            try:
                report.confidence = float(confidence_match.group(1))
            except ValueError:
                report.confidence = 0.5

        # 提取信息冲突
        conflicts_match = re.search(r"\*\*信息冲突\*\*[：:]\s*\n([\s\S]*?)(?=\n\*\*|$)", response)
        if conflicts_match:
            conflicts_text = conflicts_match.group(1)
            report.conflicts = [
                line.strip().lstrip("-•·").strip()
                for line in conflicts_text.strip().split("\n")
                if line.strip() and line.strip() not in ("无", "没有", "None")
            ]

        # 提取综合总结
        summary_match = re.search(r"\*\*综合总结\*\*[：:]\s*\n([\s\S]*?)(?=\n\*\*|$)", response)
        if summary_match:
            report.summary = summary_match.group(1).strip()

        # 提取建议
        recommendations_match = re.search(r"\*\*建议\*\*[：:]\s*\n([\s\S]*?)(?=\n\*\*|$)", response)
        if recommendations_match:
            rec_text = recommendations_match.group(1)
            report.recommendations = [
                line.strip().lstrip("0123456789.-、)） ").strip()
                for line in rec_text.strip().split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]

        # 收集来源
        report.sources = [
            r.get("url", "") for r in web_results if r.get("url")
        ]

        # 如果解析失败，使用原始响应作为总结
        if not report.summary:
            report.summary = response[:500]
        if not report.key_findings:
            report.key_findings = ["无法提取结构化发现，请查看综合总结"]

        return report

    @staticmethod
    def format_report(report: ResearchReport) -> str:
        """
        格式化研究报告为可读文本

        Args:
            report: ResearchReport 对象

        Returns:
            格式化的文本字符串
        """
        lines = [
            "=" * 60,
            f"📋 研究报告: {report.query}",
            "=" * 60,
            "",
        ]

        # 关键发现
        if report.key_findings:
            lines.append("🔍 关键发现：")
            for i, finding in enumerate(report.key_findings, 1):
                lines.append(f"  {i}. {finding}")
            lines.append("")

        # 证据等级和置信度
        lines.append(f"📊 证据等级: {report.evidence_level or '未评估'}")
        lines.append(f"🎯 置信度: {report.confidence:.0%}")
        lines.append("")

        # 信息冲突
        if report.conflicts:
            lines.append("⚠️ 信息冲突：")
            for conflict in report.conflicts:
                lines.append(f"  • {conflict}")
            lines.append("")

        # 综合总结
        if report.summary:
            lines.append("📝 综合总结：")
            lines.append(f"  {report.summary}")
            lines.append("")

        # 建议
        if report.recommendations:
            lines.append("💡 建议：")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        # 来源
        if report.sources:
            lines.append("🔗 参考来源：")
            for i, source in enumerate(report.sources, 1):
                lines.append(f"  [{i}] {source}")
            lines.append("")

        lines.append(f"📅 生成时间: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        return "\n".join(lines)
