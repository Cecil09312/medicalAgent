"""
会话摘要模块
记录会话的完整摘要信息，包括 Agent 参与情况、关键发现、经验教训和性能指标
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class AgentParticipation:
    """Agent 参与记录"""
    agent_id: str
    role: str = ""
    task_assigned: str = ""
    task_completed: bool = False
    tools_used: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class KeyFinding:
    """关键发现"""
    finding: str
    category: str = ""  # 如 "diagnosis", "treatment", "lifestyle"
    confidence: float = 0.0  # 0.0 - 1.0
    source: str = ""  # 来源 Agent 或数据源


@dataclass
class Lesson:
    """经验教训"""
    lesson: str
    context: str = ""
    applicable_to: List[str] = field(default_factory=list)  # 适用场景
    priority: str = "normal"  # low / normal / high


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_iterations: int = 0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    total_duration_seconds: float = 0.0
    llm_api_calls: int = 0
    tokens_used: int = 0

    @property
    def tool_success_rate(self) -> float:
        """工具调用成功率"""
        if self.total_tool_calls == 0:
            return 0.0
        return self.successful_tool_calls / self.total_tool_calls

    @property
    def avg_duration_per_iteration(self) -> float:
        """每次迭代平均耗时"""
        if self.total_iterations == 0:
            return 0.0
        return self.total_duration_seconds / self.total_iterations


@dataclass
class SessionSummary:
    """
    会话摘要数据类

    包含一次完整会话的所有摘要信息。
    """
    session_id: str
    task_description: str = ""
    status: str = "unknown"  # pending / in_progress / completed / failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""

    # 参与信息
    agent_participations: List[AgentParticipation] = field(default_factory=list)

    # 关键发现
    key_findings: List[KeyFinding] = field(default_factory=list)

    # 经验教训
    lessons: List[Lesson] = field(default_factory=list)

    # 性能指标
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # 最终结果
    final_answer: str = ""

    # 原始上下文摘要
    summary: str = ""

    @classmethod
    def from_shared_context(
        cls,
        session_id: str,
        shared_context: Dict[str, Any],
    ) -> "SessionSummary":
        """
        从 SharedContext 字典构建会话摘要

        Args:
            session_id: 会话 ID
            shared_context: 共享上下文字典，可包含以下字段：
                - task_description: 任务描述
                - agents: Agent 参与信息列表
                - findings: 关键发现列表
                - lessons: 经验教训列表
                - performance: 性能指标字典
                - final_answer: 最终答案
                - status: 状态

        Returns:
            SessionSummary 实例
        """
        summary = cls(session_id=session_id)
        summary.task_description = shared_context.get("task_description", "")
        summary.status = shared_context.get("status", "unknown")
        summary.final_answer = shared_context.get("final_answer", "")
        summary.summary = shared_context.get("summary", "")

        # 解析 Agent 参与信息
        for agent_data in shared_context.get("agents", []):
            participation = AgentParticipation(
                agent_id=agent_data.get("agent_id", ""),
                role=agent_data.get("role", ""),
                task_assigned=agent_data.get("task_assigned", ""),
                task_completed=agent_data.get("task_completed", False),
                tools_used=agent_data.get("tools_used", []),
                notes=agent_data.get("notes", ""),
            )
            summary.agent_participations.append(participation)

        # 解析关键发现
        for finding_data in shared_context.get("findings", []):
            finding = KeyFinding(
                finding=finding_data.get("finding", ""),
                category=finding_data.get("category", ""),
                confidence=finding_data.get("confidence", 0.0),
                source=finding_data.get("source", ""),
            )
            summary.key_findings.append(finding)

        # 解析经验教训
        for lesson_data in shared_context.get("lessons", []):
            lesson = Lesson(
                lesson=lesson_data.get("lesson", ""),
                context=lesson_data.get("context", ""),
                applicable_to=lesson_data.get("applicable_to", []),
                priority=lesson_data.get("priority", "normal"),
            )
            summary.lessons.append(lesson)

        # 解析性能指标
        perf_data = shared_context.get("performance", {})
        summary.performance = PerformanceMetrics(
            total_iterations=perf_data.get("total_iterations", 0),
            total_tool_calls=perf_data.get("total_tool_calls", 0),
            successful_tool_calls=perf_data.get("successful_tool_calls", 0),
            failed_tool_calls=perf_data.get("failed_tool_calls", 0),
            total_duration_seconds=perf_data.get("total_duration_seconds", 0.0),
            llm_api_calls=perf_data.get("llm_api_calls", 0),
            tokens_used=perf_data.get("tokens_used", 0),
        )

        return summary

    def to_markdown(self) -> str:
        """
        序列化为 Markdown 格式

        Returns:
            Markdown 字符串
        """
        lines = [
            f"# 会话摘要: {self.session_id}",
            "",
            f"- **状态**: {self.status}",
            f"- **创建时间**: {self.created_at}",
            f"- **完成时间**: {self.completed_at or '未完成'}",
            "",
        ]

        if self.task_description:
            lines.extend([
                "## 任务描述",
                "",
                self.task_description,
                "",
            ])

        if self.summary:
            lines.extend([
                "## 摘要",
                "",
                self.summary,
                "",
            ])

        # Agent 参与
        if self.agent_participations:
            lines.extend([
                "## Agent 参与",
                "",
            ])
            for ap in self.agent_participations:
                status_str = "✅" if ap.task_completed else "❌"
                lines.append(
                    f"- {status_str} **{ap.agent_id}** ({ap.role}): "
                    f"{ap.task_assigned}"
                )
                if ap.tools_used:
                    lines.append(f"  - 工具: {', '.join(ap.tools_used)}")
                if ap.notes:
                    lines.append(f"  - 备注: {ap.notes}")
            lines.append("")

        # 关键发现
        if self.key_findings:
            lines.extend([
                "## 关键发现",
                "",
            ])
            for kf in self.key_findings:
                conf_str = f"({kf.confidence:.0%})" if kf.confidence > 0 else ""
                lines.append(f"- {kf.finding} {conf_str}")
                if kf.source:
                    lines.append(f"  - 来源: {kf.source}")
            lines.append("")

        # 经验教训
        if self.lessons:
            lines.extend([
                "## 经验教训",
                "",
            ])
            for lesson in self.lessons:
                lines.append(f"- [{lesson.priority}] {lesson.lesson}")
                if lesson.context:
                    lines.append(f"  - 上下文: {lesson.context}")
            lines.append("")

        # 性能指标
        perf = self.performance
        if perf.total_iterations > 0 or perf.total_tool_calls > 0:
            lines.extend([
                "## 性能指标",
                "",
                f"- 总迭代次数: {perf.total_iterations}",
                f"- 工具调用: {perf.total_tool_calls} "
                f"(成功 {perf.successful_tool_calls}, "
                f"失败 {perf.failed_tool_calls}, "
                f"成功率 {perf.tool_success_rate:.1%})",
                f"- 总耗时: {perf.total_duration_seconds:.1f} 秒",
                f"- LLM API 调用: {perf.llm_api_calls} 次",
                f"- Token 使用量: {perf.tokens_used}",
                "",
            ])

        # 最终答案
        if self.final_answer:
            lines.extend([
                "## 最终答案",
                "",
                self.final_answer[:500] + ("..." if len(self.final_answer) > 500 else ""),
                "",
            ])

        return "\n".join(lines)


class SessionSummaryManager:
    """
    会话摘要管理器

    负责会话摘要的保存和加载。
    摘要文件存储在 memory/swarm/session_summaries/ 目录下。
    """

    def __init__(self, summaries_dir: Optional[Path] = None):
        """
        初始化摘要管理器

        Args:
            summaries_dir: 摘要文件目录（默认 memory/swarm/session_summaries/）
        """
        if summaries_dir is None:
            self.summaries_dir = Path(__file__).parent / "swarm" / "session_summaries"
        else:
            self.summaries_dir = summaries_dir

        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"SessionSummaryManager 初始化: {self.summaries_dir}")

    def _get_file_path(self, session_id: str) -> Path:
        """获取摘要文件路径"""
        return self.summaries_dir / f"{session_id}.md"

    def save_summary(self, summary: SessionSummary) -> bool:
        """
        保存会话摘要

        Args:
            summary: SessionSummary 对象

        Returns:
            是否保存成功
        """
        file_path = self._get_file_path(summary.session_id)
        try:
            content = summary.to_markdown()
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"保存会话摘要: {summary.session_id}")
            return True
        except Exception as e:
            logger.error(f"保存会话摘要失败 ({summary.session_id}): {e}")
            return False

    def load_summary(self, session_id: str) -> Optional[SessionSummary]:
        """
        加载会话摘要

        Args:
            session_id: 会话 ID

        Returns:
            SessionSummary 对象，或 None（文件不存在）
        """
        file_path = self._get_file_path(session_id)
        if not file_path.exists():
            logger.debug(f"会话摘要文件不存在: {file_path}")
            return None

        try:
            content = file_path.read_text(encoding="utf-8")
            # 简单解析：从 Markdown 重建 SessionSummary
            summary = SessionSummary(session_id=session_id)
            summary.summary = content

            # 解析状态
            for line in content.split("\n"):
                if line.startswith("- **状态**"):
                    summary.status = line.split(":")[-1].strip()
                elif line.startswith("- **创建时间**"):
                    summary.created_at = line.split(":")[-1].strip()
                elif line.startswith("- **完成时间**"):
                    val = line.split(":")[-1].strip()
                    summary.completed_at = val if val != "未完成" else ""

            logger.info(f"加载会话摘要: {session_id}")
            return summary
        except Exception as e:
            logger.error(f"加载会话摘要失败 ({session_id}): {e}")
            return None

    def list_summaries(self) -> List[str]:
        """列出所有已保存的会话 ID"""
        return [f.stem for f in self.summaries_dir.glob("*.md")]

    def delete_summary(self, session_id: str) -> bool:
        """删除会话摘要"""
        file_path = self._get_file_path(session_id)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"删除会话摘要: {session_id}")
            return True
        return False
