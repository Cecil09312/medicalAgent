"""
Agent 身份管理模块
跟踪 Agent 的核心能力、专业领域、协作记录和工具使用统计
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class AgentIdentity:
    """
    Agent 身份数据类

    记录 Agent 的核心能力、专业领域和基本信息。
    """
    agent_id: str
    name: str = ""
    description: str = ""
    core_capabilities: List[str] = field(default_factory=list)
    expertise_domains: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """
        序列化为 Markdown 格式

        Returns:
            Markdown 字符串
        """
        lines = [
            f"# Agent 身份: {self.name or self.agent_id}",
            "",
            f"- **Agent ID**: `{self.agent_id}`",
            f"- **版本**: {self.version}",
            f"- **创建时间**: {self.created_at}",
            f"- **更新时间**: {self.updated_at}",
            "",
        ]

        if self.description:
            lines.extend([
                "## 描述",
                "",
                self.description,
                "",
            ])

        if self.core_capabilities:
            lines.extend([
                "## 核心能力",
                "",
            ])
            for cap in self.core_capabilities:
                lines.append(f"- {cap}")
            lines.append("")

        if self.expertise_domains:
            lines.extend([
                "## 专业领域",
                "",
            ])
            for domain in self.expertise_domains:
                lines.append(f"- {domain}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, markdown: str) -> "AgentIdentity":
        """
        从 Markdown 字符串反序列化

        Args:
            markdown: Markdown 格式的身份信息

        Returns:
            AgentIdentity 实例
        """
        identity = cls(agent_id="unknown")
        lines = markdown.strip().split("\n")

        for line in lines:
            line = line.strip()

            # 解析 Agent ID
            if line.startswith("- **Agent ID**"):
                identity.agent_id = line.split("`")[1] if "`" in line else ""

            # 解析版本
            elif line.startswith("- **版本**"):
                identity.version = line.split(":")[-1].strip()

            # 解析创建时间
            elif line.startswith("- **创建时间**"):
                identity.created_at = line.split(":")[-1].strip()

            # 解析更新时间
            elif line.startswith("- **更新时间**"):
                identity.updated_at = line.split(":")[-1].strip()

        # 解析名称（从标题）
        for line in lines:
            if line.startswith("# Agent 身份:"):
                identity.name = line.split(":")[-1].strip()
                break

        # 解析描述
        desc_lines = []
        in_desc = False
        for line in lines:
            if line.strip() == "## 描述":
                in_desc = True
                continue
            if in_desc:
                if line.startswith("## "):
                    break
                desc_lines.append(line)
        identity.description = "\n".join(desc_lines).strip()

        # 解析核心能力
        in_caps = False
        for line in lines:
            if line.strip() == "## 核心能力":
                in_caps = True
                continue
            if in_caps:
                if line.startswith("## "):
                    break
                if line.startswith("- "):
                    identity.core_capabilities.append(line[2:].strip())

        # 解析专业领域
        in_domains = False
        for line in lines:
            if line.strip() == "## 专业领域":
                in_domains = True
                continue
            if in_domains:
                if line.startswith("## "):
                    break
                if line.startswith("- "):
                    identity.expertise_domains.append(line[2:].strip())

        return identity


@dataclass
class CollaborationRecord:
    """协作记录数据类"""
    partner_agent_id: str
    task_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    outcome: str = ""
    notes: str = ""

    def to_markdown(self) -> str:
        """序列化为 Markdown"""
        return (
            f"- **{self.partner_agent_id}** | {self.task_type} | "
            f"{self.timestamp} | 结果: {self.outcome or '未记录'}"
        )


@dataclass
class ToolUsageStats:
    """工具使用统计"""
    tool_name: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    last_used: str = ""

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.call_count if self.call_count > 0 else 0.0

    def record_call(self, success: bool, latency_ms: float = 0.0):
        """记录一次调用"""
        self.call_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_used = datetime.now().isoformat()
        # 更新平均延迟
        if self.call_count == 1:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.call_count - 1) + latency_ms)
                / self.call_count
            )

    def to_markdown(self) -> str:
        """序列化为 Markdown"""
        return (
            f"- **{self.tool_name}**: {self.call_count} 次调用 | "
            f"成功率 {self.success_rate:.1%} | "
            f"平均延迟 {self.avg_latency_ms:.0f} ms"
        )


class AgentIdentityManager:
    """
    Agent 身份管理器

    负责 Agent 身份文件的创建、加载和保存。
    身份文件存储在 memory/agents/ 目录下。
    """

    def __init__(self, agents_dir: Optional[Path] = None):
        """
        初始化身份管理器

        Args:
            agents_dir: Agent 身份文件目录（默认 memory/agents/）
        """
        if agents_dir is None:
            self.agents_dir = Path(__file__).parent / "agents"
        else:
            self.agents_dir = agents_dir

        self.agents_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"AgentIdentityManager 初始化: {self.agents_dir}")

    def _get_file_path(self, agent_id: str) -> Path:
        """获取 Agent 身份文件路径"""
        return self.agents_dir / f"{agent_id}.md"

    def load_identity(self, agent_id: str) -> Optional[AgentIdentity]:
        """
        加载 Agent 身份

        Args:
            agent_id: Agent ID

        Returns:
            AgentIdentity 对象，或 None（文件不存在）
        """
        file_path = self._get_file_path(agent_id)
        if not file_path.exists():
            logger.debug(f"Agent 身份文件不存在: {file_path}")
            return None

        try:
            content = file_path.read_text(encoding="utf-8")
            identity = AgentIdentity.from_markdown(content)
            logger.info(f"加载 Agent 身份: {agent_id}")
            return identity
        except Exception as e:
            logger.error(f"加载 Agent 身份失败 ({agent_id}): {e}")
            return None

    def save_identity(self, identity: AgentIdentity) -> bool:
        """
        保存 Agent 身份

        Args:
            identity: AgentIdentity 对象

        Returns:
            是否保存成功
        """
        file_path = self._get_file_path(identity.agent_id)
        try:
            identity.updated_at = datetime.now().isoformat()
            content = identity.to_markdown()
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"保存 Agent 身份: {identity.agent_id}")
            return True
        except Exception as e:
            logger.error(f"保存 Agent 身份失败 ({identity.agent_id}): {e}")
            return False

    def create_identity(
        self,
        agent_id: str,
        name: str = "",
        description: str = "",
        core_capabilities: Optional[List[str]] = None,
        expertise_domains: Optional[List[str]] = None,
    ) -> AgentIdentity:
        """
        创建新的 Agent 身份并保存

        Args:
            agent_id: Agent ID
            name: Agent 名称
            description: 描述
            core_capabilities: 核心能力列表
            expertise_domains: 专业领域列表

        Returns:
            创建的 AgentIdentity 对象
        """
        identity = AgentIdentity(
            agent_id=agent_id,
            name=name,
            description=description,
            core_capabilities=core_capabilities or [],
            expertise_domains=expertise_domains or [],
        )
        self.save_identity(identity)
        logger.info(f"创建 Agent 身份: {agent_id} ({name})")
        return identity

    def list_agents(self) -> List[str]:
        """列出所有已注册的 Agent ID"""
        return [f.stem for f in self.agents_dir.glob("*.md")]

    def delete_identity(self, agent_id: str) -> bool:
        """
        删除 Agent 身份文件

        Args:
            agent_id: Agent ID

        Returns:
            是否删除成功
        """
        file_path = self._get_file_path(agent_id)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"删除 Agent 身份: {agent_id}")
            return True
        return False
