"""
共享上下文
Swarm 中所有 Agent 共享的上下文和任务状态
"""
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from .events import Event, EventType


class TaskStatus(Enum):
    """子任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """子任务数据类"""
    id: str
    type: str
    description: str
    assigned_agent: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Contribution:
    """Agent 贡献"""
    agent_id: str
    task_id: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class SharedContext:
    """共享上下文"""

    def __init__(self):
        """初始化共享上下文"""
        self.data: Dict[str, Any] = {}
        self.events: List[Event] = []
        self.task_decomposition: Dict[str, Any] = {}
        self.agent_contributions: Dict[str, List[Contribution]] = defaultdict(list)
        self._subtasks: Dict[str, SubTask] = {}

    def publish_event(self, event: Event):
        """
        发布事件
        
        Args:
            event: 事件对象
        """
        self.events.append(event)
        logger.debug(f"Event published: {event.type.value} from {event.source_agent}")

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        获取事件列表
        
        Args:
            event_type: 事件类型过滤
            source_agent: 源 Agent 过滤
            target_agent: 目标 Agent 过滤
            limit: 返回数量限制
        
        Returns:
            事件列表
        """
        filtered = self.events
        
        if event_type:
            filtered = [e for e in filtered if e.type == event_type]
        
        if source_agent:
            filtered = [e for e in filtered if e.source_agent == source_agent]
        
        if target_agent:
            filtered = [e for e in filtered if e.is_for_agent(target_agent)]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered

    def add_subtask(self, subtask: SubTask):
        """
        添加子任务
        
        Args:
            subtask: 子任务对象
        """
        self._subtasks[subtask.id] = subtask
        logger.debug(f"Subtask added: {subtask.id} assigned to {subtask.assigned_agent}")

    def get_subtask(self, task_id: str) -> Optional[SubTask]:
        """
        获取子任务
        
        Args:
            task_id: 任务 ID
        
        Returns:
            子任务对象
        """
        return self._subtasks.get(task_id)

    def get_subtasks_for_agent(self, agent_id: str) -> List[SubTask]:
        """
        获取分配给指定 Agent 的所有子任务
        
        Args:
            agent_id: Agent ID
        
        Returns:
            子任务列表
        """
        return [t for t in self._subtasks.values() if t.assigned_agent == agent_id]

    def start_subtask(self, task_id: str):
        """
        标记子任务为进行中
        
        Args:
            task_id: 任务 ID
        """
        if task_id in self._subtasks:
            subtask = self._subtasks[task_id]
            subtask.status = TaskStatus.IN_PROGRESS
            subtask.started_at = datetime.now()
            logger.debug(f"Subtask started: {task_id}")

    def complete_subtask(self, task_id: str, result: Dict[str, Any]):
        """
        标记子任务为完成
        
        Args:
            task_id: 任务 ID
            result: 任务结果
        """
        if task_id in self._subtasks:
            subtask = self._subtasks[task_id]
            subtask.status = TaskStatus.COMPLETED
            subtask.result = result
            subtask.completed_at = datetime.now()
            logger.debug(f"Subtask completed: {task_id}")

    def get_contributions(self, agent_id: Optional[str] = None) -> List[Contribution]:
        """
        获取贡献列表
        
        Args:
            agent_id: Agent ID 过滤（可选）
        
        Returns:
            贡献列表
        """
        if agent_id:
            return self.agent_contributions.get(agent_id, [])
        
        all_contributions = []
        for contributions in self.agent_contributions.values():
            all_contributions.extend(contributions)
        return all_contributions

    def get_all_completed_subtasks(self) -> List[SubTask]:
        """
        获取所有已完成的子任务
        
        Returns:
            已完成的子任务列表
        """
        return [t for t in self._subtasks.values() if t.status == TaskStatus.COMPLETED]

    def is_all_subtasks_completed(self) -> bool:
        """
        检查是否所有子任务都已完成
        
        Returns:
            是否全部完成
        """
        if not self._subtasks:
            return False
        
        return all(t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for t in self._subtasks.values())

    def set_data(self, key: str, value: Any):
        """
        设置共享数据
        
        Args:
            key: 键
            value: 值
        """
        self.data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """
        获取共享数据
        
        Args:
            key: 键
            default: 默认值
        
        Returns:
            数据值
        """
        return self.data.get(key, default)

    def get_summary(self) -> Dict[str, Any]:
        """
        获取上下文摘要
        
        Returns:
            摘要字典
        """
        return {
            "total_subtasks": len(self._subtasks),
            "completed_subtasks": len(self.get_all_completed_subtasks()),
            "total_events": len(self.events),
            "total_contributions": sum(len(c) for c in self.agent_contributions.values()),
            "shared_data_keys": list(self.data.keys())
        }
