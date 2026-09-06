"""
事件系统
定义 Swarm 中的事件类型和事件数据
"""
import uuid
from enum import Enum
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field


class EventType(Enum):
    """事件类型枚举"""
    TASK_DECOMPOSED = "task_decomposed"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"
    CONTEXT_UPDATED = "context_updated"
    AGENT_QUESTION = "agent_question"
    AGENT_ANSWER = "agent_answer"
    SWARM_STARTED = "swarm_started"
    SWARM_COMPLETED = "swarm_completed"


@dataclass
class Event:
    """事件数据类"""
    type: EventType
    source_agent: str
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    target_agents: Optional[List[str]] = None  # None 表示广播给所有 Agent

    def is_for_agent(self, agent_id: str) -> bool:
        """
        检查事件是否针对指定 Agent
        
        Args:
            agent_id: Agent ID
        
        Returns:
            是否针对该 Agent
        """
        if self.target_agents is None:
            # 广播事件，所有 Agent 都能收到
            return True
        return agent_id in self.target_agents

    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为字典
        
        Returns:
            字典表示
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "source_agent": self.source_agent,
            "target_agents": self.target_agents,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
