"""
短期记忆模块
支持内存和 Redis 两种存储后端
管理会话级别的对话历史
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

# 加载 .env
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
}

from .entropy_manager import MemoryEntropyManager


@dataclass
class ConversationHistory:
    """对话历史数据结构"""
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def add_message(self, role: str, content: str):
        """添加一条消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().timestamp(),
        })
        self.updated_at = datetime.now().timestamp()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """从字典反序列化"""
        history = cls(session_id=data["session_id"])
        history.messages = data.get("messages", [])
        history.created_at = data.get("created_at", datetime.now().timestamp())
        history.updated_at = data.get("updated_at", datetime.now().timestamp())
        return history


class ShortTermMemory:
    """
    短期记忆管理器（单例模式）

    支持两种存储后端：
    - memory: 内存存储（默认）
    - redis: Redis 持久化存储

    集成 MemoryEntropyManager 进行自动清理。
    """

    _instance: Optional["ShortTermMemory"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        backend: str = "memory",
        redis_config: Optional[Dict[str, Any]] = None,
        auto_clean_enabled: bool = True,
    ):
        """
        初始化短期记忆

        Args:
            backend: 存储后端 ("memory" 或 "redis")
            redis_config: Redis 配置（backend="redis" 时使用）
            auto_clean_enabled: 是否启用自动清理
        """
        # 避免重复初始化（单例）
        if ShortTermMemory._initialized:
            return

        self.backend = backend
        self.auto_clean_enabled = auto_clean_enabled
        self.entropy_manager = MemoryEntropyManager()

        # 内存存储
        self._sessions: Dict[str, ConversationHistory] = {}

        # Redis 存储
        self._redis_client = None
        if backend == "redis":
            self._init_redis(redis_config or REDIS_CONFIG)

        ShortTermMemory._initialized = True
        logger.info(f"ShortTermMemory 初始化完成 (backend={backend})")

    def _init_redis(self, config: Dict[str, Any]):
        """初始化 Redis 连接"""
        try:
            import redis
            self._redis_client = redis.Redis(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                db=config.get("db", 0),
                decode_responses=True,
            )
            self._redis_client.ping()
            logger.info("Redis 连接成功")
        except Exception as e:
            logger.warning(f"Redis 连接失败，回退到内存存储: {e}")
            self._redis_client = None
            self.backend = "memory"

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    def create_session(self, session_id: str) -> ConversationHistory:
        """
        创建新会话

        Args:
            session_id: 会话 ID

        Returns:
            创建的 ConversationHistory 对象
        """
        history = ConversationHistory(session_id=session_id)

        if self.backend == "redis" and self._redis_client:
            self._redis_client.set(
                f"stm:{session_id}",
                json.dumps(history.to_dict(), ensure_ascii=False),
            )
        else:
            self._sessions[session_id] = history

        logger.debug(f"创建会话: {session_id}")
        return history

    def _get_session(self, session_id: str) -> Optional[ConversationHistory]:
        """获取会话（内部方法）"""
        if self.backend == "redis" and self._redis_client:
            data = self._redis_client.get(f"stm:{session_id}")
            if data:
                return ConversationHistory.from_dict(json.loads(data))
            return None
        else:
            return self._sessions.get(session_id)

    def _save_session(self, history: ConversationHistory):
        """保存会话（内部方法）"""
        if self.backend == "redis" and self._redis_client:
            self._redis_client.set(
                f"stm:{history.session_id}",
                json.dumps(history.to_dict(), ensure_ascii=False),
            )
        else:
            self._sessions[history.session_id] = history

    def clear_session(self, session_id: str):
        """
        清除指定会话

        Args:
            session_id: 会话 ID
        """
        if self.backend == "redis" and self._redis_client:
            self._redis_client.delete(f"stm:{session_id}")
        elif session_id in self._sessions:
            del self._sessions[session_id]

        logger.info(f"清除会话: {session_id}")

    # ------------------------------------------------------------------
    # 消息操作
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str):
        """
        向会话添加消息

        如果会话不存在，会自动创建。

        Args:
            session_id: 会话 ID
            role: 消息角色 (user / assistant / system / tool)
            content: 消息内容
        """
        history = self._get_session(session_id)
        if history is None:
            history = self.create_session(session_id)

        history.add_message(role=role, content=content)
        self._save_session(history)

        # 自动清理
        if self.auto_clean_enabled and len(history.messages) > 50:
            self._auto_clean_session(session_id)

    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        获取最近 N 条消息

        Args:
            session_id: 会话 ID
            limit: 返回消息数量上限

        Returns:
            消息列表（按时间正序）
        """
        history = self._get_session(session_id)
        if history is None:
            return []

        return history.messages[-limit:]

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        openai_format: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        获取会话历史（OpenAI 格式）

        Args:
            session_id: 会话 ID
            limit: 返回消息数量上限（None = 全部）
            openai_format: 是否返回 OpenAI 格式（仅保留 role 和 content）

        Returns:
            消息列表
        """
        history = self._get_session(session_id)
        if history is None:
            return []

        messages = history.messages
        if limit:
            messages = messages[-limit:]

        if openai_format:
            return [
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ]

        return messages

    # ------------------------------------------------------------------
    # 自动清理
    # ------------------------------------------------------------------

    def _auto_clean_session(self, session_id: str):
        """对指定会话执行自动清理"""
        history = self._get_session(session_id)
        if history is None:
            return

        original_count = len(history.messages)
        cleaned = self.entropy_manager.auto_clean(history.messages)
        history.messages = cleaned
        self._save_session(history)

        logger.info(
            f"会话 {session_id} 自动清理: {original_count} → {len(cleaned)} 条消息"
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_session_ids(self) -> List[str]:
        """获取所有会话 ID"""
        if self.backend == "redis" and self._redis_client:
            keys = self._redis_client.keys("stm:*")
            return [k.replace("stm:", "") for k in keys]
        else:
            return list(self._sessions.keys())

    def get_session_count(self) -> int:
        """获取会话总数"""
        if self.backend == "redis" and self._redis_client:
            return len(self._redis_client.keys("stm:*"))
        else:
            return len(self._sessions)
