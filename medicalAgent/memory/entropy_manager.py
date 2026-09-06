"""
记忆熵管理器
负责去重、清理、压缩历史记忆，降低记忆系统的熵值
"""
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class MessageRecord:
    """消息记录"""
    session_id: str
    role: str
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class MemoryEntropyManager:
    """
    记忆熵管理器

    通过去重、清理过期记忆、压缩历史等手段控制记忆系统复杂度。
    """

    def __init__(self, max_age_days: int = 30, recent_keep: int = 10):
        """
        初始化熵管理器

        Args:
            max_age_days: 记忆最大保留天数
            recent_keep: 压缩时保留最近消息数量
        """
        self.max_age_days = max_age_days
        self.recent_keep = recent_keep
        logger.debug(f"MemoryEntropyManager 初始化: max_age_days={max_age_days}, recent_keep={recent_keep}")

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(content: str) -> str:
        """计算内容的 MD5 哈希"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def deduplicate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对消息列表进行去重（基于 MD5 哈希）

        保留首次出现的消息，移除后续重复项。

        Args:
            messages: 消息列表，每项包含 content 字段

        Returns:
            去重后的消息列表
        """
        seen_hashes = set()
        unique_messages = []
        removed_count = 0

        for msg in messages:
            content = msg.get("content", "")
            msg_hash = self._content_hash(content)

            if msg_hash in seen_hashes:
                removed_count += 1
                continue

            seen_hashes.add(msg_hash)
            unique_messages.append(msg)

        if removed_count > 0:
            logger.info(f"消息去重: 移除 {removed_count} 条重复消息，保留 {len(unique_messages)} 条")

        return unique_messages

    def deduplicate_sessions(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对会话列表进行去重（基于会话摘要哈希）

        Args:
            sessions: 会话列表，每项包含 summary 或 messages 字段

        Returns:
            去重后的会话列表
        """
        seen_hashes = set()
        unique_sessions = []
        removed_count = 0

        for session in sessions:
            # 优先使用 summary 字段，否则拼接所有消息 content
            summary = session.get("summary", "")
            if not summary:
                messages = session.get("messages", [])
                summary = " ".join(m.get("content", "") for m in messages)

            session_hash = self._content_hash(summary)

            if session_hash in seen_hashes:
                removed_count += 1
                continue

            seen_hashes.add(session_hash)
            unique_sessions.append(session)

        if removed_count > 0:
            logger.info(f"会话去重: 移除 {removed_count} 条重复会话，保留 {len(unique_sessions)} 条")

        return unique_sessions

    # ------------------------------------------------------------------
    # 清理过期记忆
    # ------------------------------------------------------------------

    def cleanup_old_memories(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清理超过 max_age_days 的旧消息

        Args:
            messages: 消息列表，每项包含 timestamp 字段（Unix 时间戳）

        Returns:
            清理后的消息列表
        """
        cutoff_time = (datetime.now() - timedelta(days=self.max_age_days)).timestamp()
        original_count = len(messages)

        filtered = [
            msg for msg in messages
            if msg.get("timestamp", 0) >= cutoff_time
        ]

        removed_count = original_count - len(filtered)
        if removed_count > 0:
            logger.info(f"清理过期记忆: 移除 {removed_count} 条（>{self.max_age_days} 天）")

        return filtered

    # ------------------------------------------------------------------
    # 压缩会话历史
    # ------------------------------------------------------------------

    def compress_session_history(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        压缩会话历史：保留最近 N 条消息，将更早的消息压缩为一条摘要

        Args:
            messages: 完整消息列表

        Returns:
            压缩后的消息列表（摘要 + 最近消息）
        """
        if len(messages) <= self.recent_keep:
            return messages

        # 分割：旧消息 + 最近消息
        old_messages = messages[:-self.recent_keep]
        recent_messages = messages[-self.recent_keep:]

        # 将旧消息压缩为一条摘要
        old_summary_parts = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]  # 截取前 100 字符
            old_summary_parts.append(f"[{role}] {content}")

        compressed_content = (
            f"（以下为 {len(old_messages)} 条历史消息的压缩摘要）\n"
            + "\n".join(old_summary_parts)
        )

        compressed_message = {
            "role": "system",
            "content": compressed_content,
            "timestamp": old_messages[0].get("timestamp", datetime.now().timestamp()),
            "compressed": True,
        }

        logger.info(f"压缩会话历史: {len(old_messages)} 条旧消息 → 1 条摘要，保留 {len(recent_messages)} 条最近消息")

        return [compressed_message] + recent_messages

    # ------------------------------------------------------------------
    # 熵值估算
    # ------------------------------------------------------------------

    def estimate_entropy(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        估算消息列表的熵值（统计信息）

        Args:
            messages: 消息列表

        Returns:
            熵值统计字典
        """
        if not messages:
            return {
                "total_messages": 0,
                "unique_messages": 0,
                "duplicate_ratio": 0.0,
                "avg_content_length": 0.0,
                "total_content_length": 0,
            }

        # 统计唯一消息
        unique_contents = set()
        total_length = 0

        for msg in messages:
            content = msg.get("content", "")
            unique_contents.add(self._content_hash(content))
            total_length += len(content)

        total = len(messages)
        unique = len(unique_contents)
        duplicate_ratio = 1.0 - (unique / total) if total > 0 else 0.0

        stats = {
            "total_messages": total,
            "unique_messages": unique,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "avg_content_length": round(total_length / total, 2) if total > 0 else 0.0,
            "total_content_length": total_length,
        }

        logger.debug(f"熵值估算: total={total}, unique={unique}, dup_ratio={duplicate_ratio:.2%}")
        return stats

    # ------------------------------------------------------------------
    # 自动清理
    # ------------------------------------------------------------------

    def auto_clean(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        自动清理：去重 + 清理过期 + 压缩

        组合调用上述方法，一键降低记忆熵值。

        Args:
            messages: 原始消息列表

        Returns:
            清理后的消息列表
        """
        logger.info(f"自动清理开始: 输入 {len(messages)} 条消息")

        # Step 1: 去重
        result = self.deduplicate_messages(messages)

        # Step 2: 清理过期
        result = self.cleanup_old_memories(result)

        # Step 3: 压缩
        result = self.compress_session_history(result)

        logger.info(f"自动清理完成: 输出 {len(result)} 条消息")
        return result
