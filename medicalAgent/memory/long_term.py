"""
长期记忆模块
基于 Mem0 云服务的长期记忆存储与检索
支持优雅降级（Mem0 不可用时回退到本地存储）
"""
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
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

MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
# Mem0 云端记忆统一归属的用户标识（保存与检索必须一致，否则互相不可见）
MEM0_USER_ID = "medical_system"

# 检测 Mem0 可用性
try:
    from mem0 import MemoryClient as Mem0Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("mem0ai 未安装，长期记忆将使用本地文件存储")


class LongTermMemory:
    """
    长期记忆管理器

    优先使用 Mem0 云服务进行向量存储和语义检索。
    如果 Mem0 不可用（未安装或 API Key 为空），优雅降级为本地 JSON 文件存储。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化长期记忆

        Args:
            config: Mem0 配置字典（可选，默认从 .env 读取 MEM0_API_KEY）
        """
        self.config = config or {"api_key": MEM0_API_KEY}
        self._mem0_client = None
        self._use_mem0 = False

        # 本地降级存储
        self._local_storage_dir = _project_root / "memory" / "data" / "long_term"
        self._local_storage_dir.mkdir(parents=True, exist_ok=True)
        self._local_memories: List[Dict[str, Any]] = []

        # 加载本地已保存的记忆（降级存储重启后不丢失历史）
        self._load_local()

        # 尝试初始化 Mem0
        self._init_mem0()

        logger.info(
            f"LongTermMemory 初始化完成 "
            f"(use_mem0={self._use_mem0}, local_storage={self._local_storage_dir})"
        )

    def _init_mem0(self):
        """初始化 Mem0 客户端"""
        if not MEM0_AVAILABLE:
            logger.info("Mem0 库未安装，使用本地存储")
            return

        api_key = self.config.get("api_key", "")
        if not api_key:
            logger.info("Mem0 API Key 未配置，使用本地存储")
            return

        try:
            # mem0ai 2.x 平台版：直接以 api_key 构造 MemoryClient
            self._mem0_client = Mem0Memory(api_key=api_key)
            self._use_mem0 = True
            logger.info("Mem0 云服务初始化成功")
        except Exception as e:
            logger.warning(f"Mem0 初始化失败，回退到本地存储: {e}")
            self._use_mem0 = False

    # ------------------------------------------------------------------
    # 添加会话摘要
    # ------------------------------------------------------------------

    def add_session_summary(
        self,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        保存会话摘要到长期记忆

        Args:
            summary: 会话摘要文本
            metadata: 附加元数据（如 agent_id、task_type 等）
            session_id: 会话 ID

        Returns:
            是否保存成功
        """
        record = {
            "summary": summary,
            "metadata": metadata or {},
            "session_id": session_id or "",
            "timestamp": datetime.now().isoformat(),
        }

        if self._use_mem0 and self._mem0_client:
            try:
                # mem0ai 2.x：messages 为位置参数；user_id 统一固定，保证检索时可见
                self._mem0_client.add(
                    summary,
                    user_id=MEM0_USER_ID,
                    metadata=metadata or {},
                )
                logger.info(f"会话摘要已保存到 Mem0 (session={session_id})")
                return True
            except Exception as e:
                logger.warning(f"Mem0 保存失败，回退到本地存储: {e}")

        # 本地存储
        self._local_memories.append(record)
        self._save_local()
        logger.info(f"会话摘要已保存到本地 (session={session_id})")
        return True

    # ------------------------------------------------------------------
    # 语义搜索
    # ------------------------------------------------------------------

    def search_similar_sessions(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索相似的会话

        Args:
            query: 查询文本
            top_k: 返回结果数量
            metadata_filter: 元数据过滤条件

        Returns:
            相似会话列表
        """
        if self._use_mem0 and self._mem0_client:
            try:
                # mem0ai 2.x：user_id 需放入 filters；与保存时使用同一标识
                results = self._mem0_client.search(
                    query=query,
                    filters={"user_id": MEM0_USER_ID},
                    limit=top_k,
                )
                # mem0ai 2.x 平台返回 {"results": [...]}，统一解包为列表
                if isinstance(results, dict):
                    results = results.get("results", [])
                logger.debug(f"Mem0 搜索 '{query[:30]}...' 返回 {len(results)} 条结果")
                return results
            except Exception as e:
                logger.warning(f"Mem0 搜索失败，回退到本地搜索: {e}")

        # 本地简单搜索（关键词匹配）
        return self._local_search(query, top_k)

    def _local_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """本地关键词搜索（降级方案）"""
        query_keywords = set(query.lower().split())
        scored_results = []

        for record in self._local_memories:
            summary_lower = record["summary"].lower()
            score = sum(1 for kw in query_keywords if kw in summary_lower)
            if score > 0:
                scored_results.append({
                    **record,
                    "score": score / len(query_keywords) if query_keywords else 0,
                })

        # 按得分排序
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        results = scored_results[:top_k]

        logger.debug(f"本地搜索 '{query[:30]}...' 返回 {len(results)} 条结果")
        return results

    # ------------------------------------------------------------------
    # 本地存储
    # ------------------------------------------------------------------

    def _save_local(self):
        """保存本地记忆到文件"""
        storage_file = self._local_storage_dir / "memories.json"
        try:
            with open(storage_file, "w", encoding="utf-8") as f:
                json.dump(self._local_memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"本地记忆保存失败: {e}")

    def _load_local(self):
        """从文件加载本地记忆"""
        storage_file = self._local_storage_dir / "memories.json"
        if storage_file.exists():
            try:
                with open(storage_file, "r", encoding="utf-8") as f:
                    self._local_memories = json.load(f)
                logger.info(f"加载 {len(self._local_memories)} 条本地记忆")
            except Exception as e:
                logger.error(f"本地记忆加载失败: {e}")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_memory_count(self) -> int:
        """获取记忆总数"""
        if self._use_mem0:
            # Mem0 不直接提供 count 接口，返回本地计数
            return len(self._local_memories)
        return len(self._local_memories)

    def clear_all(self):
        """清除所有本地记忆"""
        self._local_memories.clear()
        self._save_local()
        logger.info("已清除所有本地长期记忆")
