"""
知识库模块
支持 Qdrant 向量数据库或内存存储
用于医疗知识的向量检索
"""
import uuid
import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed, using in-memory storage")


@dataclass
class Document:
    """文档数据结构"""
    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: List[float] = field(default_factory=list)
    score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class KnowledgeBase:
    """
    知识库
    支持 Qdrant 向量数据库或内存存储
    """

    def __init__(
        self,
        collection_name: str = "medical_kb",
        vector_size: int = 384,
        use_qdrant: bool = False,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        """
        初始化知识库

        Args:
            collection_name: 集合名称
            vector_size: 向量维度
            use_qdrant: 是否使用 Qdrant（需要安装 qdrant-client）
            qdrant_host: Qdrant 主机地址
            qdrant_port: Qdrant 端口
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE

        # 内存存储
        self._memory_storage: List[Document] = []

        # Qdrant 客户端
        if self.use_qdrant:
            try:
                self.qdrant_client = QdrantClient(
                    host=qdrant_host,
                    port=qdrant_port,
                )
                # 创建集合（如果不存在）
                collections = self.qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]

                if collection_name not in collection_names:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info(f"Created Qdrant collection: {collection_name}")

                logger.info(f"KnowledgeBase using Qdrant: {qdrant_host}:{qdrant_port}")

            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}, falling back to memory")
                self.use_qdrant = False
                self.qdrant_client = None
        else:
            self.qdrant_client = None
            logger.info("KnowledgeBase using in-memory storage")

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None,
    ) -> str:
        """
        添加文档

        Args:
            content: 文档内容
            metadata: 文档元数据
            vector: 文档向量（可选，不提供则生成随机向量）

        Returns:
            文档ID
        """
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}

        # 生成随机向量（实际应用中应使用 embedding 模型）
        if vector is None:
            import random
            vector = [random.uniform(-1, 1) for _ in range(self.vector_size)]

        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            vector=vector,
        )

        if self.use_qdrant and self.qdrant_client:
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=doc_id,
                            vector=vector,
                            payload={"content": content, **metadata},
                        )
                    ],
                )
                logger.debug(f"Added document to Qdrant: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to add to Qdrant: {e}")
                self._memory_storage.append(doc)
        else:
            self._memory_storage.append(doc)
            logger.debug(f"Added document to memory: {doc_id}")

        return doc_id

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        搜索文档

        Args:
            query_vector: 查询向量
            top_k: 返回结果数
            filters: 过滤条件

        Returns:
            文档列表
        """
        results = []

        if self.use_qdrant and self.qdrant_client:
            try:
                search_params = {
                    "collection_name": self.collection_name,
                    "query_vector": query_vector,
                    "limit": top_k,
                }

                # 构建过滤条件
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                    if conditions:
                        search_params["query_filter"] = Filter(must=conditions)

                hits = self.qdrant_client.search(**search_params)

                for hit in hits:
                    payload = hit.payload or {}
                    results.append(Document(
                        id=str(hit.id),
                        content=payload.get("content", ""),
                        metadata={k: v for k, v in payload.items() if k != "content"},
                        score=hit.score,
                    ))

                logger.debug(f"Qdrant search returned {len(results)} results")

            except Exception as e:
                logger.error(f"Qdrant search failed: {e}")

        else:
            # 内存模式：余弦相似度
            for doc in self._memory_storage:
                if doc.vector:
                    similarity = self._cosine_similarity(query_vector, doc.vector)
                    results.append(Document(
                        id=doc.id,
                        content=doc.content,
                        metadata=doc.metadata,
                        score=similarity,
                    ))

            # 按相似度排序
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]

            logger.debug(f"Memory search returned {len(results)} results")

        return results

    def bulk_add(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        批量添加文档

        Args:
            documents: 文档列表，每个包含 content, metadata, vector（可选）

        Returns:
            文档ID列表
        """
        doc_ids = []
        for doc in documents:
            doc_id = self.add_document(
                content=doc.get("content", ""),
                metadata=doc.get("metadata"),
                vector=doc.get("vector"),
            )
            doc_ids.append(doc_id)

        logger.info(f"Bulk added {len(doc_ids)} documents")
        return doc_ids

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息字典
        """
        if self.use_qdrant and self.qdrant_client:
            try:
                info = self.qdrant_client.get_collection(self.collection_name)
                return {
                    "storage_type": "qdrant",
                    "collection_name": self.collection_name,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": str(info.status),
                }
            except Exception as e:
                logger.error(f"Failed to get Qdrant stats: {e}")

        return {
            "storage_type": "memory",
            "collection_name": self.collection_name,
            "documents_count": len(self._memory_storage),
            "vector_size": self.vector_size,
        }

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度值（0-1）
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


def search_knowledge_base(
    query_vector: List[float],
    top_k: int = 5,
    collection_name: str = "medical_kb",
) -> List[Document]:
    """
    便捷函数：搜索知识库

    Args:
        query_vector: 查询向量
        top_k: 返回结果数
        collection_name: 集合名称

    Returns:
        文档列表
    """
    kb = KnowledgeBase(collection_name=collection_name)
    return kb.search(query_vector=query_vector, top_k=top_k)
