"""
医疗知识库 - 基于 Milvus Lite 的向量检索
使用 BAAI/bge-small-zh-v1.5 嵌入模型
"""
import os
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys

from loguru import logger
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# 项目路径
_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = _PROJECT_ROOT / "knowledge" / "data"

# 从 .env 读取知识库配置
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

_MILVUS_DB_PATH = str(_PROJECT_ROOT / os.getenv("KB_DB_PATH", "knowledge/data/milvus_lite.db"))
EMBEDDING_MODEL_NAME = os.getenv("KB_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("KB_EMBEDDING_DIM", "512"))
SIMILARITY_METRIC = os.getenv("KB_SIMILARITY_METRIC", "COSINE")
DEFAULT_CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "50"))
DEFAULT_TOP_K = int(os.getenv("KB_TOP_K", "5"))
DEFAULT_BATCH_SIZE = int(os.getenv("KB_BATCH_SIZE", "64"))
DEFAULT_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "medical_knowledge")

# 重排序（Rerank）配置：默认关闭
# 开启后检索变为两阶段：向量召回 RERANK_CANDIDATE_K 条候选 -> CrossEncoder 精排 -> 返回 top_k
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
try:
    RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "15"))
except ValueError:
    RERANK_CANDIDATE_K = 15

# 检索缓存（Redis 防护组件）：向量检索+重排是当前最贵的查询，缓存结果并做
# 击穿互斥重建（singleflight）与 TTL 抖动（雪崩保护）；Redis 不可用时自动直通
KB_SEARCH_CACHE_ENABLED = os.getenv("KB_SEARCH_CACHE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
try:
    KB_SEARCH_CACHE_TTL = int(os.getenv("KB_SEARCH_CACHE_TTL", "300"))
except ValueError:
    KB_SEARCH_CACHE_TTL = 300
# top_k 超过该值的查询结果集过大，不缓存（避免逼近大 key 阈值）
_KB_CACHE_MAX_TOP_K = 20
# 缓存世代号键：知识库内容变更时 INCR，旧世代键靠 TTL 自然淘汰（无需 SCAN 清理）
_KB_CACHE_GEN_KEY = "kb:search:gen"

from core.redis_guard import get_shared_client, singleflight_get_or_build

# HuggingFace 本地缓存路径检测
_HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
_MODEL_CACHE_DIR = os.path.join(_HF_HOME, "hub")


def _detect_model_cached(model_name: str) -> bool:
    """检测嵌入模型是否已缓存到本地（供模块导入早期使用，不依赖 logger）"""
    st_model_path = Path(_HF_HOME) / "sentence_transformers" / model_name.replace("/", "_")
    hub_model_dir = Path(_MODEL_CACHE_DIR) / f"models--{model_name.replace('/', '--')}"
    return st_model_path.exists() or hub_model_dir.exists()


# 若模型已缓存到本地，则在导入 sentence_transformers 之前启用离线模式，
# 避免加载模型时访问 huggingface.co 检查更新而产生长时间网络重试（每次重试约 20 秒）
if _detect_model_cached(EMBEDDING_MODEL_NAME):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer


def _is_model_cached(model_name: str) -> bool:
    """
    检测嵌入模型是否已缓存到本地

    Args:
        model_name: HuggingFace 模型名称

    Returns:
        是否已缓存
    """
    # sentence-transformers 默认缓存路径
    st_cache = Path(_HF_HOME) / "sentence_transformers"
    model_short_name = model_name.replace("/", "_")
    st_model_path = st_cache / model_short_name

    # 也检查 models--xxx 格式（HuggingFace Hub 缓存）
    hub_model_dir = Path(_MODEL_CACHE_DIR) / f"models--{model_name.replace('/', '--')}"

    cached = st_model_path.exists() or hub_model_dir.exists()
    if cached:
        logger.info(f"嵌入模型已检测到本地缓存: {model_name}")
    else:
        logger.info(f"嵌入模型未找到本地缓存，将自动下载: {model_name}")
    return cached


def _kb_cache_generation(client) -> int:
    """当前缓存世代号（知识变更时 INCR；读取失败按 0 处理，仅影响命中率）"""
    try:
        return int(client.get(_KB_CACHE_GEN_KEY) or 0)
    except Exception:
        return 0


def _invalidate_search_cache() -> None:
    """知识库内容变更后失效全部检索缓存（INCR 世代号；Redis 不可用时靠 TTL 兜底）"""
    client = get_shared_client()
    if client is None:
        return
    try:
        client.incr(_KB_CACHE_GEN_KEY)
        logger.info("检索缓存已失效（知识库内容变更）")
    except Exception as e:
        logger.debug(f"检索缓存失效失败（等待 TTL 自然过期兜底）: {e}")


def _kb_cache_key(client, query: str, top_k: int, doc_type: Optional[str]) -> str:
    """缓存键 = 世代号 + 查询指纹（query/top_k/doc_type/重排配置共同影响结果）"""
    fingerprint = hashlib.sha256(
        f"{query}|{top_k}|{doc_type or ''}|{RERANK_ENABLED}|{RERANK_CANDIDATE_K}".encode("utf-8")
    ).hexdigest()[:24]
    return f"kb:search:g{_kb_cache_generation(client)}:{fingerprint}"


class MedicalKnowledgeBase:
    """
    医疗知识库（单例模式）

    基于 Milvus Lite 本地向量数据库，使用 BAAI/bge-small-zh-v1.5 嵌入模型。
    支持文档添加、语义搜索、集合删除和文档计数。
    """

    _instance: Optional["MedicalKnowledgeBase"] = None
    _initialized: bool = False
    _init_lock = threading.Lock()  # 类级锁：保证多线程并发初始化时模型只加载一次

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        db_path: str = _MILVUS_DB_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        """
        初始化知识库

        Args:
            db_path: Milvus Lite 数据库文件路径
            model_name: 嵌入模型名称
            collection_name: 集合名称
        """
        # 加锁防止并发初始化（如 Web 预热线程与首个搜索请求同时触发）
        with MedicalKnowledgeBase._init_lock:
            # 避免重复初始化（单例）
            if MedicalKnowledgeBase._initialized and self._client is not None:
                return

            self.db_path = db_path
            self.model_name = model_name
            self.collection_name = collection_name

            # 确保数据目录存在
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            # 检测模型缓存
            cached = _is_model_cached(model_name)

            # 初始化嵌入模型
            # 已缓存时启用 local_files_only，避免加载时访问 huggingface.co 检查更新
            # （离线环境每次网络重试约 20 秒，会严重拖慢首次搜索）
            logger.info(f"加载嵌入模型: {model_name}")
            self.embedding_model = SentenceTransformer(model_name, local_files_only=cached)
            logger.info("嵌入模型加载完成")

            # 初始化 Milvus Lite 客户端
            logger.info(f"连接 Milvus Lite: {db_path}")
            self._client = MilvusClient(uri=db_path)

            # 确保集合存在
            self._ensure_collection()

            MedicalKnowledgeBase._initialized = True
            logger.info(f"MedicalKnowledgeBase 初始化完成 (collection={collection_name})")

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def _ensure_collection(self):
        """确保集合存在，若不存在则创建"""
        if self._client.has_collection(self.collection_name):
            # 确保集合已加载：Milvus Lite 在新进程中连接已有集合时可能处于 released 状态
            self._client.load_collection(self.collection_name)
            logger.debug(f"集合已存在并加载: {self.collection_name}")
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=256)

        # FLAT 暴力索引：无需构建/落盘索引文件，避免 Milvus Lite 在中文路径下
        # 重建索引失败的问题；知识库文档量小，检索精度与速度均可满足
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type=SIMILARITY_METRIC)

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"创建集合: {self.collection_name} (dim={EMBEDDING_DIM}, metric={SIMILARITY_METRIC})")

    # ------------------------------------------------------------------
    # 文本分块
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """
        将长文本切分为多个块

        Args:
            text: 原始文本
            chunk_size: 每块最大字符数
            overlap: 相邻块重叠字符数

        Returns:
            文本块列表
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap

        # 过滤空块
        return [c for c in chunks if c]

    # ------------------------------------------------------------------
    # 嵌入
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: List[str],
        doc_type: str = "general",
        source: str = "",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        pages: Optional[List[Optional[int]]] = None,
    ) -> int:
        """
        添加文档到知识库

        Args:
            texts: 文档文本列表
            doc_type: 文档类型 (lifestyle / disease_classification / clinical_guideline / general)
            source: 文档来源（文件名）
            chunk_size: 分块大小
            overlap: 分块重叠
            pages: 每个文本对应的页码（可选，与 texts 一一对应；无页码传 None）

        Returns:
            成功插入的文档块数量
        """
        all_chunks: List[str] = []
        chunk_meta: List[Dict[str, Any]] = []

        for idx, text in enumerate(texts):
            # 该文本的页码（如有），其所有分块继承该页码
            page = pages[idx] if pages and idx < len(pages) else None
            chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for c_idx, chunk in enumerate(chunks):
                # doc_id 纳入页码：同一 PDF 逐页导入时 source/idx/c_idx 相同，需靠页码区分主键
                doc_id = hashlib.md5(f"{source}_{page}_{idx}_{c_idx}".encode()).hexdigest()[:16]
                all_chunks.append(chunk)
                chunk_meta.append({
                    "id": doc_id,
                    "doc_type": doc_type,
                    "source": source or f"doc_{idx}",
                    "page": page,
                })

        if not all_chunks:
            logger.warning("没有有效的文本块可插入")
            return 0

        # 批量嵌入（分批处理，避免 OOM）
        batch_size = DEFAULT_BATCH_SIZE
        all_embeddings: List[List[float]] = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = self._embed(batch)
            all_embeddings.extend(embeddings)

        # 构造插入数据（page 为动态字段，无页码时不写入）
        data = []
        for emb, chunk, meta in zip(all_embeddings, all_chunks, chunk_meta):
            item = {
                "id": meta["id"],
                "embedding": emb,
                "text": chunk,
                "doc_type": meta["doc_type"],
                "source": meta["source"],
            }
            if meta.get("page") is not None:
                item["page"] = meta["page"]
            data.append(item)

        self._client.insert(collection_name=self.collection_name, data=data)
        logger.info(f"成功插入 {len(data)} 个文档块 (doc_type={doc_type}, source={source})")
        # 内容已变更：失效检索缓存，确保后续查询不命中旧结果
        _invalidate_search_cache()
        return len(data)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索（带 Redis 检索缓存）

        缓存开启且 Redis 可用时：结果按 query/top_k/doc_type/重排配置指纹缓存
        KB_SEARCH_CACHE_TTL 秒（含随机抖动）；miss 时经 singleflight 互斥重建，
        防止热点查询失效瞬间并发请求全部击穿到底层检索（缓存击穿保护）。
        Redis 不可用或 top_k 超限时直通检索，行为与无缓存完全一致。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            doc_type: 过滤文档类型（可选）

        Returns:
            搜索结果列表，每项包含 text / score / doc_type / source / page（精排开启时另含 rerank_score）
        """
        if not (KB_SEARCH_CACHE_ENABLED and top_k <= _KB_CACHE_MAX_TOP_K):
            return self._search_uncached(query=query, top_k=top_k, doc_type=doc_type)
        client = get_shared_client()
        if client is None:
            return self._search_uncached(query=query, top_k=top_k, doc_type=doc_type)
        cache_key = _kb_cache_key(client, query, top_k, doc_type)
        return singleflight_get_or_build(
            client,
            cache_key,
            builder=lambda: self._search_uncached(query=query, top_k=top_k, doc_type=doc_type),
            ttl_seconds=KB_SEARCH_CACHE_TTL,
        )

    def _search_uncached(
        self,
        query: str,
        top_k: int,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义检索（不含缓存层）

        开启 RERANK_ENABLED 时执行两阶段检索：向量召回 RERANK_CANDIDATE_K 条候选，
        经 CrossEncoder 精排后返回 top_k 条（结果含 rerank_score 字段，按其降序）；
        关闭时为单阶段向量检索，行为与精排上线前一致。
        """
        query_embedding = self._embed([query])[0]

        search_params = {"metric_type": SIMILARITY_METRIC}

        filter_expr = None
        if doc_type:
            filter_expr = f'doc_type == "{doc_type}"'

        # 重排序开启时扩大向量召回候选池（第一阶段保召回率），精排后再截断到 top_k；
        # 关闭时召回数量与现状完全一致
        recall_limit = max(top_k, RERANK_CANDIDATE_K) if RERANK_ENABLED else top_k

        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=recall_limit,
            search_params=search_params,
            output_fields=["text", "doc_type", "source", "page"],
            filter=filter_expr,
        )

        hits = []
        for hit_list in results:
            for hit in hit_list:
                hits.append({
                    "text": hit["entity"].get("text", ""),
                    "score": hit["distance"],
                    "doc_type": hit["entity"].get("doc_type", ""),
                    "source": hit["entity"].get("source", ""),
                    # 页码为动态字段，无页码的文档返回 None
                    "page": hit["entity"].get("page"),
                })

        # 第二阶段：CrossEncoder 重排序（可选，默认关闭）
        # 懒加载：关闭时不导入 reranker 模块、不加载模型，执行路径与单阶段检索完全一致
        if RERANK_ENABLED and hits:
            from knowledge.reranker import get_default_reranker
            candidate_count = len(hits)
            reranker = get_default_reranker()
            hits = reranker.rerank(query, hits, top_k=top_k)
            logger.debug(f"重排序完成: 候选 {candidate_count} 条，精排返回 {len(hits)} 条")

        logger.debug(f"搜索 '{query[:30]}...' 返回 {len(hits)} 条结果")
        return hits

    def search_by_text(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        按文本查询语义检索

        内部先对查询文本生成嵌入向量，再基于向量执行相似度检索。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 透传给 search() 的可选过滤参数（如 doc_type）

        Returns:
            搜索结果列表，每项包含 text / score / doc_type / source / page
        """
        # 说明：search() 的首个参数即为查询文本，其内部会先调用 _embed 生成
        # 查询向量再执行向量检索，因此这里直接透传文本，避免重复嵌入计算
        return self.search(query=query, top_k=top_k, **kwargs)

    def delete_by_source(self, source: str) -> int:
        """
        按来源删除文档块（知识库增量更新时替换旧版本使用）

        Args:
            source: 文档来源标识（add_documents 时的 source 参数）

        Returns:
            删除的文档块数量
        """
        if not source:
            return 0
        if not self._client.has_collection(self.collection_name):
            return 0
        try:
            result = self._client.delete(
                collection_name=self.collection_name,
                filter=f'source == "{source}"',
            )
            delete_count = 0
            if isinstance(result, dict):
                delete_count = int(result.get("delete_count", 0) or 0)
            logger.info(f"已按 source 删除 {delete_count} 个文档块 (source={source})")
            _invalidate_search_cache()  # 内容已变更：失效检索缓存
            return delete_count
        except Exception as e:
            logger.error(f"按 source 删除文档块失败 (source={source}): {e}")
            raise

    def delete_collection(self):
        """删除当前集合"""
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)
            logger.info(f"已删除集合: {self.collection_name}")
            _invalidate_search_cache()  # 内容已变更：失效检索缓存

    def count_documents(self) -> int:
        """
        返回集合中的文档块数量

        Returns:
            文档块数量
        """
        if not self._client.has_collection(self.collection_name):
            return 0
        stats = self._client.get_collection_stats(self.collection_name)
        count = int(stats.get("row_count", 0))
        return count
