"""
重排序器（Reranker） - 基于 CrossEncoder 的查询-文档精排

两阶段检索的第二阶段：向量检索召回候选集后，用交叉编码器对
"查询-文档"配对逐一打分（交叉注意力，精度高于双塔向量相似度），
按分数降序截取 top_k 返回。

默认模型 BAAI/bge-reranker-base，可通过环境变量 RERANK_MODEL 更换。
进程级单例，模型只加载一次。
"""
import os
import threading
from typing import List, Dict, Any, Optional

from loguru import logger
# 必须在导入 transformers 等库之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 复用 milvus_kb 的环境准备（.env 加载、sys.path 设置）与 HuggingFace 缓存检测
from knowledge.milvus_kb import _detect_model_cached

# 重排序模型名称（可经 .env 更换为 bge-reranker-v2-m3 等更强模型）
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# 模型缓存检测：决定离线加载还是联网下载
_model_cached = _detect_model_cached(RERANK_MODEL_NAME)
if _model_cached:
    # 已缓存：启用离线模式，避免加载时访问 huggingface.co 检查更新
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    # 未缓存：需要联网自动下载模型。milvus_kb 导入时可能因嵌入模型已缓存
    # 而设置了离线标志，这里显式关闭，否则下载会被离线模式阻止
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

from sentence_transformers import CrossEncoder


class Reranker:
    """
    重排序器（进程级单例）

    基于 sentence-transformers 的 CrossEncoder，对"查询-文档"配对
    进行相关性打分，用于向量检索候选集的精排。
    """

    _instance: Optional["Reranker"] = None
    _initialized: bool = False
    _init_lock = threading.Lock()  # 类级锁：保证多线程并发初始化时模型只加载一次

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = RERANK_MODEL_NAME):
        """
        初始化重排序模型

        Args:
            model_name: HuggingFace CrossEncoder 模型名称
        """
        # 加锁防止并发初始化（如 Web 预热线程与首个检索请求同时触发）
        with Reranker._init_lock:
            # 避免重复初始化（单例）
            if Reranker._initialized:
                return

            self.model_name = model_name
            cached = _detect_model_cached(model_name)

            # 已缓存时启用 local_files_only，避免加载时访问 huggingface.co 检查更新
            logger.info(f"加载重排序模型: {model_name}")
            self.model = CrossEncoder(model_name, local_files_only=cached)
            logger.info("重排序模型加载完成")

            Reranker._initialized = True

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        对候选文档按与查询的相关性精排

        Args:
            query: 查询文本
            documents: 候选文档列表（每项为含 "text" 字段的字典，原样返回并写入分数）
            top_k: 精排后返回的条数；None 表示返回全部候选

        Returns:
            按相关性降序排列的文档列表，每项新增 "rerank_score" 字段（CrossEncoder 原始分数）
        """
        if not documents:
            return []

        # CrossEncoder 输入为 [查询, 文档] 配对列表
        pairs = [[query, str(doc.get("text", ""))] for doc in documents]
        scores = self.model.predict(pairs)

        # numpy 数组转为 float 并写回各文档
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # 按精排分数降序
        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

        # 截断到 top_k
        if top_k is not None and top_k >= 0:
            ranked = ranked[:top_k]

        return ranked


# 模块级单例
_default_reranker: Optional[Reranker] = None


def get_default_reranker() -> Reranker:
    """
    获取进程级默认重排序器单例

    Returns:
        Reranker 单例实例
    """
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker
