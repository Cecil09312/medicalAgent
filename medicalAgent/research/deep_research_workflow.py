"""
深度研究工作流
整合网络搜索、知识库和证据合成
支持查询规划、并行搜索和迭代优化
"""
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from .web_search import WebSearchTool
from .evidence_synthesizer import EvidenceSynthesizer, ResearchReport
from .knowledge_base import Document


class DeepResearchWorkflow:
    """
    深度研究工作流
    整合多个信息源，进行结构化研究
    """

    def __init__(
        self,
        llm_client=None,
        use_web_search: bool = True,
        use_knowledge_base: bool = True,
    ):
        """
        初始化深度研究工作流

        Args:
            llm_client: LLM 客户端实例
            use_web_search: 是否使用网络搜索
            use_knowledge_base: 是否使用知识库
        """
        self.use_web_search = use_web_search
        self.use_knowledge_base = use_knowledge_base

        # 初始化 LLM 客户端
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            from core.llm_client import LLMClient
            self.llm_client = LLMClient()

        # 初始化工具
        self.web_search_tool = WebSearchTool() if use_web_search else None
        self.evidence_synthesizer = EvidenceSynthesizer(llm_client=self.llm_client)

        logger.info(
            f"DeepResearchWorkflow initialized: "
            f"web_search={use_web_search}, knowledge_base={use_knowledge_base}"
        )

    def get_knowledge_base(self) -> Optional[Any]:
        """
        获取知识库实例（单例）

        返回 Milvus 医疗知识库实例（MedicalKnowledgeBase），
        供 _search_milvus 按文本进行语义检索。

        Returns:
            MedicalKnowledgeBase 实例或 None
        """
        if not hasattr(self, "_kb_instance"):
            if self.use_knowledge_base:
                # 延迟导入：避免模块导入期加载嵌入模型等重依赖
                from knowledge.milvus_kb import MedicalKnowledgeBase
                self._kb_instance = MedicalKnowledgeBase()
            else:
                self._kb_instance = None
        return self._kb_instance

    async def run(
        self,
        question: str,
        max_web_results: int = 10,
        max_kb_results: int = 5,
    ) -> ResearchReport:
        """
        执行深度研究

        Args:
            question: 研究问题
            max_web_results: 最大网络搜索结果数
            max_kb_results: 最大知识库结果数

        Returns:
            ResearchReport 研究报告
        """
        logger.info(f"Starting deep research: {question}")

        # 1. 查询规划：分解为子查询
        sub_queries = await self._plan_queries(question)
        logger.info(f"Planned {len(sub_queries)} sub-queries: {sub_queries}")

        # 2. 并行搜索（网络 + 知识库）
        web_results = []
        kb_results = []

        tasks = []

        # 网络搜索任务
        if self.use_web_search and self.web_search_tool:
            for sq in sub_queries:
                tasks.append(
                    self.web_search_tool.search(sq, max_results=max_web_results)
                )

        # 知识库搜索任务（使用 Milvus）
        if self.use_knowledge_base:
            for sq in sub_queries:
                tasks.append(self._search_milvus(sq, top_k=max_kb_results))

        # 并行执行所有搜索
        if tasks:
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 分离网络搜索和知识库结果
            idx = 0
            if self.use_web_search and self.web_search_tool:
                for _ in sub_queries:
                    if idx < len(all_results) and not isinstance(all_results[idx], Exception):
                        web_results.extend(all_results[idx])
                    idx += 1

            if self.use_knowledge_base:
                for _ in sub_queries:
                    if idx < len(all_results) and not isinstance(all_results[idx], Exception):
                        kb_results.extend(all_results[idx])
                    idx += 1

        logger.info(f"Collected {len(web_results)} web results, {len(kb_results)} KB results")

        # 3. 证据合成
        report = await self.evidence_synthesizer.synthesize(
            query=question,
            web_results=[
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in web_results
            ],
            kb_results=[
                {"content": r.content, "score": r.score}
                for r in kb_results
            ],
        )

        logger.info(f"Deep research completed for: {question}")
        return report

    async def _search_milvus(self, query: str, top_k: int = 5) -> List[Any]:
        """
        搜索 Milvus 知识库

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            搜索结果列表
        """
        kb = self.get_knowledge_base()
        if not kb:
            return []

        try:
            # 改动说明：原实现使用随机向量检索，检索结果完全无效；
            # 现改为调用 search_by_text，由知识库内部先对查询文本生成
            # 嵌入向量，再执行真实的语义向量检索
            results = kb.search_by_text(query, top_k=top_k)

            # 兼容性适配：search_by_text 返回字典（含 text / score 等键），
            # 转换为 Document 对象，保持 run() 中 r.content / r.score 的访问方式不变
            documents = [
                Document(content=item.get("text", ""), score=item.get("score", 0.0))
                for item in results
            ]
            logger.debug(f"Milvus search returned {len(documents)} results for: {query}")
            return documents

        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []

    async def _plan_queries(self, question: str) -> List[str]:
        """
        查询规划：使用 LLM 将问题分解为子查询

        Args:
            question: 原始问题

        Returns:
            子查询列表（2-3个）
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个查询规划助手。请将用户的问题分解为2-3个更具体的子查询，用于搜索医疗信息。每行输出一个子查询，不要编号，不要其他内容。",
                },
                {"role": "user", "content": question},
            ]

            response = await self.llm_client.chat(messages, temperature=0.3)

            # 解析子查询
            sub_queries = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            # 限制数量
            sub_queries = sub_queries[:3]

            # 确保至少有原始问题
            if not sub_queries:
                sub_queries = [question]

            return sub_queries

        except Exception as e:
            logger.error(f"Query planning failed: {e}")
            return [question]

    async def research_with_refinement(
        self,
        question: str,
        max_iterations: int = 2,
    ) -> ResearchReport:
        """
        带迭代优化的深度研究

        Args:
            question: 研究问题
            max_iterations: 最大迭代次数

        Returns:
            ResearchReport 研究报告
        """
        logger.info(f"Starting research with refinement: {question}")

        current_report = None

        for iteration in range(max_iterations):
            logger.info(f"Research iteration {iteration + 1}/{max_iterations}")

            # 执行研究
            current_report = await self.run(question)

            # 质量检查
            quality_ok = await self._check_quality(question, current_report)

            if quality_ok:
                logger.info(f"Research quality is sufficient at iteration {iteration + 1}")
                break
            else:
                logger.warning(f"Research quality insufficient, refining...")
                # 可以在这里添加查询优化逻辑

        return current_report

    async def _check_quality(self, question: str, report: ResearchReport) -> bool:
        """
        检查研究报告质量

        Args:
            question: 原始问题
            report: 研究报告

        Returns:
            质量是否达标
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个研究质量评估专家。请判断以下研究报告是否充分回答了用户的问题。只需回答'是'或'否'。",
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n报告摘要：{report.summary}\n\n置信度：{report.confidence}\n\n这个回答是否充分？请回答'是'或'否'。",
                },
            ]

            response = await self.llm_client.chat(messages, temperature=0.0)

            quality_ok = "是" in response and "否" not in response
            logger.debug(f"Quality check result: {'pass' if quality_ok else 'fail'}")
            return quality_ok

        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return True  # 失败时默认通过


def deep_research() -> DeepResearchWorkflow:
    """
    便捷函数：创建深度研究工作流实例

    Returns:
        DeepResearchWorkflow 实例
    """
    return DeepResearchWorkflow()
