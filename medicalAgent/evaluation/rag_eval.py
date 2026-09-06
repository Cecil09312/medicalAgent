"""
RAG 知识库评估
使用 DeepEval 内置指标评估知识库检索和回答质量
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from .config import EVAL_CONFIG, KNOWLEDGE_CONFIG
from .datasets import RAGTestCase


@dataclass
class RAGEvalResult:
    """RAG 评估结果"""
    total_cases: int = 0
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_no_hallucination: float = 0.0
    per_case_results: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"RAG 评估结果 ({self.total_cases} 个测试用例):",
            f"  Faithfulness:        {self.avg_faithfulness:.3f}  (阈值: {EVAL_CONFIG['rag']['min_faithfulness']})",
            f"  Answer Relevancy:    {self.avg_relevancy:.3f}  (阈值: {EVAL_CONFIG['rag']['min_relevancy']})",
            f"  Context Precision:   {self.avg_context_precision:.3f}  (阈值: {EVAL_CONFIG['rag']['min_context_precision']})",
            f"  Context Recall:      {self.avg_context_recall:.3f}  (阈值: {EVAL_CONFIG['rag']['min_context_recall']})",
            f"  No Hallucination:    {self.avg_no_hallucination:.3f}  (阈值: {EVAL_CONFIG['rag']['min_no_hallucination']})",
            f"  总体: {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


class RAGEvaluator:
    """RAG 知识库评估器"""

    def __init__(self, knowledge_base=None, llm_client=None):
        """
        初始化 RAG 评估器

        Args:
            knowledge_base: MedicalKnowledgeBase 实例（可选，延迟初始化）
            llm_client: LLMClient 实例（可选，用于生成答案）
        """
        self.knowledge_base = knowledge_base
        self.llm_client = llm_client

    def _get_kb(self):
        """获取知识库实例"""
        if self.knowledge_base is None:
            from knowledge.milvus_kb import MedicalKnowledgeBase
            self.knowledge_base = MedicalKnowledgeBase()
        return self.knowledge_base

    def _get_llm(self):
        """获取 LLM 客户端"""
        if self.llm_client is None:
            from core.llm_client import LLMClient
            self.llm_client = LLMClient()
        return self.llm_client

    async def _retrieve_and_generate(self, test_case: RAGTestCase) -> Dict[str, Any]:
        """
        检索上下文并生成答案

        Returns:
            包含 actual_answer, retrieved_contexts 的字典
        """
        kb = self._get_kb()
        llm = self._get_llm()

        # 检索相关上下文
        search_results = kb.search(test_case.question, top_k=KNOWLEDGE_CONFIG["top_k"])
        retrieved_contexts = [r["text"] for r in search_results if r.get("text")]

        # 基于检索上下文生成答案
        context_text = "\n\n".join(retrieved_contexts) if retrieved_contexts else "无相关信息"
        messages = [
            {"role": "system", "content": "你是医疗助手。请基于以下参考资料回答用户问题。如果资料中没有相关信息，请如实说明。\n\n参考资料：\n" + context_text},
            {"role": "user", "content": test_case.question},
        ]
        actual_answer = await llm.chat(messages=messages, temperature=0.3)

        return {
            "actual_answer": actual_answer,
            "retrieved_contexts": retrieved_contexts,
        }

    async def evaluate(self, test_cases: List[RAGTestCase]) -> RAGEvalResult:
        """
        运行 RAG 评估

        Args:
            test_cases: RAG 测试用例列表

        Returns:
            RAG 评估结果
        """
        logger.info(f"开始 RAG 评估，共 {len(test_cases)} 个测试用例")
        result = RAGEvalResult(total_cases=len(test_cases))

        # 收集所有 DeepEval 测试用例
        deepeval_test_cases = []

        for tc in test_cases:
            try:
                # 检索并生成答案
                rag_output = await self._retrieve_and_generate(tc)
                actual_answer = rag_output["actual_answer"]
                retrieved_contexts = rag_output["retrieved_contexts"]

                # 构建 DeepEval 测试用例
                deepeval_tc = LLMTestCase(
                    input=tc.question,
                    actual_output=actual_answer,
                    # Hallucination 指标要求 context 字段，与 retrieval_context 同源（检索到的上下文）
                    context=retrieved_contexts if retrieved_contexts else ["无相关信息"],
                    retrieval_context=retrieved_contexts if retrieved_contexts else ["无相关信息"],
                    expected_context=tc.expected_contexts,
                    expected_output=tc.expected_answer,
                )
                deepeval_test_cases.append(deepeval_tc)

                # 保存单条结果
                result.per_case_results.append({
                    "id": tc.id,
                    "question": tc.question,
                    "actual_answer": actual_answer[:200],
                    "retrieved_count": len(retrieved_contexts),
                })

            except Exception as e:
                logger.error(f"RAG 评估用例 {tc.id} 失败: {e}")
                result.per_case_results.append({
                    "id": tc.id,
                    "error": str(e),
                })

        if not deepeval_test_cases:
            logger.warning("没有成功的测试用例")
            result.passed = False
            return result

        # 运行 DeepEval 评估
        metrics = [
            FaithfulnessMetric(threshold=EVAL_CONFIG["rag"]["min_faithfulness"]),
            AnswerRelevancyMetric(threshold=EVAL_CONFIG["rag"]["min_relevancy"]),
            ContextualPrecisionMetric(threshold=EVAL_CONFIG["rag"]["min_context_precision"]),
            ContextualRecallMetric(threshold=EVAL_CONFIG["rag"]["min_context_recall"]),
            HallucinationMetric(threshold=EVAL_CONFIG["rag"]["min_no_hallucination"]),
        ]

        # 汇总分数容器（指标显示名 -> 结果字段 -> 分数列表）
        all_scores = {"faithfulness": [], "relevancy": [], "context_precision": [], "context_recall": [], "no_hallucination": []}

        try:
            # 同步模式运行（避免 DashScope 限流）并关闭交互面板（无终端环境会卡住/报错）；
            # deepeval 4.x 的分数需从返回值 test_results[*].metrics_data 读取，
            # 共享 metric 对象的 .score 不会被每个用例的结果回填
            from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
            eval_result = evaluate(
                deepeval_test_cases,
                metrics,
                async_config=AsyncConfig(run_async=False),
                display_config=DisplayConfig(
                    show_indicator=False, print_results=False,
                    verbose_mode=False, inspect_after_run=False,
                ),
            )

            # 指标显示名 -> 结果字段 的映射
            name_map = {
                "Faithfulness": "faithfulness",
                "Answer Relevancy": "relevancy",
                "Contextual Precision": "context_precision",
                "Contextual Recall": "context_recall",
                "Hallucination": "no_hallucination",
            }
            for tr in eval_result.test_results:
                for md in tr.metrics_data:
                    key = name_map.get(md.name)
                    if key and md.score is not None and md.success:
                        all_scores[key].append(md.score)
        except Exception as e:
            logger.error(f"DeepEval 评估运行失败: {e}")

        # 计算平均值
        result.avg_faithfulness = sum(all_scores["faithfulness"]) / len(all_scores["faithfulness"]) if all_scores["faithfulness"] else 0.0
        result.avg_relevancy = sum(all_scores["relevancy"]) / len(all_scores["relevancy"]) if all_scores["relevancy"] else 0.0
        result.avg_context_precision = sum(all_scores["context_precision"]) / len(all_scores["context_precision"]) if all_scores["context_precision"] else 0.0
        result.avg_context_recall = sum(all_scores["context_recall"]) / len(all_scores["context_recall"]) if all_scores["context_recall"] else 0.0
        result.avg_no_hallucination = sum(all_scores["no_hallucination"]) / len(all_scores["no_hallucination"]) if all_scores["no_hallucination"] else 0.0

        # 判断是否通过
        rag_config = EVAL_CONFIG["rag"]
        result.passed = (
            result.avg_faithfulness >= rag_config["min_faithfulness"]
            and result.avg_relevancy >= rag_config["min_relevancy"]
            and result.avg_context_precision >= rag_config["min_context_precision"]
            and result.avg_context_recall >= rag_config["min_context_recall"]
            and result.avg_no_hallucination >= rag_config["min_no_hallucination"]
        )

        logger.info(f"RAG 评估完成: {'PASS' if result.passed else 'FAIL'}")
        return result
