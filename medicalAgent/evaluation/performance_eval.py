"""
端到端性能评估
纯指标采集，不依赖 DeepEval
"""
import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from .config import EVAL_CONFIG


@dataclass
class PerfEvalResult:
    """性能评估结果"""
    total_queries: int = 0
    iterations_per_query: int = 1
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    avg_tokens_prompt: float = 0.0
    avg_tokens_completion: float = 0.0
    avg_tokens_total: float = 0.0
    throughput_qpm: float = 0.0
    avg_tool_calls: float = 0.0
    per_query_results: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        """生成摘要"""
        perf_config = EVAL_CONFIG["performance"]
        lines = [
            f"性能评估结果 ({self.total_queries} 个查询 x {self.iterations_per_query} 次迭代):",
            f"  Latency P50:    {self.latency_p50:.2f}s  (阈值: {perf_config['max_latency_p50']}s)",
            f"  Latency P95:    {self.latency_p95:.2f}s  (阈值: {perf_config['max_latency_p95']}s)",
            f"  Latency P99:    {self.latency_p99:.2f}s  (阈值: {perf_config['max_latency_p99']}s)",
            f"  Tokens (avg):   prompt={self.avg_tokens_prompt:.0f}, completion={self.avg_tokens_completion:.0f}, total={self.avg_tokens_total:.0f}",
            f"  Tokens/query:   {self.avg_tokens_total:.0f}  (阈值: {perf_config['max_tokens_per_query']})",
            f"  Throughput:     {self.throughput_qpm:.1f} QPM  (阈值: {perf_config['min_throughput_qpm']} QPM)",
            f"  Tool Calls:     {self.avg_tool_calls:.1f} (avg per query)",
            f"  总体: {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


class PerformanceEvaluator:
    """端到端性能评估器"""

    def __init__(self, coordinator=None, agents: Optional[Dict[str, Any]] = None):
        """
        初始化性能评估器

        Args:
            coordinator: SwarmCoordinator 实例（可选）
            agents: Agent 实例字典（可选）
        """
        self.coordinator = coordinator
        self.agents = agents or {}

    def _get_coordinator(self):
        """获取 SwarmCoordinator 实例"""
        if self.coordinator is not None:
            return self.coordinator

        try:
            from swarm.swarm_coordinator import SwarmCoordinator
            from agents.consultation_agent import ConsultationAgent
            from agents.diagnostic_agent import DiagnosticAgent
            from agents.research_agent import ResearchAgent

            agents = {
                "consultation_agent": ConsultationAgent(),
                "diagnostic_agent": DiagnosticAgent(),
                "research_agent": ResearchAgent(),
            }
            self.coordinator = SwarmCoordinator(worker_agents=agents)
        except Exception as e:
            logger.error(f"无法创建 SwarmCoordinator: {e}")
            raise

        return self.coordinator

    async def _run_single_query(self, question: str) -> Dict[str, Any]:
        """
        运行单次查询并采集性能指标

        Returns:
            包含 latency, token_usage, tool_calls 的字典
        """
        coordinator = self._get_coordinator()

        start_time = time.time()
        result = await coordinator.process(question)
        elapsed = time.time() - start_time

        # 提取 Token 使用信息（如果可用）
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(coordinator, "llm_client") and hasattr(coordinator.llm_client, "token_stats"):
            token_usage = coordinator.llm_client.token_stats

        # 提取工具调用次数
        tool_call_count = 0
        if "swarm_summary" in result:
            summary = result.get("swarm_summary", {})
            if isinstance(summary, dict):
                subtasks = summary.get("subtasks", [])
                tool_call_count = len(subtasks)

        return {
            "latency": elapsed,
            "token_usage": token_usage,
            "tool_calls": tool_call_count,
            "answer_length": len(result.get("answer", "")),
        }

    async def evaluate(
        self,
        test_cases: List[Dict[str, str]],
        iterations: int = 3,
    ) -> PerfEvalResult:
        """
        运行性能评估

        Args:
            test_cases: 测试用例列表，每项为 {"question": "..."}
            iterations: 每个用例重复运行次数

        Returns:
            性能评估结果
        """
        logger.info(f"开始性能评估，{len(test_cases)} 个查询 x {iterations} 次迭代")
        result = PerfEvalResult(
            total_queries=len(test_cases),
            iterations_per_query=iterations,
        )

        all_latencies = []
        all_prompt_tokens = []
        all_completion_tokens = []
        all_total_tokens = []
        all_tool_calls = []

        for tc in test_cases:
            question = tc.get("question", "")
            query_latencies = []

            for i in range(iterations):
                try:
                    perf = await self._run_single_query(question)
                    query_latencies.append(perf["latency"])
                    all_latencies.append(perf["latency"])

                    tokens = perf["token_usage"]
                    all_prompt_tokens.append(tokens.get("prompt_tokens", 0))
                    all_completion_tokens.append(tokens.get("completion_tokens", 0))
                    all_total_tokens.append(tokens.get("total_tokens", 0))
                    all_tool_calls.append(perf["tool_calls"])

                except Exception as e:
                    logger.warning(f"性能测试 [{question[:30]}...] 第 {i+1} 次失败: {e}")

            # 记录单查询结果
            if query_latencies:
                result.per_query_results.append({
                    "question": question[:50],
                    "latency_avg": statistics.mean(query_latencies),
                    "latency_min": min(query_latencies),
                    "latency_max": max(query_latencies),
                })

        # 计算统计指标
        if all_latencies:
            sorted_latencies = sorted(all_latencies)
            n = len(sorted_latencies)
            result.latency_p50 = sorted_latencies[int(n * 0.50)]
            result.latency_p95 = sorted_latencies[min(int(n * 0.95), n - 1)]
            result.latency_p99 = sorted_latencies[min(int(n * 0.99), n - 1)]

        if all_total_tokens:
            result.avg_tokens_prompt = statistics.mean(all_prompt_tokens) if all_prompt_tokens else 0
            result.avg_tokens_completion = statistics.mean(all_completion_tokens) if all_completion_tokens else 0
            result.avg_tokens_total = statistics.mean(all_total_tokens) if all_total_tokens else 0

        if all_tool_calls:
            result.avg_tool_calls = statistics.mean(all_tool_calls)

        # 计算吞吐量（查询/分钟）
        total_time = sum(all_latencies) if all_latencies else 1
        result.throughput_qpm = (len(all_latencies) / total_time) * 60 if total_time > 0 else 0

        # 判断是否通过
        perf_config = EVAL_CONFIG["performance"]
        result.passed = (
            result.latency_p50 <= perf_config["max_latency_p50"]
            and result.latency_p95 <= perf_config["max_latency_p95"]
            and result.latency_p99 <= perf_config["max_latency_p99"]
            and result.avg_tokens_total <= perf_config["max_tokens_per_query"]
            and result.throughput_qpm >= perf_config["min_throughput_qpm"]
        )

        logger.info(f"性能评估完成: {'PASS' if result.passed else 'FAIL'}")
        return result
