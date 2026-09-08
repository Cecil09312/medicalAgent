"""
Swarm 协作评估
使用自定义指标评估任务分解、Agent 选择和结果汇总质量
"""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from .config import EVAL_CONFIG
from .datasets import SwarmTestCase


def _run_llm_chat_sync(messages, temperature: float = 0.1) -> str:
    """
    在任意线程中同步调用异步 LLMClient.chat

    deepeval 4.x 用线程池并发执行指标 measure，工作线程没有事件循环，
    直接 asyncio.get_event_loop() 会抛错。此处为当前线程按需创建循环。
    """
    from core.llm_client import LLMClient
    llm = LLMClient()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("当前线程已有运行中的事件循环")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(llm.chat(messages=messages, temperature=temperature))


@dataclass
class SwarmEvalResult:
    """Swarm 评估结果"""
    total_cases: int = 0
    avg_decomposition_quality: float = 0.0
    avg_agent_selection_accuracy: float = 0.0
    avg_coordination_efficiency: float = 0.0
    avg_synthesis_quality: float = 0.0
    avg_coordination_time: float = 0.0
    per_case_results: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"Swarm 评估结果 ({self.total_cases} 个测试用例):",
            f"  Decomposition Quality:   {self.avg_decomposition_quality:.3f}  (阈值: {EVAL_CONFIG['swarm']['min_decomposition_quality']})",
            f"  Agent Selection Accuracy:{self.avg_agent_selection_accuracy:.3f}  (阈值: {EVAL_CONFIG['swarm']['min_agent_selection_accuracy']})",
            f"  Coordination Efficiency: {self.avg_coordination_efficiency:.3f}",
            f"  Synthesis Quality:       {self.avg_synthesis_quality:.3f}  (阈值: {EVAL_CONFIG['swarm']['min_synthesis_quality']})",
            f"  Avg Coordination Time:   {self.avg_coordination_time:.1f}s  (阈值: {EVAL_CONFIG['swarm']['max_coordination_time']}s)",
            f"  总体: {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


class DecompositionQualityMetric(BaseMetric):
    """任务分解质量评估（LLM-as-Judge）"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "DecompositionQuality"

    def measure(self, test_case: LLMTestCase) -> float:
        """评估任务分解的合理性"""
        from core.llm_client import LLMClient

        # deepeval 4.x 自定义字段通过 additional_metadata 传递
        metadata = getattr(test_case, "additional_metadata", None) or {}
        decomposition_info = metadata.get("decomposition_info", "") if isinstance(metadata, dict) else ""
        expected_behavior = metadata.get("expected_behavior", "") if isinstance(metadata, dict) else ""

        prompt = f"""请评估以下 Swarm 任务分解的质量。

**用户问题：** {test_case.input}
**期望行为：** {expected_behavior}
**实际分解：** {decomposition_info}

评估维度：
1. 子任务划分是否合理（不重叠、不遗漏）
2. 每个子任务的 Agent 分配是否正确
3. 子任务数量是否适当

请严格按以下 JSON 格式返回：
{{"score": 0.0-1.0, "reason": "评估理由"}}
"""
        try:
            response = _run_llm_chat_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            result = json.loads(response)
            self.score = float(result.get("score", 0.0))
            self.reason = result.get("reason", "")
        except Exception as e:
            logger.warning(f"Decomposition quality metric failed: {e}")
            self.score = 0.5
            self.reason = f"评估失败: {e}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return getattr(self, "success", False)


class SynthesisQualityMetric(BaseMetric):
    """结果汇总质量评估（LLM-as-Judge）"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "SynthesisQuality"

    def measure(self, test_case: LLMTestCase) -> float:
        """评估结果汇总的质量"""
        from core.llm_client import LLMClient

        prompt = f"""请评估以下多 Agent 协作的最终答案质量。

**用户问题：** {test_case.input}
**最终答案：** {test_case.actual_output}

评估维度：
1. 是否整合了所有 Agent 的贡献
2. 是否逻辑连贯、无重复
3. 是否包含必要的免责声明
4. 语言是否通俗易懂

请严格按以下 JSON 格式返回：
{{"score": 0.0-1.0, "reason": "评估理由"}}
"""
        try:
            response = _run_llm_chat_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            result = json.loads(response)
            self.score = float(result.get("score", 0.0))
            self.reason = result.get("reason", "")
        except Exception as e:
            logger.warning(f"Synthesis quality metric failed: {e}")
            self.score = 0.5
            self.reason = f"评估失败: {e}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return getattr(self, "success", False)


class SwarmEvaluator:
    """Swarm 协作评估器"""

    def __init__(self, coordinator=None, agents: Optional[Dict[str, Any]] = None):
        """
        初始化 Swarm 评估器

        Args:
            coordinator: 编排器实例（可选）
            agents: Agent 实例字典（可选）
        """
        self.coordinator = coordinator
        self.agents = agents or {}

    def _get_coordinator(self):
        """获取编排器实例（经工厂构建，与主链路同一组装方式）"""
        if self.coordinator is not None:
            return self.coordinator

        # 动态创建
        try:
            from orchestrator.factory import build_orchestrator

            self.coordinator = build_orchestrator(worker_agents=self.agents or None)
        except Exception as e:
            logger.error(f"无法创建编排器: {e}")
            raise

        return self.coordinator

    async def _run_swarm(self, test_case: SwarmTestCase) -> Dict[str, Any]:
        """运行 Swarm 并收集详细信息"""
        coordinator = self._get_coordinator()

        start_time = time.time()
        result = await coordinator.process(test_case.question)
        elapsed = time.time() - start_time

        # 提取分解信息
        decomposition_info = ""
        if "swarm_summary" in result:
            summary = result["swarm_summary"]
            if isinstance(summary, dict):
                subtasks = summary.get("subtasks", [])
                contributions = summary.get("contributions", {})
                decomposition_info = f"子任务数: {len(subtasks)}, 贡献: {list(contributions.keys()) if isinstance(contributions, dict) else 'N/A'}"

        return {
            "answer": result.get("answer", ""),
            "elapsed_time": elapsed,
            "decomposition_info": decomposition_info,
            "timeout_occurred": result.get("timeout_occurred", False),
            "swarm_summary": result.get("swarm_summary", {}),
        }

    def _calculate_agent_selection_score(
        self, expected_agents: List[str], actual_summary: Dict
    ) -> float:
        """计算 Agent 选择准确率"""
        if not expected_agents:
            return 1.0

        # 从 swarm_summary 中提取实际使用的 Agent
        actual_agents = set()
        if isinstance(actual_summary, dict):
            contributions = actual_summary.get("contributions", {})
            if isinstance(contributions, dict):
                actual_agents = set(contributions.keys())
            subtasks = actual_summary.get("subtasks", [])
            for st in subtasks:
                if isinstance(st, dict) and "assigned_agent" in st:
                    actual_agents.add(st["assigned_agent"])

        if not actual_agents:
            return 0.5  # 无法判断时给中间分

        expected_set = set(expected_agents)
        intersection = len(expected_set & actual_agents)
        union = len(expected_set | actual_agents)
        return intersection / union if union > 0 else 0.0

    async def evaluate(self, test_cases: List[SwarmTestCase]) -> SwarmEvalResult:
        """
        运行 Swarm 评估

        Args:
            test_cases: Swarm 测试用例列表

        Returns:
            Swarm 评估结果
        """
        logger.info(f"开始 Swarm 评估，共 {len(test_cases)} 个测试用例")
        result = SwarmEvalResult(total_cases=len(test_cases))

        agent_selection_scores = []
        coordination_times = []
        deepeval_test_cases = []

        for tc in test_cases:
            try:
                # 运行 Swarm
                swarm_output = await self._run_swarm(tc)
                elapsed = swarm_output["elapsed_time"]
                coordination_times.append(elapsed)

                # Agent 选择准确率
                agent_score = self._calculate_agent_selection_score(
                    tc.expected_agents, swarm_output.get("swarm_summary", {})
                )
                agent_selection_scores.append(agent_score)

                # 构建 DeepEval 测试用例
                # deepeval 4.x 的 LLMTestCase 为 pydantic 模型，不支持动态属性，
                # 自定义字段（期望行为、分解信息）通过 additional_metadata 传递
                deepeval_tc = LLMTestCase(
                    input=tc.question,
                    actual_output=swarm_output["answer"],
                    additional_metadata={
                        "expected_behavior": tc.expected_behavior or "",
                        "decomposition_info": swarm_output["decomposition_info"] or "",
                    },
                )
                deepeval_test_cases.append(deepeval_tc)

                result.per_case_results.append({
                    "id": tc.id,
                    "expected_agents": tc.expected_agents,
                    "expected_mode": tc.expected_coordination_mode,
                    "elapsed_time": elapsed,
                    "agent_selection_score": agent_score,
                    "timeout": swarm_output["timeout_occurred"],
                })

            except Exception as e:
                logger.error(f"Swarm 评估用例 {tc.id} 失败: {e}")
                result.per_case_results.append({"id": tc.id, "error": str(e)})

        # 运行 DeepEval 自定义指标
        if deepeval_test_cases:
            from deepeval import evaluate

            decomp_metric = DecompositionQualityMetric(
                threshold=EVAL_CONFIG["swarm"]["min_decomposition_quality"]
            )
            synth_metric = SynthesisQualityMetric(
                threshold=EVAL_CONFIG["swarm"]["min_synthesis_quality"]
            )

            # 收集分数容器：指标显示名 -> 每用例分数列表
            decomp_scores = []
            synth_scores = []

            try:
                # 同步模式运行（避免 DashScope 限流）并关闭交互面板；
                # deepeval 4.x 的分数需从返回值 test_results[*].metrics_data 读取
                from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
                eval_result = evaluate(
                    deepeval_test_cases,
                    [decomp_metric, synth_metric],
                    async_config=AsyncConfig(run_async=False),
                    display_config=DisplayConfig(
                        show_indicator=False, print_results=False,
                        verbose_mode=False, inspect_after_run=False,
                    ),
                )

                # 自定义指标 __name__ -> 分数列表 的映射
                name_map = {
                    "DecompositionQuality": decomp_scores,
                    "SynthesisQuality": synth_scores,
                }
                for tr in eval_result.test_results:
                    for md in tr.metrics_data:
                        target = name_map.get(md.name)
                        if target is not None and md.score is not None and md.success:
                            target.append(md.score)
            except Exception as e:
                logger.error(f"DeepEval Swarm 评估失败: {e}")

            result.avg_decomposition_quality = sum(decomp_scores) / len(decomp_scores) if decomp_scores else 0.0
            result.avg_synthesis_quality = sum(synth_scores) / len(synth_scores) if synth_scores else 0.0

        # 汇总
        result.avg_agent_selection_accuracy = (
            sum(agent_selection_scores) / len(agent_selection_scores)
            if agent_selection_scores else 0.0
        )
        result.avg_coordination_time = (
            sum(coordination_times) / len(coordination_times)
            if coordination_times else 0.0
        )
        # 协调效率：时间越短越好，以阈值为基准归一化
        max_time = EVAL_CONFIG["swarm"]["max_coordination_time"]
        if result.avg_coordination_time > 0:
            result.avg_coordination_efficiency = max(
                0.0, 1.0 - (result.avg_coordination_time / max_time)
            )
        else:
            result.avg_coordination_efficiency = 0.0

        # 判断是否通过
        swarm_config = EVAL_CONFIG["swarm"]
        result.passed = (
            result.avg_decomposition_quality >= swarm_config["min_decomposition_quality"]
            and result.avg_agent_selection_accuracy >= swarm_config["min_agent_selection_accuracy"]
            and result.avg_synthesis_quality >= swarm_config["min_synthesis_quality"]
            and result.avg_coordination_time <= swarm_config["max_coordination_time"]
        )

        logger.info(f"Swarm 评估完成: {'PASS' if result.passed else 'FAIL'}")
        return result
