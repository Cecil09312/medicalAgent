"""
Agent 能力评估
使用 DeepEval 指标 + 自定义指标评估 Agent 的工具选择和回答质量
"""
import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

from .config import EVAL_CONFIG
from .datasets import AgentTestCase


def _run_llm_chat_sync(messages, temperature: float = 0.1) -> str:
    """
    在任意线程中同步调用异步 LLMClient.chat

    deepeval 4.x 即使在 run_async=False 时也会用线程池（deepeval-timeout-worker）
    并发执行指标 measure，这些工作线程没有事件循环，直接 asyncio.get_event_loop()
    会抛 "There is no current event loop in thread"。此处为当前线程按需创建循环。

    Args:
        messages: 消息列表
        temperature: 采样温度

    Returns:
        LLM 响应文本
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
class AgentEvalResult:
    """Agent 评估结果"""
    agent_id: str = ""
    total_cases: int = 0
    avg_tool_accuracy: float = 0.0
    avg_answer_correctness: float = 0.0
    avg_task_completion: float = 0.0
    avg_response_quality: float = 0.0
    per_case_results: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"Agent [{self.agent_id}] 评估结果 ({self.total_cases} 个测试用例):",
            f"  Tool Accuracy:       {self.avg_tool_accuracy:.3f}  (阈值: {EVAL_CONFIG['agent']['min_tool_accuracy']})",
            f"  Answer Correctness:  {self.avg_answer_correctness:.3f}  (阈值: {EVAL_CONFIG['agent']['min_answer_correctness']})",
            f"  Task Completion:     {self.avg_task_completion:.3f}  (阈值: {EVAL_CONFIG['agent']['min_task_completion']})",
            f"  Response Quality:    {self.avg_response_quality:.3f}  (阈值: {EVAL_CONFIG['agent']['min_response_quality']})",
            f"  总体: {'PASS' if self.passed else 'FAIL'}",
        ]
        return "\n".join(lines)


class MedicalResponseQualityMetric(BaseMetric):
    """自定义医疗回答质量评分（完整性/准确性/安全性）"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "MedicalResponseQuality"

    def measure(self, test_case: LLMTestCase) -> float:
        """使用 LLM-as-Judge 评估医疗回答质量"""
        from core.llm_client import LLMClient

        prompt = f"""请评估以下医疗回答的质量，从完整性、准确性、安全性三个维度打分（0-1）。

**用户问题：** {test_case.input}
**回答内容：** {test_case.actual_output}

请严格按以下 JSON 格式返回：
{{"completeness": 0.0-1.0, "accuracy": 0.0-1.0, "safety": 0.0-1.0, "overall": 0.0-1.0, "reason": "评分理由"}}
"""
        try:
            response = _run_llm_chat_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            # 解析 JSON
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            result = json.loads(response)
            self.score = float(result.get("overall", 0.0))
            self.reason = result.get("reason", "")
        except Exception as e:
            logger.warning(f"Medical quality metric failed: {e}")
            self.score = 0.5
            self.reason = f"评估失败: {e}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return getattr(self, "success", False)


class TaskCompletionMetric(BaseMetric):
    """任务完成度评估（LLM-as-Judge）"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "TaskCompletion"

    def measure(self, test_case: LLMTestCase) -> float:
        """判断任务是否完成"""
        from core.llm_client import LLMClient

        expected_behavior = ""
        metadata = getattr(test_case, "additional_metadata", None)
        if isinstance(metadata, dict):
            expected_behavior = metadata.get("expected_behavior", "")
        prompt = f"""请判断以下回答是否完成了用户的任务。

**用户问题：** {test_case.input}
**期望行为：** {expected_behavior}
**实际回答：** {test_case.actual_output}

请严格按以下 JSON 格式返回：
{{"completed": true/false, "score": 0.0-1.0, "reason": "判断理由"}}
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
            logger.warning(f"Task completion metric failed: {e}")
            self.score = 0.5
            self.reason = f"评估失败: {e}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return getattr(self, "success", False)


class MedicalAnswerCorrectnessMetric(BaseMetric):
    """医疗回答正确率评估（LLM-as-Judge）

    替代 deepeval 旧版的 AnswerCorrectnessMetric（deepeval 4.x 已移除该内置指标），
    对比实际回答与期望答案的核心事实与结论一致性进行打分。
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    @property
    def __name__(self):
        return "MedicalAnswerCorrectness"

    def measure(self, test_case: LLMTestCase) -> float:
        """使用 LLM-as-Judge 评估回答正确性"""
        from core.llm_client import LLMClient

        expected_output = getattr(test_case, "expected_output", "") or ""
        prompt = f"""请评估以下回答与期望答案的一致性和正确性，打分（0-1）。

**用户问题：** {test_case.input}
**期望答案：** {expected_output}
**实际回答：** {test_case.actual_output}

评分标准：核心事实和结论与期望答案一致得高分；存在事实错误或结论相悖得低分；
实际回答更详细但核心一致同样可得高分。

请严格按以下 JSON 格式返回：
{{"score": 0.0-1.0, "reason": "评分理由"}}
"""
        try:
            response = _run_llm_chat_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            # 解析 JSON
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            result = json.loads(response)
            self.score = float(result.get("score", 0.0))
            self.reason = result.get("reason", "")
        except Exception as e:
            logger.warning(f"Answer correctness metric failed: {e}")
            self.score = 0.5
            self.reason = f"评估失败: {e}"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return getattr(self, "success", False)


class AgentEvaluator:
    """Agent 能力评估器"""

    def __init__(self, agents: Optional[Dict[str, Any]] = None, llm_client: Optional[Any] = None):
        """
        初始化 Agent 评估器

        Args:
            agents: Agent 实例字典 {agent_id: agent_instance}
            llm_client: LLMClient 实例
        """
        self.agents = agents or {}
        self.llm_client = llm_client

    def _get_agent(self, agent_id: str):
        """获取 Agent 实例"""
        if agent_id in self.agents:
            return self.agents[agent_id]
        # 尝试动态创建
        try:
            if agent_id == "consultation_agent":
                from agents.consultation_agent import ConsultationAgent
                return ConsultationAgent()
            elif agent_id == "diagnostic_agent":
                from agents.diagnostic_agent import DiagnosticAgent
                return DiagnosticAgent()
            elif agent_id == "research_agent":
                from agents.research_agent import ResearchAgent
                return ResearchAgent()
        except Exception as e:
            logger.error(f"无法创建 Agent {agent_id}: {e}")
        return None

    def _determine_agent_for_case(self, test_case: AgentTestCase) -> str:
        """根据测试用例确定使用哪个 Agent"""
        # 根据期望工具推断 Agent
        tool_to_agent = {
            "recommend_lifestyle": "consultation_agent",
            "search_knowledge": "consultation_agent",
            "assess_risk": "diagnostic_agent",
            "analyze_symptoms": "diagnostic_agent",
            "disease_code": "consultation_agent",
            "clinical_guideline": "research_agent",
            "deep_research": "research_agent",
        }
        if test_case.expected_tools:
            primary_tool = test_case.expected_tools[0]
            return tool_to_agent.get(primary_tool, "consultation_agent")
        return "consultation_agent"

    async def _run_agent(self, agent, question: str) -> Dict[str, Any]:
        """运行 Agent 并收集工具调用链"""
        from core.agent_loop import AgentLoop

        agent_loop = AgentLoop(max_iterations=5, max_tool_calls=5)
        result = await agent_loop.run(agent, {"question": question})

        # 提取工具调用链
        tool_calls = []
        if hasattr(agent_loop, "state_manager") and agent_loop.state_manager:
            # task_id 为 run() 内部生成的随机 uuid，外部无法获知；
            # 每个评估用例使用独立 AgentLoop，其 StateManager 仅含本次运行的 state，
            # 故直接遍历并按 agent_id 校验
            for state in agent_loop.state_manager.states.values():
                if getattr(state, "agent_id", "") != getattr(agent, "agent_id", ""):
                    continue
                for step in getattr(state, "intermediate_results", []):
                    # 中间结果结构:
                    # {'iteration':..,'result':{'llm_response':{'tool_calls':[{'name':..,'arguments':..}]}}}
                    llm_resp = (step.get("result") or {}).get("llm_response") or {}
                    for tc in (llm_resp.get("tool_calls") or []):
                        name = tc.get("name")
                        if name:
                            tool_calls.append(name)

        return {
            "answer": result.get("answer", ""),
            "tool_calls": tool_calls,
            "iterations": getattr(agent_loop, "tool_call_count", 0),
        }

    async def evaluate(self, test_cases: List[AgentTestCase], agent_id: Optional[str] = None) -> AgentEvalResult:
        """
        运行 Agent 评估

        Args:
            test_cases: Agent 测试用例列表
            agent_id: 指定 Agent ID（可选，自动推断）

        Returns:
            Agent 评估结果
        """
        logger.info(f"开始 Agent 评估，共 {len(test_cases)} 个测试用例")
        result = AgentEvalResult(total_cases=len(test_cases), agent_id=agent_id or "auto")

        tool_accuracy_scores = []
        task_completion_scores = []
        deepeval_test_cases = []

        for tc in test_cases:
            try:
                # 确定 Agent
                target_agent_id = agent_id or self._determine_agent_for_case(tc)
                agent = self._get_agent(target_agent_id)
                if not agent:
                    logger.warning(f"Agent {target_agent_id} 不可用，跳过用例 {tc.id}")
                    continue

                # 运行 Agent
                agent_output = await self._run_agent(agent, tc.question)
                actual_answer = agent_output["answer"]
                actual_tools = agent_output["tool_calls"]

                # 计算工具准确率
                expected_set = set(tc.expected_tools)
                actual_set = set(actual_tools) if actual_tools else set()
                if expected_set:
                    intersection = len(expected_set & actual_set)
                    union = len(expected_set | actual_set)
                    tool_accuracy = intersection / union if union > 0 else 0.0
                else:
                    tool_accuracy = 1.0
                tool_accuracy_scores.append(tool_accuracy)

                # 构建 DeepEval 测试用例
                # deepeval 4.x 的 LLMTestCase 为 pydantic 模型，不支持动态属性，
                # 自定义字段（期望行为）通过 additional_metadata 传递
                deepeval_tc = LLMTestCase(
                    input=tc.question,
                    actual_output=actual_answer,
                    expected_output=tc.expected_answer or tc.expected_behavior,
                    additional_metadata={"expected_behavior": tc.expected_behavior or ""},
                )
                deepeval_test_cases.append(deepeval_tc)

                result.per_case_results.append({
                    "id": tc.id,
                    "agent": target_agent_id,
                    "expected_tools": tc.expected_tools,
                    "actual_tools": actual_tools,
                    "tool_accuracy": tool_accuracy,
                })

            except Exception as e:
                logger.error(f"Agent 评估用例 {tc.id} 失败: {e}")
                result.per_case_results.append({"id": tc.id, "error": str(e)})

        # 运行 DeepEval 指标
        if deepeval_test_cases:
            quality_metric = MedicalResponseQualityMetric(threshold=EVAL_CONFIG["agent"]["min_response_quality"])
            completion_metric = TaskCompletionMetric(threshold=EVAL_CONFIG["agent"]["min_task_completion"])
            correctness_metric = MedicalAnswerCorrectnessMetric(threshold=EVAL_CONFIG["agent"]["min_answer_correctness"])

            # 收集分数容器：指标显示名 -> 每用例分数列表
            correctness_scores = []
            quality_scores = []
            completion_scores = []

            try:
                # 同步模式运行（避免 DashScope 限流）并关闭交互面板；
                # deepeval 4.x 的分数需从返回值 test_results[*].metrics_data 读取，
                # 共享 metric 对象的 .score 只保留最后一个用例的值
                from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
                eval_result = evaluate(
                    deepeval_test_cases,
                    [correctness_metric, quality_metric, completion_metric],
                    async_config=AsyncConfig(run_async=False),
                    display_config=DisplayConfig(
                        show_indicator=False, print_results=False,
                        verbose_mode=False, inspect_after_run=False,
                    ),
                )

                # 自定义指标 __name__ -> 分数列表 的映射
                name_map = {
                    "MedicalAnswerCorrectness": correctness_scores,
                    "MedicalResponseQuality": quality_scores,
                    "TaskCompletion": completion_scores,
                }
                for tr in eval_result.test_results:
                    for md in tr.metrics_data:
                        target = name_map.get(md.name)
                        if target is not None and md.score is not None and md.success:
                            target.append(md.score)
            except Exception as e:
                logger.error(f"DeepEval Agent 评估失败: {e}")

            result.avg_answer_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
            result.avg_response_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            result.avg_task_completion = sum(completion_scores) / len(completion_scores) if completion_scores else 0.0

        # 工具准确率
        result.avg_tool_accuracy = sum(tool_accuracy_scores) / len(tool_accuracy_scores) if tool_accuracy_scores else 0.0

        # 判断是否通过
        agent_config = EVAL_CONFIG["agent"]
        result.passed = (
            result.avg_tool_accuracy >= agent_config["min_tool_accuracy"]
            and result.avg_answer_correctness >= agent_config["min_answer_correctness"]
            and result.avg_task_completion >= agent_config["min_task_completion"]
            and result.avg_response_quality >= agent_config["min_response_quality"]
        )

        logger.info(f"Agent 评估完成: {'PASS' if result.passed else 'FAIL'}")
        return result
