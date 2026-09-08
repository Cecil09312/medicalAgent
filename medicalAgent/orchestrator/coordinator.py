"""
LangGraphOrchestrator
基于 LangGraph 的多Agent编排门面:

    process(question, context, session_id) -> {answer, suggestions, swarm_summary, ...}

职责:
- 任务分解(LLM JSON 输出)与默认兜底
- 综合各 Worker 贡献生成最终答案
- 记忆检索/保存、Worker 派发与回退
- 简单问题路由(请求级 contextvar)、统一异常兜底
- 紧急硬约束改写与异步审核入队(postprocess)
"""
import asyncio
import json
from typing import Any, Dict, List, Optional

from loguru import logger

from core.llm_client import LLMClient
from core.model_router import classify_question, reset_route, set_route

from .graph import build_medical_graph
from .worker_runner import WorkerRunner

# 专家审核集成
try:
    from review.review_trigger import ReviewTrigger
    from review.review_queue import get_default_queue
    REVIEW_ENABLED = True
except ImportError:
    REVIEW_ENABLED = False


DECOMPOSE_SYSTEM_PROMPT = """你是医疗 Agent Swarm 的 Lead Agent，负责分析和分解用户问题。

你有 3 个 Worker Agents 可以协调：
1. **consultation_agent** - 咨询 Agent
   - 职责：生活方式建议、健康知识查询、风险评估、历史记录查询
   - 适用：简单健康问题、饮食运动建议、生活习惯指导

2. **diagnostic_agent** - 诊断 Agent
   - 职责：症状分析、疾病风险评估、临床指南查询
   - 适用：症状评估、高风险情况判断、需要医学推理的场景

3. **research_agent** - 研究 Agent
   - 职责：深度研究、临床指南查询、文献综述
   - 适用：需要最新研究证据、临床指南、系统综述的问题

**任务分配策略（重要：优先单 Agent，分解越多回答越慢）：**
- 绝大多数日常健康问题（症状咨询、用药、生活方式建议等）→ 只分解为 1 个任务，交给 consultation_agent
- 仅当问题同时涉及多个相互独立的方面（如"诊断分析 + 最新研究进展"）时，才分解为 2 个任务
- 只有真正复杂的综合案例才分解为 3 个任务，最多不超过 3 个
- 注意：单任务由该 Agent 直接回答，无需额外综合，速度最快，请优先采用

**输出格式（JSON，字段尽量简短）：**
```json
{
  "analysis": "一句话问题分析",
  "decomposition": [
    {
      "type": "任务类型",
      "description": "任务描述",
      "assigned_agent": "agent_id",
      "dependencies": []
    }
  ],
  "coordination_strategy": "一句话协作策略说明"
}
```

请根据用户问题，分析并分解为子任务。记住：优先只分解为 1 个任务。
"""

SYNTHESIS_SYSTEM_PROMPT = "你是医疗 Swarm 系统的综合助手，负责整合多个 Agent 的输出。"


class LangGraphOrchestrator:
    """
    LangGraph 编排器

    graph 节点委托本类实例方法执行实际逻辑;测试可
    monkeypatch _decompose/_execute_worker 等接缝。
    """

    def __init__(
        self,
        worker_agents: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
        short_term_memory: Optional[Any] = None,
        long_term_memory: Optional[Any] = None,
        worker_timeout: float = 75.0,
        graph_timeout: float = 120.0,
        max_tool_calls: int = 2,
    ):
        self.worker_agents = worker_agents or {}
        self.llm_client = llm_client or LLMClient()
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.worker_timeout = worker_timeout
        self.graph_timeout = graph_timeout

        self.runners: Dict[str, WorkerRunner] = {}
        for agent_id, agent in self.worker_agents.items():
            self.runners[agent_id] = WorkerRunner(
                agent,
                short_term_memory=short_term_memory,
                max_tool_calls=max_tool_calls,
            )

        self.graph = build_medical_graph(self)
        logger.info(
            f"LangGraphOrchestrator initialized with {len(self.runners)} workers"
        )

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户问题(主入口)
        """
        logger.info(f"Processing question: {question[:100]}...")

        # 简单问题路由:请求级 contextvar 机制,
        # graph 在同一 asyncio 任务树中执行,Send 并行分支会复制上下文
        route_token = set_route("small") if await classify_question(question) else None
        if route_token is not None:
            logger.info("简单问题路由: 本次请求使用小模型")

        try:
            try:
                initial_state = {
                    "question": question,
                    "session_id": session_id,
                    "context": context,
                    "worker_results": [],
                    "worker_timeout": False,
                }
                state = await asyncio.wait_for(
                    self.graph.ainvoke(initial_state, config={"recursion_limit": 60}),
                    timeout=self.graph_timeout,
                )
                result = state.get("result")
                if not result:
                    return {
                        "answer": "系统暂时无法处理您的问题，请稍后重试。",
                        "warning": "empty_result",
                    }
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Graph timeout after {self.graph_timeout}s")
                return {
                    "answer": "抱歉，本次协作处理未能获得有效结果，请稍后重试或换个问题试试。",
                    "suggestions": [],
                    "timeout_occurred": True,
                    "warning": "graph_timeout",
                }
            except Exception as e:
                # 主流程统一兜底:任一节点异常时返回结构化错误,不抛给上层
                logger.exception(f"process 主流程处理异常: {e}")
                return {
                    "answer": "系统暂时无法处理您的问题，请稍后重试。",
                    "error": str(e),
                    "warning": "system_error"
                }
        finally:
            if route_token is not None:
                reset_route(route_token)

    # ========== 图节点委托方法(测试接缝) ==========

    async def _retrieve_memories(self, question: str, session_id: Optional[str]) -> Optional[List[Dict]]:
        """检索短期/长期记忆,失败降级返回 None"""
        memories = []

        if self.short_term_memory and session_id:
            try:
                history = self.short_term_memory.get_history(session_id, limit=5)
                if history:
                    memories.extend(history)
            except Exception as e:
                logger.warning(f"Failed to retrieve short-term memory: {e}")

        if self.long_term_memory:
            try:
                long_mem = self.long_term_memory.search_similar_sessions(question)
                if long_mem:
                    memories.extend(long_mem)
            except Exception as e:
                logger.warning(f"Failed to retrieve long-term memory: {e}")

        return memories if memories else None

    async def _decompose(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """分析并分解问题(LLM 输出 JSON,解析失败回退单 Agent 默认分解)"""
        logger.info(f"Lead Agent analyzing question: {question[:100]}...")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            messages.insert(1, {
                "role": "system",
                "content": f"上下文信息：\n{context_str}"
            })

        try:
            response = await self.llm_client.chat(messages=messages, temperature=0.3)
            decomposition = _parse_json_response(response)

            if not decomposition:
                logger.warning("Failed to parse decomposition, using default")
                decomposition = {
                    "analysis": "问题分析失败，使用默认策略",
                    "decomposition": [
                        {
                            "type": "consultation",
                            "description": question,
                            "assigned_agent": "consultation_agent",
                            "dependencies": []
                        }
                    ],
                    "coordination_strategy": "单 Agent 处理"
                }

            logger.info(f"Decomposed into {len(decomposition.get('decomposition', []))} subtasks")
            return decomposition

        except Exception as e:
            logger.error(f"Failed to decompose question: {e}")
            return {
                "analysis": f"分析失败：{str(e)}",
                "decomposition": [],
                "coordination_strategy": "错误处理"
            }

    async def _execute_worker(
        self,
        agent_id: str,
        question_input: str,
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """执行单个 Worker(按 agent_id 取 runner,找不到回退 consultation_agent)"""
        runner = self.runners.get(agent_id)
        if runner is None:
            logger.warning(f"Agent {agent_id} not found, falling back to consultation_agent")
            agent_id = "consultation_agent"
            runner = self.runners.get(agent_id)

        if runner is None:
            logger.error("No agent available")
            raise RuntimeError("no_agent_available")

        return await asyncio.wait_for(
            runner.run(question_input, session_id),
            timeout=self.worker_timeout,
        )

    async def _synthesize(
        self,
        question: str,
        worker_results: List[Dict[str, Any]],
        timeout_occurred: bool = False
    ) -> str:
        """综合各 Agent 贡献生成最终答案(冲突时优先采信诊断/研究结论)"""
        logger.info("Synthesizing results from all agents")

        contributions_text = ""
        for contribution in worker_results:
            agent_answer = contribution.get("answer", "")
            contributions_text += (
                f"\n\n**{contribution.get('agent_id', 'unknown')}** "
                f"({contribution.get('task_type', 'unknown')}):\n{agent_answer}"
            )

        synthesis_prompt = f"""请综合以下多个 Agent 的贡献，为用户问题提供完整、连贯的答案。

**用户问题：** {question}

**各 Agent 贡献：**
{contributions_text}

**要求：**
1. 整合所有信息，避免重复
2. 保持逻辑连贯
3. 如有冲突，优先采信诊断和研究 Agent 的结论
4. 添加必要的免责声明
5. 使用通俗易懂的语言
6. 保留各 Agent 回答中的来源标注，在答案末尾统一汇总为"参考来源"节；页码以检索结果中给出的为准，未给出页码时只写文件名，禁止编造页码
"""

        if timeout_occurred:
            synthesis_prompt += "\n7. 注意：部分任务可能未完成，请基于已有信息提供答案，并说明信息可能不完整"

        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt}
        ]

        try:
            final_answer = await self.llm_client.chat(
                messages=messages, temperature=0.5, max_tokens=2048
            )
            logger.info(f"Synthesis completed, answer length: {len(final_answer)}")
            return final_answer
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"综合结果失败：{str(e)}\n\n各 Agent 原始贡献：{contributions_text}"

    async def _postprocess(self, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """维度信息 + 医疗安全硬约束 + 异步审核入队"""
        result = dict(result)

        # 附加审核维度信息(疾病类别/Skill/Agent,供飞轮指标下钻)
        try:
            from flywheel.dimensions import build_review_dimensions
            result["review_dimensions"] = build_review_dimensions(question, result)
        except Exception as e:
            logger.warning(f"维度信息构建失败: {e}")

        # 医疗安全硬约束:高危关键词 + 回答无就医引导时强制改写
        try:
            from constraints.emergency import (
                detect_emergency, emergency_response, has_urgent_care_guidance,
            )
            matched_keyword = detect_emergency(question)
            if matched_keyword and result.get("answer"):
                result["emergency_keyword"] = matched_keyword
                if not has_urgent_care_guidance(result["answer"]):
                    result["reference_answer"] = result["answer"]
                    result["answer"] = emergency_response(matched_keyword)
                    result["warning"] = "emergency_hard_constraint"
                    logger.warning(
                        f"医疗安全硬约束触发（关键词: {matched_keyword}），"
                        "回答已强制改写为紧急就医引导"
                    )
        except Exception as e:
            logger.warning(f"医疗安全硬约束执行失败: {e}")

        # 专家审核集成(异步抽样)
        if REVIEW_ENABLED and result.get("answer"):
            try:
                trigger = ReviewTrigger()
                should_async, async_reason = trigger.should_enqueue_async(result)
                if should_async:
                    review_queue = get_default_queue()
                    review_queue.enqueue_async(
                        question, result["answer"], async_reason,
                        dimensions=result.get("review_dimensions"),
                    )
            except Exception as e:
                logger.warning(f"Swarm 审核集成失败: {e}")

        return result

    async def _save_memories(
        self,
        question: str,
        answer: str,
        session_id: Optional[str],
        save_stm: bool = True
    ):
        """
        保存记忆

        save_stm=False 用于单 Agent 模式:WorkerRunner 已回写 assistant 消息,
        此处不再重复写短期记忆,只补长期记忆
        """
        if save_stm and self.short_term_memory and session_id:
            try:
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer[:500]
                )
            except Exception as e:
                logger.warning(f"Failed to save short-term memory: {e}")

        if self.long_term_memory:
            try:
                self.long_term_memory.add_session_summary(
                    summary=f"Q: {question}\nA: {answer[:500]}",
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to save long-term memory: {e}")


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回中的 JSON 块(兼容 ```json 围栏与裸 JSON)"""
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            json_str = response.strip()

        return json.loads(json_str)
    except Exception as e:
        logger.debug(f"JSON parse failed: {e}")
        return None
