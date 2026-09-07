"""
Swarm Coordinator
协调多个 Worker Agents 并行处理任务
"""
import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

from core.llm_client import LLMClient
from core.agent_loop import AgentLoop
from core.model_router import classify_question, reset_route, set_route
from .shared_context import SharedContext, SubTask, TaskStatus, Contribution
from .events import Event, EventType
from .lead_agent import LeadAgent

# 专家审核集成
try:
    from review.review_trigger import ReviewTrigger
    from review.review_queue import ReviewQueue, get_default_queue
    REVIEW_ENABLED = True
except ImportError:
    REVIEW_ENABLED = False


class SwarmCoordinator:
    """Swarm 协调器"""

    def __init__(
        self,
        worker_agents: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
        short_term_memory: Optional[Any] = None,
        long_term_memory: Optional[Any] = None
    ):
        """
        初始化 Swarm 协调器
        
        Args:
            worker_agents: Worker Agent 字典 {agent_id: agent_instance}
            llm_client: LLM 客户端
            short_term_memory: 短期记忆管理器
            long_term_memory: 长期记忆管理器
        """
        self.worker_agents = worker_agents or {}
        self.llm_client = llm_client or LLMClient()
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.lead_agent = LeadAgent(self.llm_client)
        
        # 为每个 Worker 创建 AgentLoop
        self.worker_loops: Dict[str, AgentLoop] = {}
        for agent_id, agent in self.worker_agents.items():
            self.worker_loops[agent_id] = AgentLoop(
                max_iterations=5,
                short_term_memory=short_term_memory
            )
        
        logger.info(f"SwarmCoordinator initialized with {len(self.worker_agents)} workers")

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户问题（主入口）
        
        Args:
            question: 用户问题
            context: 上下文信息
            session_id: 会话 ID
        
        Returns:
            处理结果字典
        """
        logger.info(f"Processing question: {question[:100]}...")

        # 简单问题路由判定：简单问题本次请求强制使用小模型，复杂问题保持默认路由；
        # 记录 token 以便请求结束后恢复默认路由。
        # classify_question 在启用二级 LLM 意图分类时对临界问题做一次小模型分类，
        # 未启用/分类失败时退化为纯启发式判断（行为与现状一致）
        route_token = set_route("small") if await classify_question(question) else None
        if route_token is not None:
            logger.info("简单问题路由: 本次请求使用小模型")

        try:
            # 主流程统一兜底：记忆检索、任务分解、模式路由、审核集成等任一环节异常时，
            # 记录完整异常并返回结构化错误结果，避免异常直接抛给上层调用方
            try:
                # 检索记忆
                memories = await self._retrieve_memories(question, session_id)
                if memories:
                    context = context or {}
                    context["memories"] = memories

                # Lead Agent 分析和分解
                decomposition = await self.lead_agent.assess_and_decompose(question, context)

                subtasks = decomposition.get("decomposition", [])

                # 判断是否需要 Swarm（多 Agent 协作）
                if len(subtasks) <= 1:
                    # 单 Agent 处理
                    logger.info("Single agent mode")
                    result = await self._process_single_agent(question, subtasks, session_id)
                else:
                    # Swarm 模式
                    logger.info(f"Swarm mode with {len(subtasks)} agents")
                    result = await self._process_with_swarm(question, decomposition, session_id)

                # 附加审核维度信息（疾病类别/Skill/Agent，供飞轮指标下钻）
                try:
                    from flywheel.dimensions import build_review_dimensions
                    result["review_dimensions"] = build_review_dimensions(question, result)
                except Exception as e:
                    logger.warning(f"维度信息构建失败: {e}")

                # 医疗安全硬约束：问题命中高危症状关键词时，
                # 若回答未包含紧急就医引导（且未获得专家审核修正），强制改写为标准就医话术；
                # 原回答保留在 reference_answer 字段（不直接展示给用户）。
                # 实时专家审核已在 AgentLoop 内按 high_risk 原因强制触发（见 review_trigger）
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

                # 专家审核集成（异步抽样）
                if REVIEW_ENABLED and result.get("answer"):
                    try:
                        trigger = ReviewTrigger()
                        should_async, async_reason = trigger.should_enqueue_async(result)
                        if should_async:
                            # 使用进程级单例队列：保证审核页面能实时看到此处入队的审核项
                            review_queue = get_default_queue()
                            review_queue.enqueue_async(
                                question, result["answer"], async_reason,
                                dimensions=result.get("review_dimensions"),
                            )
                    except Exception as e:
                        logger.warning(f"Swarm 审核集成失败: {e}")

                return result
            except Exception as e:
                # 记录完整异常堆栈，返回统一的系统错误兜底结果
                logger.exception(f"process 主流程处理异常: {e}")
                return {
                    "answer": "系统暂时无法处理您的问题，请稍后重试。",
                    "error": str(e),
                    "warning": "system_error"
                }
        finally:
            # 恢复默认路由，避免影响后续请求
            if route_token is not None:
                reset_route(route_token)

    async def _retrieve_memories(self, question: str, session_id: Optional[str]) -> Optional[List[Dict]]:
        """检索记忆"""
        memories = []
        
        # 短期记忆
        if self.short_term_memory and session_id:
            try:
                history = self.short_term_memory.get_history(session_id, limit=5)
                if history:
                    memories.extend(history)
            except Exception as e:
                logger.warning(f"Failed to retrieve short-term memory: {e}")
        
        # 长期记忆
        if self.long_term_memory:
            try:
                # LongTermMemory 的检索接口为 search_similar_sessions
                long_mem = self.long_term_memory.search_similar_sessions(question)
                if long_mem:
                    memories.extend(long_mem)
            except Exception as e:
                logger.warning(f"Failed to retrieve long-term memory: {e}")
        
        return memories if memories else None

    async def _process_single_agent(
        self,
        question: str,
        subtasks: List[Dict],
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """单 Agent 处理"""
        if not subtasks:
            # 默认使用 consultation_agent
            agent_id = "consultation_agent"
        else:
            agent_id = subtasks[0].get("assigned_agent", "consultation_agent")
        
        agent = self.worker_agents.get(agent_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found, falling back to consultation_agent")
            agent_id = "consultation_agent"
            agent = self.worker_agents.get(agent_id)
        
        if not agent:
            return {"answer": "没有可用的 Agent", "error": "no_agent_available"}
        
        agent_loop = self.worker_loops.get(agent_id)
        if not agent_loop:
            agent_loop = AgentLoop(short_term_memory=self.short_term_memory)
            self.worker_loops[agent_id] = agent_loop
        
        result = await agent_loop.run(agent, {"question": question}, session_id)
        
        # 保存记忆
        await self._save_memories(question, result.get("answer", ""), session_id)
        
        return result

    async def _process_with_swarm(
        self,
        question: str,
        decomposition: Dict[str, Any],
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Swarm 模式处理
        
        Args:
            question: 用户问题
            decomposition: 分解结果
            session_id: 会话 ID
        
        Returns:
            处理结果
        """
        shared_context = SharedContext()
        
        # 发布 Swarm 开始事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_STARTED,
            source_agent="lead_agent",
            data={"question": question}
        ))
        
        # 创建子任务
        subtasks = self.lead_agent.create_subtasks(decomposition, shared_context)
        
        # 发布任务分解事件
        shared_context.publish_event(Event(
            type=EventType.TASK_DECOMPOSED,
            source_agent="lead_agent",
            data={"subtasks": [t.id for t in subtasks]}
        ))
        
        # 并行执行 Worker Agents
        worker_tasks = []
        for subtask in subtasks:
            agent_id = subtask.assigned_agent
            if agent_id in self.worker_agents:
                worker_tasks.append(
                    self._worker_execute_assigned_tasks(
                        agent_id, subtask.id, shared_context, session_id
                    )
                )
            else:
                logger.warning(f"Worker agent {agent_id} not found")
        
        # 等待完成，带超时
        timeout = 90
        timeout_occurred = False
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*worker_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Swarm timeout after {timeout}s")
            timeout_occurred = True
        
        # 兜底检查：判断所有 Worker 是否均未产生任何有效贡献（含全部失败或超时导致全部中断的场景）
        # agent_contributions 为 Dict[agent_id, List[Contribution]]，只有 Worker 成功完成时才会追加贡献；
        # 所有贡献列表均为空（或字典整体为空）即视为全部 Worker 失败
        all_contributions_empty = all(
            not contributions
            for contributions in shared_context.agent_contributions.values()
        )
        if all_contributions_empty:
            # 全部 Worker 失败：记录错误日志，跳过 LLM 综合合成，直接返回兜底结果
            logger.error(
                f"Swarm 所有 Worker 均失败，未产生任何有效贡献（timeout_occurred={timeout_occurred}），"
                "跳过 LLM 综合合成，直接返回兜底结果"
            )
            return {
                "answer": "抱歉，本次协作处理未能获得有效结果，请稍后重试或换个问题试试。",
                "suggestions": [],
                "swarm_summary": shared_context.get_summary(),
                "timeout_occurred": timeout_occurred,
                "warning": "all_workers_failed"
            }

        # 综合结果
        final_answer = await self.lead_agent.synthesize_results(
            question, shared_context, timeout_occurred
        )
        
        # 发布 Swarm 完成事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_COMPLETED,
            source_agent="lead_agent",
            data={"final_answer_length": len(final_answer)}
        ))
        
        # 保存记忆
        await self._save_memories(question, final_answer, session_id)
        
        # 提取建议
        suggestions = self._extract_suggestions(final_answer)
        
        return {
            "answer": final_answer,
            "suggestions": suggestions,
            "swarm_summary": shared_context.get_summary(),
            "timeout_occurred": timeout_occurred
        }

    async def _worker_execute_assigned_tasks(
        self,
        agent_id: str,
        task_id: str,
        shared_context: SharedContext,
        session_id: Optional[str]
    ):
        """
        Worker 执行分配的任务
        
        Args:
            agent_id: Agent ID
            task_id: 任务 ID
            shared_context: 共享上下文
            session_id: 会话 ID
        """
        subtask = shared_context.get_subtask(task_id)
        if not subtask:
            logger.error(f"Subtask {task_id} not found")
            return
        
        agent = self.worker_agents.get(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found")
            return
        
        agent_loop = self.worker_loops.get(agent_id)
        if not agent_loop:
            agent_loop = AgentLoop(short_term_memory=self.short_term_memory)
            self.worker_loops[agent_id] = agent_loop
        
        # 标记任务开始
        shared_context.start_subtask(task_id)
        shared_context.publish_event(Event(
            type=EventType.SUBTASK_STARTED,
            source_agent=agent_id,
            data={"task_id": task_id}
        ))
        
        try:
            # 执行任务
            result = await agent_loop.run(
                agent,
                {"question": subtask.description},
                session_id
            )
            
            # 标记任务完成
            shared_context.complete_subtask(task_id, result)
            
            # 记录贡献
            contribution = Contribution(
                agent_id=agent_id,
                task_id=task_id,
                content=result
            )
            shared_context.agent_contributions[agent_id].append(contribution)
            
            shared_context.publish_event(Event(
                type=EventType.SUBTASK_COMPLETED,
                source_agent=agent_id,
                data={"task_id": task_id, "result_length": len(result.get("answer", ""))}
            ))
            
            logger.info(f"Worker {agent_id} completed task {task_id}")
            
        except Exception as e:
            logger.error(f"Worker {agent_id} failed task {task_id}: {e}")
            subtask.status = TaskStatus.FAILED
            subtask.result = {"error": str(e)}

    def _extract_suggestions(self, answer: str) -> List[str]:
        """
        从最终答案中提取建议
        
        Args:
            answer: 最终答案文本
        
        Returns:
            建议列表
        """
        suggestions = []
        
        # 简单的建议提取规则
        suggestion_patterns = [
            r"建议[：:]\s*(.+)",
            r"可以\s*(.+)",
            r"应该\s*(.+)",
            r"推荐\s*(.+)"
        ]
        
        import re
        for pattern in suggestion_patterns:
            matches = re.findall(pattern, answer)
            suggestions.extend(matches[:3])  # 每种模式最多 3 条
        
        return suggestions[:10]  # 总共最多 10 条

    async def _save_memories(self, question: str, answer: str, session_id: Optional[str]):
        """保存记忆"""
        # 保存短期记忆
        if self.short_term_memory and session_id:
            try:
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer[:500]  # 只保存前 500 字符
                )
            except Exception as e:
                logger.warning(f"Failed to save short-term memory: {e}")
        
        # 保存长期记忆
        if self.long_term_memory:
            try:
                # LongTermMemory 的写入接口为 add_session_summary
                self.long_term_memory.add_session_summary(
                    summary=f"Q: {question}\nA: {answer[:500]}",
                    session_id=session_id,
                )
            except Exception as e:
                logger.warning(f"Failed to save long-term memory: {e}")


async def process_with_swarm(
    question: str,
    worker_agents: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    short_term_memory: Optional[Any] = None,
    long_term_memory: Optional[Any] = None
) -> Dict[str, Any]:
    """
    便捷函数：使用 Swarm 处理问题

    Args:
        question: 用户问题
        worker_agents: Worker Agent 字典（可选，缺省时自动创建 3 个默认 Worker）
        context: 上下文信息
        session_id: 会话 ID
        llm_client: LLM 客户端
        short_term_memory: 短期记忆管理器
        long_term_memory: 长期记忆管理器

    Returns:
        处理结果
    """
    # 未传入 Worker 时创建默认组合（CLI 模式等场景直接使用）
    if llm_client is None:
        llm_client = LLMClient()
    if worker_agents is None:
        from agents.consultation_agent import ConsultationAgent
        from agents.diagnostic_agent import DiagnosticAgent
        from agents.research_agent import ResearchAgent
        worker_agents = {
            "consultation_agent": ConsultationAgent(llm_client=llm_client),
            "diagnostic_agent": DiagnosticAgent(llm_client=llm_client),
            "research_agent": ResearchAgent(llm_client=llm_client),
        }

    coordinator = SwarmCoordinator(
        worker_agents=worker_agents,
        llm_client=llm_client,
        short_term_memory=short_term_memory,
        long_term_memory=long_term_memory
    )
    
    return await coordinator.process(question, context, session_id)
