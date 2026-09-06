"""
Agent循环引擎
实现 LLM 驱动的 Skill 调用循环
支持短期记忆集成
支持约束验证（Harness Engineering）
"""
import uuid
import json
from typing import Dict, Any, List, Optional
from loguru import logger

from .state_manager import StateManager, TaskStatus
from .llm_client import LLMResponse

# Harness Engineering: 约束验证和自动修复
try:
    from constraints import ConstraintValidator
    from validation import AutoFixer
    CONSTRAINTS_ENABLED = True
except ImportError:
    logger.warning("Constraints module not found, running without constraint validation")
    CONSTRAINTS_ENABLED = False

# 专家审核集成
try:
    from review.review_trigger import ReviewTrigger
    from review.review_queue import ReviewQueue, get_default_queue
    REVIEW_ENABLED = True
except ImportError:
    REVIEW_ENABLED = False


class AgentLoop:
    """
    Agent循环引擎
    LLM 自主决策 Skill 调用，循环直到任务完成
    """

    def __init__(self, max_iterations: int = 10, short_term_memory: Optional[Any] = None, max_tool_calls: int = 2):
        """
        初始化Agent循环引擎

        Args:
            max_iterations: 最大迭代次数
            short_term_memory: 短期记忆管理器（可选）
            max_tool_calls: 最大 Skill 调用次数（硬性限制）
        """
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.state_manager = StateManager()
        self.short_term_memory = short_term_memory
        self.tool_call_count = 0

        # Harness Engineering: 约束验证器和自动修复器
        self.validator = ConstraintValidator() if CONSTRAINTS_ENABLED else None
        self.auto_fixer = AutoFixer() if CONSTRAINTS_ENABLED else None
        if CONSTRAINTS_ENABLED:
            logger.debug("Constraint validation enabled")

    async def run(self, agent, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行Agent循环

        Args:
            agent: Agent实例
            input_data: 输入数据
            session_id: 会话ID

        Returns:
            最终结果
        """
        task_id = str(uuid.uuid4())
        state = self.state_manager.create_state(
            task_id=task_id,
            agent_id=agent.agent_id,
            input_data=input_data,
            max_iterations=self.max_iterations
        )

        self.tool_call_count = 0

        logger.info(f"Starting Agent Loop for {agent.agent_id}, task_id={task_id}")

        try:
            state.status = TaskStatus.IN_PROGRESS

            # 初始化消息历史（包含历史对话）
            messages = self._initialize_messages(agent, input_data, session_id)

            # 记录用户消息到短期记忆
            if self.short_term_memory and session_id:
                user_message = messages[-1]["content"] if messages else str(input_data)
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="user",
                    content=user_message
                )

            # 获取 Agent 的 Skills (OpenAI format)
            tools_openai_format = agent.get_tools_for_llm()

            logger.debug(f"Agent has {len(tools_openai_format) if tools_openai_format else 0} skills available")

            # 连续 LLM 调用失败计数器：用于在 LLM 服务完全不可用时提前进入兜底路径
            consecutive_llm_failures = 0

            # 主循环：LLM -> Skill Calls -> Results -> LLM
            while state.should_continue():
                state.iteration += 1
                logger.debug(f"=== Iteration {state.iteration}/{state.max_iterations} ===")

                try:
                    # 调用 LLM（max_tokens 默认 2000，与输出长度约束一致，避免超长生成拖慢响应）
                    try:
                        llm_response: LLMResponse = await agent.llm_client.chat_with_tools_retry(
                            messages=messages,
                            tools=tools_openai_format,
                            tool_choice="auto",
                            temperature=agent.config.get('temperature', 0.7),
                            max_tokens=agent.config.get('max_tokens', 2000)
                        )
                        # LLM 调用成功，连续失败计数器归零
                        consecutive_llm_failures = 0
                    except Exception as llm_error:
                        # LLM 调用失败（内部已重试 3 次仍失败）：累计连续失败轮数
                        consecutive_llm_failures += 1
                        logger.error(f"LLM call failed in iteration {state.iteration}: {llm_error}")
                        if consecutive_llm_failures >= 2:
                            # 连续 2 轮 LLM 调用失败，判定 LLM 服务不可用，提前跳出主循环
                            # break 后进入下方 "if not state.is_completed():" 的兜底答案生成路径
                            logger.warning("连续 2 轮 LLM 调用失败，提前进入兜底答案生成")
                            break
                        # 单次瞬时失败：跳过本轮，进入下一轮重试（保持旧行为）
                        continue

                    # 记录中间结果
                    state.add_intermediate_result({
                        'iteration': state.iteration,
                        'llm_response': {
                            'content': llm_response.content,
                            'tool_calls': [
                                {'name': tc.name, 'arguments': tc.arguments}
                                for tc in llm_response.tool_calls
                            ],
                            'finish_reason': llm_response.finish_reason
                        }
                    })

                    # 情况1: LLM 返回 tool_calls
                    if llm_response.has_tool_calls():
                        if self.tool_call_count >= self.max_tool_calls:
                            logger.warning(f"Max Skill calls reached ({self.max_tool_calls}), forcing final answer")
                            messages.append({
                                'role': 'user',
                                'content': f'已完成 {self.max_tool_calls} 次信息检索。请基于已获取的信息提供最终答复。'
                            })
                            continue

                        logger.info(f"LLM requested {len(llm_response.tool_calls)} tool calls (total: {self.tool_call_count}/{self.max_tool_calls})")

                        messages.append(self._create_assistant_message_with_tools(llm_response))

                        if self.short_term_memory and session_id:
                            tool_names = [tc.name for tc in llm_response.tool_calls]
                            self.short_term_memory.add_message(
                                session_id=session_id,
                                role="assistant",
                                content=f"调用工具：{', '.join(tool_names)}"
                            )

                        for tool_call in llm_response.tool_calls:
                            self.tool_call_count += 1
                            logger.debug(f"Executing: {tool_call.name}({tool_call.arguments}) - call #{self.tool_call_count}")

                            # Harness Engineering: 验证调用
                            if self.validator:
                                validation_result = self.validator.validate_tool_call(
                                    agent.agent_id,
                                    tool_call.name
                                )
                                if not validation_result.get("valid"):
                                    logger.warning(f"Constraint warning: {validation_result.get('reason')}")

                            tool_result = await agent.execute_tool(
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments
                            )

                            messages.append(
                                agent.llm_client.create_tool_message(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                    result=tool_result
                                )
                            )

                            if self.short_term_memory and session_id:
                                result_summary = str(tool_result)[:200]
                                self.short_term_memory.add_message(
                                    session_id=session_id,
                                    role="tool",
                                    content=f"{tool_call.name}: {result_summary}"
                                )

                        continue

                    # 情况2: LLM 返回文本响应，任务完成
                    else:
                        logger.info("LLM provided final response (no tool calls)")

                        final_answer = llm_response.content

                        # Harness Engineering: 验证和修复输出
                        if self.validator and final_answer:
                            validation_result = self.validator.validate_output(
                                agent.agent_id,
                                final_answer
                            )

                            if not validation_result.get("valid"):
                                logger.warning(f"Output constraint violated: {validation_result.get('violations')}")

                                if self.auto_fixer and validation_result.get("auto_fixable"):
                                    fixed_answer = self.auto_fixer.fix_output(
                                        final_answer,
                                        validation_result.get("auto_fixable", [])
                                    )
                                    if fixed_answer != final_answer:
                                        logger.info("Output auto-fixed")
                                        final_answer = fixed_answer

                        if self.short_term_memory and session_id:
                            self.short_term_memory.add_message(
                                session_id=session_id,
                                role="assistant",
                                content=final_answer or "(empty response)"
                            )

                        result = {
                            'answer': final_answer,
                            'iterations': state.iteration,
                            'agent_id': agent.agent_id
                        }

                        if hasattr(agent, 'post_process_result'):
                            result = await agent.post_process_result(result, final_answer)

                        state.mark_completed(result)
                        break

                except Exception as e:
                    logger.error(f"Error in iteration {state.iteration}: {e}")
                    if state.iteration >= state.max_iterations:
                        state.mark_failed(str(e))
                        break

            # 达到最大迭代次数
            if not state.is_completed():
                logger.warning("Max iterations reached without completion")

                try:
                    logger.info("Forcing LLM to provide final answer")
                    messages.append({
                        'role': 'user',
                        'content': '请基于以上信息，提供最终的答复。'
                    })

                    final_response = await agent.llm_client.chat_with_tools_retry(
                        messages=messages,
                        tools=None,
                        temperature=0.7
                    )

                    result = {
                        'answer': final_response.content or '抱歉，未能完成任务',
                        'iterations': state.iteration,
                        'warning': 'max_iterations_reached'
                    }

                    if self.short_term_memory and session_id:
                        self.short_term_memory.add_message(
                            session_id=session_id,
                            role="assistant",
                            content=result['answer']
                        )

                    state.mark_completed(result)
                    logger.info("Generated fallback answer after max iterations")

                except Exception as e:
                    logger.error(f"Failed to generate fallback answer: {e}")
                    result = {
                        'answer': '抱歉，系统在处理您的问题时遇到了问题。建议您简化问题或稍后重试。',
                        'iterations': state.iteration,
                        'warning': 'max_iterations_reached',
                        'error': str(e)
                    }
                    state.mark_completed(result)

            logger.info(f"Agent Loop finished: status={state.status.value}, iterations={state.iteration}")

            # 专家审核集成
            if REVIEW_ENABLED and state.final_result:
                try:
                    trigger = ReviewTrigger()
                    result = state.final_result
                    forced_stop = result.get("warning") == "max_iterations_reached"

                    should_review, reason = trigger.should_review_realtime(
                        agent_output=result,
                        forced_stop=forced_stop,
                    )

                    if should_review:
                        # 使用进程级单例队列，保证专家审核页的操作能唤醒此处等待的协程
                        review_queue = get_default_queue()
                        question = input_data.get("question", str(input_data))
                        answer = result.get("answer", "")
                        review_result = await review_queue.submit_realtime(
                            question=question,
                            answer=answer,
                            trigger_reason=reason,
                        )
                        if review_result.action == "corrected" and review_result.correction:
                            result["answer"] = review_result.correction
                            result["expert_reviewed"] = True
                            logger.info(f"回答已被专家修正 [review_id={review_result.review_id}]")
                        elif review_result.action == "timeout":
                            result["expert_review_timeout"] = True
                            logger.warning("专家审核超时，使用原始回答")

                    # 异步抽样审核
                    should_async, async_reason = trigger.should_enqueue_async(result)
                    if should_async:
                        # 同样使用单例队列，保证审核页面能看到入队项
                        review_queue = get_default_queue()
                        question = input_data.get("question", str(input_data))
                        review_queue.enqueue_async(question, result.get("answer", ""), async_reason)
                except Exception as e:
                    logger.warning(f"专家审核集成失败: {e}")

            return state.final_result or {}

        except Exception as e:
            logger.error(f"Agent Loop failed: {e}")
            state.mark_failed(str(e))
            raise

    def _initialize_messages(self, agent, input_data: Dict[str, Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """初始化消息列表，包含历史对话上下文"""
        messages = []

        system_prompt = agent.get_system_prompt()
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # 加载历史对话（短期记忆）
        if self.short_term_memory and session_id:
            history = self.short_term_memory.get_history(session_id, limit=5)
            if history:
                logger.info(f"Loaded {len(history)} historical messages from short-term memory")
                messages.extend(history)

        user_message = agent.format_user_input(input_data)
        messages.append({
            'role': 'user',
            'content': user_message
        })

        return messages

    def _create_assistant_message_with_tools(self, llm_response: LLMResponse) -> Dict[str, Any]:
        """创建包含 tool_calls 的 assistant 消息"""
        message = {
            'role': 'assistant',
            'content': llm_response.content or None
        }

        if llm_response.tool_calls:
            message['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.name,
                        'arguments': json.dumps(tc.arguments, ensure_ascii=False)
                    }
                }
                for tc in llm_response.tool_calls
            ]

        return message
