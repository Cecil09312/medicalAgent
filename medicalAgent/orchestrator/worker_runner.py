"""
Worker Runner
用 LangGraph create_react_agent 承载单个 Worker Agent 的工具调用循环,
替代自研 core/agent_loop.py 的 Think-Act-Observe 循环。

承载的横切关注点:
- max_tool_calls 硬限制(orchestrator/tools.ToolCallBudget)
- Harness Engineering 约束校验与自动修复
- 短期记忆注入(历史对话)与回写
- 专家审核集成(实时修正 + 异步抽样)
- agent.post_process_result 后处理
"""
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from loguru import logger

from .llm_adapter import LLMChatModel, convert_messages_to_openai
from .tools import ToolCallBudget, build_langchain_tools

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
    from review.review_queue import get_default_queue
    REVIEW_ENABLED = True
except ImportError:
    REVIEW_ENABLED = False


class WorkerRunner:
    """
    单个 Worker Agent 的 react 执行器

    每次 run() 复用同一个编译后的 react 图(图无状态,运行期状态在
    budget/消息列表中,由 run() 开头重置)。
    """

    def __init__(
        self,
        agent,
        short_term_memory: Optional[Any] = None,
        max_tool_calls: int = 2,
        recursion_limit: int = 15,
    ):
        self.agent = agent
        self.short_term_memory = short_term_memory
        self.max_tool_calls = max_tool_calls
        self.recursion_limit = recursion_limit

        self.budget = ToolCallBudget(agent, max_tool_calls=max_tool_calls)
        if CONSTRAINTS_ENABLED:
            self.budget.validator = ConstraintValidator()
            self.auto_fixer: Optional[Any] = AutoFixer()
        else:
            self.auto_fixer = None

        self.model = LLMChatModel(
            llm_client=agent.llm_client,
            temperature=agent.config.get("temperature", 0.7),
            max_tokens=agent.config.get("max_tokens", 2000),
        )
        self.tools = build_langchain_tools(agent, self.budget)
        self.graph = create_react_agent(model=self.model, tools=self.tools)

        logger.info(
            f"WorkerRunner initialized for {agent.agent_id} "
            f"({len(self.tools)} skills, max_tool_calls={max_tool_calls})"
        )

    async def run(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行一次完整的工具调用循环

        Returns:
            结果字典: {answer, agent_id, skills_used, warning?}
        """
        self.budget.reset()

        messages = self._initialize_messages(question, session_id)
        self._record_memory(session_id=session_id, role="user", content=question)

        warning = None
        try:
            final_state = await self.graph.ainvoke(
                {"messages": messages},
                config={"recursion_limit": self.recursion_limit},
            )
            final_answer = self._extract_final_answer(final_state)
        except GraphRecursionError:
            logger.warning("Max iterations reached without completion")
            final_answer = await self._forced_final_answer(messages)
            warning = "max_iterations_reached"

        final_answer = self._validate_and_fix_output(final_answer or "")

        self._record_memory(
            session_id=session_id, role="assistant", content=final_answer or "(empty response)"
        )

        result: Dict[str, Any] = {
            "answer": final_answer,
            "agent_id": self.agent.agent_id,
            "skills_used": list(self.budget.used_tools),
        }
        if warning:
            result["warning"] = warning

        if hasattr(self.agent, "post_process_result"):
            result = await self.agent.post_process_result(result, final_answer)

        result = await self._integrate_review(result, question)

        return result

    # ========== 消息构建 ==========

    def _initialize_messages(self, question: str, session_id: Optional[str]) -> List:
        """系统提示词 + 短期记忆历史 + 本次问题"""
        messages: List = []

        system_prompt = self.agent.get_system_prompt()
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if self.short_term_memory and session_id:
            try:
                history = self.short_term_memory.get_history(session_id, limit=5)
            except Exception as e:
                logger.warning(f"Failed to load short-term memory: {e}")
                history = None
            if history:
                logger.info(f"Loaded {len(history)} historical messages from short-term memory")
                for entry in history:
                    role = entry.get("role")
                    content = entry.get("content") or ""
                    if not content:
                        continue
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                    # 历史 tool 摘要消息跳过,保持 react 消息结构合法

        messages.append(HumanMessage(content=question))
        return messages

    def _extract_final_answer(self, final_state: Dict[str, Any]) -> Optional[str]:
        """从 react 图最终状态提取答案:最后一个不含 tool_calls 的 AIMessage"""
        state_messages = (final_state or {}).get("messages") or []
        for message in reversed(state_messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                content = message.content
                if isinstance(content, list):
                    parts = [
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                    ]
                    return "".join(parts)
                return content if isinstance(content, str) else str(content)
        return None

    async def _forced_final_answer(self, messages: List) -> str:
        """
        达到迭代上限后的兜底:不带工具再问一次,要求基于已有信息作答
        (达到迭代上限后的兜底路径)
        """
        try:
            openai_messages = convert_messages_to_openai(messages)
            openai_messages.append({
                "role": "user",
                "content": "请基于以上信息,提供最终的答复。",
            })
            response = await self.agent.llm_client.chat(
                messages=openai_messages, temperature=0.7
            )
            return response or "抱歉,未能完成任务"
        except Exception as e:
            logger.error(f"Failed to generate fallback answer: {e}")
            return "抱歉,系统在处理您的问题时遇到了问题。建议您简化问题或稍后重试。"

    # ========== 横切关注点 ==========

    def _record_memory(self, session_id: Optional[str], role: str, content: str):
        if self.short_term_memory and session_id and content:
            try:
                self.short_term_memory.add_message(
                    session_id=session_id, role=role, content=content[:32000]
                )
            except Exception as e:
                logger.warning(f"Failed to save short-term memory: {e}")

    def _validate_and_fix_output(self, final_answer: str) -> str:
        """Harness Engineering: 输出约束校验与自动修复"""
        validator = getattr(self.budget, "validator", None)
        if validator is None or not final_answer:
            return final_answer

        try:
            validation = validator.validate_output(self.agent.agent_id, final_answer)
        except Exception as e:
            logger.debug(f"validate_output failed: {e}")
            return final_answer

        if validation.get("valid"):
            return final_answer

        logger.warning(f"Output constraint violated: {validation.get('violations')}")
        if self.auto_fixer and validation.get("auto_fixable"):
            try:
                fixed = self.auto_fixer.fix_output(
                    final_answer, validation.get("auto_fixable", [])
                )
                if fixed != final_answer:
                    logger.info("Output auto-fixed")
                    return fixed
            except Exception as e:
                logger.debug(f"fix_output failed: {e}")
        return final_answer

    async def _integrate_review(self, result: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        专家审核集成:
        - 实时审核:高风险/强制停止时阻塞等待专家修正
        - 异步抽样:按触发规则入队,不阻塞
        """
        if not (REVIEW_ENABLED and result):
            return result

        try:
            trigger = ReviewTrigger()
            forced_stop = result.get("warning") == "max_iterations_reached"

            should_review, reason = trigger.should_review_realtime(
                agent_output=result,
                forced_stop=forced_stop,
                question=question,
            )

            if should_review:
                review_queue = get_default_queue()
                review_result = await review_queue.submit_realtime(
                    question=question,
                    answer=result.get("answer", ""),
                    trigger_reason=reason,
                    dimensions=result.get("review_dimensions"),
                )
                if review_result.action == "corrected" and review_result.correction:
                    result["answer"] = review_result.correction
                    result["expert_reviewed"] = True
                    logger.info(f"回答已被专家修正 [review_id={review_result.review_id}]")
                elif review_result.action == "timeout":
                    result["expert_review_timeout"] = True
                    logger.warning("专家审核超时,使用原始回答")

            should_async, async_reason = trigger.should_enqueue_async(result)
            if should_async:
                review_queue = get_default_queue()
                review_queue.enqueue_async(
                    question,
                    result.get("answer", ""),
                    async_reason,
                    dimensions=result.get("review_dimensions"),
                )
        except Exception as e:
            logger.warning(f"专家审核集成失败: {e}")

        return result
