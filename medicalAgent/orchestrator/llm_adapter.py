"""
LLM 适配器
将 core/llm_client.LLMClient 包装为 langchain BaseChatModel,
供 LangGraph 的 react agent 使用。

所有模型调用仍经 LLMClient.chat_with_tools_retry 下发,
因此大小模型路由(contextvar)、熔断器、3 次指数退避重试对 LangGraph 透明生效。

工具 schema 直接透传 OpenAI dict 格式(由 orchestrator/tools.py 在
StructuredTool 上附加 _medical_openai_tool 属性提供),不经过 langchain
的格式转换,保证工具 schema 与底层 LLM 客户端的请求体一致,规避供应商兼容性差异。
"""
import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from loguru import logger

from core.llm_client import LLMClient


def _content_to_str(content: Any) -> str:
    """langchain 消息 content 兼容转换:str 原样;list 取各 text part 拼接"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def convert_messages_to_openai(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """
    langchain 消息列表 -> OpenAI chat.completions 消息格式

    覆盖 react agent 循环中出现的全部消息类型:
    SystemMessage / HumanMessage / AIMessage(含 tool_calls) / ToolMessage
    """
    openai_messages: List[Dict[str, Any]] = []

    for message in messages:
        kind = message.__class__.__name__
        if kind == "SystemMessage":
            openai_messages.append({"role": "system", "content": _content_to_str(message.content)})
        elif kind == "HumanMessage":
            openai_messages.append({"role": "user", "content": _content_to_str(message.content)})
        elif kind == "AIMessage":
            entry: Dict[str, Any] = {
                "role": "assistant",
                "content": _content_to_str(message.content) or None,
            }
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            openai_messages.append(entry)
        elif kind == "ToolMessage":
            openai_messages.append({
                "role": "tool",
                "tool_call_id": getattr(message, "tool_call_id", ""),
                "content": _content_to_str(message.content),
            })
        else:
            openai_messages.append({
                "role": "user",
                "content": _content_to_str(message.content),
            })

    return openai_messages


class LLMChatModel(BaseChatModel):
    """LLMClient 的 langchain 适配器(异步路径)"""

    llm_client: Any
    temperature: float = 0.7
    max_tokens: int = 2000

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "medical_llm_client"

    def bind_tools(
        self,
        tools: Sequence[Dict[str, Any] | Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ):
        formatted: List[Dict[str, Any]] = []
        for tool in tools:
            custom = getattr(tool, "_medical_openai_tool", None)
            if custom:
                formatted.append(custom)
                continue
            from langchain_core.utils.function_calling import convert_to_openai_tool

            formatted.append(convert_to_openai_tool(tool))

        if tool_choice and tool_choice not in ("auto", "none"):
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("LLMChatModel 仅支持异步调用(_agenerate)")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        openai_messages = convert_messages_to_openai(messages)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", "auto")

        response = await self.llm_client.chat_with_tools_retry(
            messages=openai_messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        tool_calls = [
            {
                "name": tc.name,
                "args": tc.arguments,
                "id": tc.id,
                "type": "tool_call",
            }
            for tc in response.tool_calls
        ]
        ai_message = AIMessage(
            content=response.content or "",
            tool_calls=tool_calls,
        )

        generation = ChatGeneration(message=ai_message)
        result = ChatResult(generations=[generation])
        result.llm_output = {"finish_reason": response.finish_reason}
        if tool_calls:
            logger.debug(f"LLMChatModel: {len(tool_calls)} tool calls requested")
        return result
