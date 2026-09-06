"""
LLM客户端
支持调用 OpenAI 兼容的 API（如字节跳动豆包、OpenAI、Deepseek 等）
支持 function calling
"""
import sys
import os
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from loguru import logger

# 加载 .env
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass

# 熔断器与模型路由(需在 sys.path 修正之后导入,保证 core 包可被定位)
from core.circuit_breaker import get_circuit_breaker
from core.model_router import get_route


@dataclass
class ToolCall:
    """Function call 数据结构"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 响应数据结构（支持 function calling）"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: str  # "stop", "tool_calls", "length", "content_filter"

    def has_tool_calls(self) -> bool:
        """是否包含 function calls"""
        return len(self.tool_calls) > 0


class LLMClient:
    """统一的LLM客户端，支持多种模型"""

    def __init__(self, model_type: str = "openai_compatible"):
        """
        初始化LLM客户端

        Args:
            model_type: 模型类型，默认 "openai_compatible"
        """
        self.model_type = model_type

        if model_type == "openai_compatible":
            self.client = AsyncOpenAI(
                api_key=os.getenv("LLM_API_KEY", ""),
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
            )
            self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
            self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
            self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))

            # 小模型配置:配置了 LLM_SMALL_MODEL_NAME 时启用小模型降级与路由
            self.small_model = (os.getenv("LLM_SMALL_MODEL_NAME") or "").strip()
            if self.small_model:
                self.client_small = AsyncOpenAI(
                    api_key=(os.getenv("LLM_SMALL_API_KEY") or "").strip() or os.getenv("LLM_API_KEY", ""),
                    base_url=(os.getenv("LLM_SMALL_BASE_URL") or "").strip() or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
                )
                logger.info(f"已启用小模型降级与路由，简单问题或大模型熔断时将使用小模型: {self.small_model}")
            else:
                self.client_small = None

            # 主模型熔断器(按主模型名进程级共享单例)
            self._breaker = get_circuit_breaker(self.model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def _pick_route(self):
        """选择本次 API 调用使用的客户端与模型。返回 (client, model_name, is_small, reason)"""
        # 1. 未配置小模型时,始终使用主模型
        if self.client_small is None:
            return self.client, self.model_name, False, "默认"
        # 2. 请求级路由:强制走小模型
        if get_route() == "small":
            logger.info(f"模型路由: 简单问题路由，使用模型 {self.small_model}")
            return self.client_small, self.small_model, True, "简单问题路由"
        # 3. 熔断状态:允许请求说明处于闭合或半开探测状态,使用大模型
        if await self._breaker.allow_request():
            reason = "半开探测" if self._breaker.state == "half_open" else "默认"
            if reason != "默认":
                logger.info(f"模型路由: {reason}，使用模型 {self.model_name}")
            return self.client, self.model_name, False, reason
        # 4. 熔断打开或半开探测位被占用,降级走小模型
        logger.info(f"模型路由: 熔断降级，使用模型 {self.small_model}")
        return self.client_small, self.small_model, True, "熔断降级"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        异步聊天接口

        Args:
            messages: 消息列表
            temperature: 温度参数（可选）
            max_tokens: 最大token数（可选）

        Returns:
            模型返回的文本
        """
        try:
            temperature = temperature or self.temperature
            max_tokens = max_tokens or self.max_tokens

            logger.debug(f"Calling LLM ({self.model_type}) with {len(messages)} messages")

            # 选择本次调用的客户端与模型(小模型路由与熔断降级)
            client, model_name, is_small, reason = await self._pick_route()

            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception:
                # 大模型调用失败:计入熔断器失败;小模型调用失败不影响熔断器
                if not is_small:
                    await self._breaker.record_failure()
                raise

            # 大模型调用成功:重置熔断器失败计数;小模型调用不参与熔断计数
            if not is_small:
                await self._breaker.record_success()

            content = response.choices[0].message.content
            logger.debug(f"LLM response length: {len(content)} chars")
            return content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        带重试的聊天接口

        Args:
            messages: 消息列表
            max_retries: 最大重试次数

        Returns:
            模型返回的文本
        """
        for attempt in range(max_retries):
            try:
                return await self.chat(messages, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                await asyncio.sleep(2 ** attempt)

    def create_message(self, role: str, content: str) -> Dict[str, str]:
        """创建消息对象"""
        return {"role": role, "content": content}

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        带工具支持的聊天接口

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI format）
            tool_choice: 工具选择策略
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLMResponse 对象
        """
        try:
            temperature = temperature or self.temperature
            max_tokens = max_tokens or self.max_tokens

            logger.debug(f"Calling LLM with {len(tools) if tools else 0} tools")

            # 选择本次调用的客户端与模型(小模型路由与熔断降级)
            client, model_name, is_small, reason = await self._pick_route()

            request_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            if tools:
                request_params["tools"] = tools
                if tool_choice != "auto":
                    request_params["tool_choice"] = tool_choice

            try:
                response = await client.chat.completions.create(**request_params)
            except Exception:
                # 大模型调用失败:计入熔断器失败;小模型调用失败不影响熔断器
                if not is_small:
                    await self._breaker.record_failure()
                raise

            # 大模型调用成功:重置熔断器失败计数;小模型调用不参与熔断计数
            if not is_small:
                await self._breaker.record_success()

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))
                logger.debug(f"LLM requested {len(tool_calls)} tool calls")

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )

        except Exception as e:
            logger.error(f"LLM call with tools failed: {e}")
            raise

    async def chat_with_tools_retry(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        带重试的工具调用聊天接口

        循环最多 3 次调用 chat_with_tools，成功直接返回；
        失败时若已是最后一次则抛出异常，否则记录警告并按指数退避等待后重试
        （与 chat_with_retry 的重试风格保持一致）

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI format）
            tool_choice: 工具选择策略
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLMResponse 对象
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                await asyncio.sleep(2 ** attempt)

    def create_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建工具执行结果消息"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False)
        }
