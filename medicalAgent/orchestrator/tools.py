"""
工具适配器
将 BaseAgent.skill_registry 中注册的技能转换为 langchain StructuredTool,
供 LangGraph create_react_agent 使用。

同时实现 ToolCallBudget:
- max_tool_calls 硬限制:超限后不再执行技能,
  返回引导性提示让模型基于已有信息作答
- 记录本次运行实际使用的技能名单(skills_used,供飞轮维度统计)
"""
import asyncio
import inspect
import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import Field, create_model

from core.skill_registry import SkillParameter

_JSON_TYPES = {
    "string": (str, ...),
    "integer": (int, ...),
    "number": (float, ...),
    "boolean": (bool, ...),
    "object": (dict, ...),
    "array": (list, ...),
}


def _param_defaults(function) -> Dict[str, Any]:
    """从技能函数签名提取可选参数的默认值"""
    defaults: Dict[str, Any] = {}
    try:
        sig = inspect.signature(function)
    except (TypeError, ValueError):
        return defaults
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    return defaults


def _build_args_schema(parameters: List[SkillParameter], function) -> type:
    """SkillParameter 列表 -> Pydantic 模型(无参数技能返回空模型)"""
    fields: Dict[str, Any] = {}
    defaults = _param_defaults(function)
    for param in parameters:
        py_type = _JSON_TYPES.get(param.type, _JSON_TYPES["string"])[0]
        if param.required:
            fields[param.name] = (py_type, Field(description=param.description))
        else:
            default = defaults.get(param.name, None)
            fields[param.name] = (
                Optional[py_type],
                Field(default=default, description=param.description),
            )

    return create_model("SkillArgs", **fields)


def _single_skill_openai_format(name: str, skill: Dict[str, Any]) -> Dict[str, Any]:
    """生成与 SkillRegistry.to_openai_format 完全一致的单技能 schema"""
    properties: Dict[str, Any] = {}
    required = []
    for param in skill["parameters"]:
        prop = {"type": param.type, "description": param.description}
        if param.enum:
            prop["enum"] = param.enum
        properties[param.name] = prop
        if param.required:
            required.append(param.name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": skill["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class ToolCallBudget:
    """
    单次 Worker 运行的工具调用预算

    记录调用次数与去重后的技能名单;超过 max_tool_calls 后拒绝执行,
    返回引导模型基于已有信息作答的提示文本。
    """

    def __init__(self, agent, max_tool_calls: int = 2):
        self.agent = agent
        self.max_tool_calls = max_tool_calls
        self.reset()

    def reset(self):
        self.call_count = 0
        self.used_tools: List[str] = []

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if self.call_count >= self.max_tool_calls:
            logger.warning(
                f"Max Skill calls reached ({self.max_tool_calls}), forcing final answer"
            )
            return (
                f"已达工具调用上限({self.max_tool_calls} 次)。"
                "请基于已获取的信息提供最终答复,不要再次调用工具。"
            )

        self.call_count += 1
        if tool_name not in self.used_tools:
            self.used_tools.append(tool_name)

        # Harness Engineering: 工具调用约束校验(仅告警,不阻断)
        validator = getattr(self, "validator", None)
        if validator is not None:
            try:
                validation = validator.validate_tool_call(self.agent.agent_id, tool_name)
                if not validation.get("valid"):
                    logger.warning(f"Constraint warning: {validation.get('reason')}")
            except Exception as e:
                logger.debug(f"validate_tool_call failed: {e}")

        try:
            result = await self.agent.execute_tool(tool_name=tool_name, arguments=arguments)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e), "skill": tool_name}, ensure_ascii=False)


def build_langchain_tools(agent, budget: ToolCallBudget) -> List[StructuredTool]:
    """
    把 agent.skill_registry 中的全部技能包装为 StructuredTool

    每个工具附加 _medical_openai_tool 属性(OpenAI dict 格式 schema),
    LLMChatModel.bind_tools 优先使用它,保证发给 LLM 的 schema 与注册表定义一致。
    """
    tools: List[StructuredTool] = []

    for name, skill in agent.skill_registry.get_all().items():
        args_schema = _build_args_schema(skill["parameters"], skill["function"])
        budget_ref = budget
        tool_name = name

        async def _arun(_budget=budget_ref, _name=tool_name, **kwargs) -> str:
            return await _budget.execute(_name, kwargs)

        def _run(_budget=budget_ref, _name=tool_name, **kwargs) -> str:
            return asyncio.run(_budget.execute(_name, kwargs))

        tool = StructuredTool(
            name=tool_name,
            description=skill["description"],
            func=_run,
            coroutine=_arun,
            args_schema=args_schema,
        )
        tool._medical_openai_tool = _single_skill_openai_format(tool_name, skill)
        tools.append(tool)

    return tools
