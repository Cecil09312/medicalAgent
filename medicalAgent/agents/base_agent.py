"""
BaseAgent - 智能体基类
定义所有医疗Agent的基础接口和行为
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

from core.llm_client import LLMClient
from core.agent_loop import AgentLoop
from core.skill_registry import SkillRegistry


class BaseAgent(ABC):
    """
    智能体基类
    提供 LLM 驱动的工具调用循环能力
    """

    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        llm_client: Optional[LLMClient] = None
    ):
        """
        初始化智能体

        Args:
            agent_id: 智能体唯一标识
            config: 配置字典
            llm_client: LLM客户端（可选，默认创建）
        """
        self.agent_id = agent_id
        self.config = config
        self.llm_client = llm_client or LLMClient()

        # 初始化核心组件
        self.loop = AgentLoop(
            max_iterations=config.get('max_iterations', 10),
            max_tool_calls=config.get('max_tool_calls', 2)
        )
        self.skill_registry = SkillRegistry()

        # Swarm 相关属性
        self._capabilities = set()
        self._shared_context = None
        self._identity_manager = None

        # 注册工具
        self.register_tools()

        logger.info(f"Agent {agent_id} initialized with {len(self.skill_registry.skills)} skills")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        子类必须实现

        Returns:
            系统提示词字符串
        """
        pass

    @abstractmethod
    def register_tools(self):
        """
        注册工具/技能
        子类必须实现
        """
        pass

    def get_tools_for_llm(self):
        """
        获取 OpenAI 格式的工具定义

        Returns:
            OpenAI tools 格式的列表
        """
        return self.skill_registry.to_openai_format()

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        return await self.skill_registry.execute(tool_name, **arguments)

    def format_user_input(self, input_data: Dict[str, Any]) -> str:
        """
        格式化用户输入

        Args:
            input_data: 输入数据字典

        Returns:
            格式化后的字符串
        """
        # 默认实现：提取 question 或 query
        return input_data.get('question') or input_data.get('query') or str(input_data)

    async def post_process_result(self, result: Dict[str, Any], final_response: str) -> Dict[str, Any]:
        """
        后处理结果
        子类可重写以提取特定信息

        Args:
            result: 原始结果
            final_response: LLM 最终响应

        Returns:
            处理后的结果
        """
        # 默认不做处理
        return result

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入（主入口）

        Args:
            input_data: 输入数据

        Returns:
            处理结果
        """
        session_id = input_data.get('session_id')
        return await self.run_loop(input_data, session_id)

    async def run_loop(self, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        运行 Agent 循环

        Args:
            input_data: 输入数据
            session_id: 会话ID

        Returns:
            循环结果
        """
        return await self.loop.run(self, input_data, session_id)

    # ========== Swarm 方法 ==========

    def set_capabilities(self, capabilities: set):
        """
        设置智能体能力集

        Args:
            capabilities: 能力集合
        """
        self._capabilities = capabilities
        logger.debug(f"Agent {self.agent_id} capabilities set: {capabilities}")

    def get_capabilities(self) -> set:
        """
        获取智能体能力集

        Returns:
            能力集合
        """
        return self._capabilities

    def attach_shared_context(self, context: Any):
        """
        附加共享上下文

        Args:
            context: 共享上下文对象
        """
        self._shared_context = context
        logger.debug(f"Agent {self.agent_id} attached shared context")

    def attach_identity_manager(self, identity_manager: Any):
        """
        附加身份管理器

        Args:
            identity_manager: 身份管理器对象
        """
        self._identity_manager = identity_manager
        logger.debug(f"Agent {self.agent_id} attached identity manager")

    async def process_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理子任务（Swarm 模式）

        Args:
            subtask: 子任务数据

        Returns:
            子任务结果
        """
        logger.info(f"Agent {self.agent_id} processing subtask: {subtask.get('task_id', 'unknown')}")
        return await self.process(subtask)
