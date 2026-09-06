"""
SkillRegistryMixin - 技能注册混入类
提供自动发现和注册技能的能力
"""
import inspect
from typing import List, Dict, Any
from loguru import logger

from core.skill_registry import SkillParameter
from core.skill_loader import discover_skills


class SkillRegistryMixin:
    """
    技能注册混入类
    为 Agent 提供自动技能发现和注册能力
    """

    def register_all_skills(self):
        """
        自动发现并注册所有技能
        扫描 .claude/skills 目录下的所有技能
        """
        logger.info(f"Agent {self.agent_id}: discovering and registering all skills...")

        discovered_skills = discover_skills()

        for skill_info in discovered_skills:
            skill_name = skill_info['name']
            function_name = skill_info['function_name']
            metadata = skill_info['metadata']
            function = skill_info['function']

            # 推断参数
            parameters = self._infer_skill_parameters(skill_info)

            # 注册到 skill_registry
            self.skill_registry.register(
                name=function_name,
                function=function,
                description=metadata.get('description', f'Skill: {skill_name}'),
                parameters=parameters
            )

            logger.debug(f"Registered skill: {function_name} with {len(parameters)} parameters")

        logger.info(f"Agent {self.agent_id}: registered {len(discovered_skills)} skills in total")

    def _infer_skill_parameters(self, skill_info: Dict[str, Any]) -> List[SkillParameter]:
        """
        从技能函数签名推断参数列表

        Args:
            skill_info: 技能信息字典

        Returns:
            SkillParameter 列表
        """
        function = skill_info['function']
        metadata = skill_info.get('metadata', {})

        # 获取函数签名
        sig = inspect.signature(function)
        parameters = []

        # 获取参数描述（从 metadata 中）
        param_descriptions = metadata.get('parameters', {})

        for param_name, param in sig.parameters.items():
            # 跳过 self 参数
            if param_name == 'self':
                continue

            # 推断类型
            param_type = "string"  # 默认类型
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"

            # 获取描述
            description = param_descriptions.get(param_name, f"Parameter: {param_name}")

            # 判断是否必需
            required = param.default == inspect.Parameter.empty

            parameters.append(SkillParameter(
                name=param_name,
                type=param_type,
                description=description,
                required=required
            ))

        return parameters
