"""
ResearchAgent - 研究智能体
提供文献检索、证据综合和事实核查服务
"""
import re
from typing import Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin


class ResearchAgent(BaseAgent, SkillRegistryMixin):
    """
    研究智能体
    提供文献检索、证据综合和事实核查
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None):
        """
        初始化研究智能体

        Args:
            config: 配置字典
            llm_client: LLM客户端
        """
        config = config or {}
        super().__init__(
            agent_id="research_agent",
            config=config,
            llm_client=llm_client
        )

        # 设置能力
        self.set_capabilities({
            'literature_search',
            'evidence_synthesis',
            'fact_checking',
            'guideline_lookup',
            'deep_research',
            'latest_information'
        })

    def get_system_prompt(self) -> str:
        """
        获取系统提示词

        Returns:
            研究智能体提示词
        """
        return """你是一位专业的医学研究智能体，具备文献检索和证据综合能力。

## 可用技能

1. **search_knowledge** - 搜索医学知识库
2. **clinical_guideline** - 查询临床指南（优先使用）
3. **deep_research** - 深度研究（优先使用）
4. **disease_code** - 查询 ICD-10 编码
5. **analyze_symptoms** - 分析症状模式
6. **assess_risk** - 评估风险等级
7. **recommend_lifestyle** - 推荐生活方式

## 使用原则

- 优先使用 clinical_guideline 和 deep_research 技能
- 注重证据等级和来源可靠性
- 提供系统性的文献综述
- 引用具体来源

## 输出格式

### 文献检索结果
[检索到的文献和指南列表，每条保留其来源标注（文件名及页码，以检索结果中给出的为准）]

### 证据摘要
[关键证据的综合摘要]

### 综合评估
[基于证据的综合评估和结论]

### 参考来源
[汇总列出所有引用的知识库文件，与检索结果中的来源标注保持一致。仅当检索结果中明确给出了页码时才标注页码，否则只写文件名，禁止编造页码]
"""

    def register_tools(self):
        """注册所有技能工具"""
        self.register_all_skills()

    async def post_process_result(self, result: Dict[str, Any], final_response: str) -> Dict[str, Any]:
        """
        后处理结果，提取证据等级和文献数量

        Args:
            result: 原始结果
            final_response: LLM 最终响应

        Returns:
            处理后的结果
        """
        # 提取证据等级
        evidence_level = "unknown"
        evidence_match = re.search(r'证据等级[：:]\s*([一二三四五1-5])', final_response)
        if evidence_match:
            evidence_level = evidence_match.group(1)

        # 提取文献数量
        literature_count = 0
        count_match = re.search(r'共\s*(\d+)\s*篇', final_response)
        if count_match:
            literature_count = int(count_match.group(1))
        else:
            # 尝试其他模式
            count_match = re.search(r'(\d+)\s*篇文献', final_response)
            if count_match:
                literature_count = int(count_match.group(1))

        result['evidence_level'] = evidence_level
        result['literature_count'] = literature_count

        logger.debug(f"Research post-processing: evidence_level={evidence_level}, literature_count={literature_count}")

        return result


async def research(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷研究函数

    Args:
        query: 研究问题
        session_id: 会话ID

    Returns:
        研究结果
    """
    agent = ResearchAgent()
    input_data = {
        'question': query,
        'session_id': session_id
    }
    return await agent.process(input_data)
