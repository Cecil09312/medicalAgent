"""
ConsultationAgent - 健康咨询智能体
提供专业的健康咨询、风险评估和症状分诊服务
"""
import re
from typing import Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin


class ConsultationAgent(BaseAgent, SkillRegistryMixin):
    """
    健康咨询智能体
    提供专业健康建议、风险评估和症状分诊
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None):
        """
        初始化咨询智能体

        Args:
            config: 配置字典
            llm_client: LLM客户端
        """
        config = config or {}
        super().__init__(
            agent_id="consultation_agent",
            config=config,
            llm_client=llm_client
        )

        # 设置能力
        self.set_capabilities({
            'general_health_advice',
            'risk_assessment',
            'symptom_triage'
        })

    def get_system_prompt(self) -> str:
        """
        获取系统提示词

        Returns:
            专业健康顾问提示词
        """
        return """你是一位专业的健康咨询顾问，具备全面的医学知识和丰富的咨询经验。

## 可用技能

你可以使用以下 9 个技能来获取信息：

1. **search_knowledge** - 搜索医学知识库，获取疾病、症状、治疗等基础知识
2. **assess_risk** - 评估症状风险等级（高/中/低）
3. **analyze_symptoms** - 分析症状模式，识别可能的疾病
4. **recommend_lifestyle** - 推荐生活方式调整建议
5. **disease_code** - 查询疾病 ICD-10 编码
6. **clinical_guideline** - 查询临床指南
7. **deep_research** - 深度研究特定医学问题

## 使用原则

- 每次咨询最多使用 2-3 个技能，避免过度调用
- 优先使用 assess_risk 评估风险等级
- 根据症状选择合适的技能组合
- 始终提供免责声明
- 引用知识库检索结果时，必须在回答中保留其来源标注（文件名、页码如有），不得省略

## 输出格式

请按以下格式回答：

### 回答
[详细回答用户问题]

### 核心建议
[1-3 条核心建议]

### 参考来源
[列出回答所依据的知识库文件，与检索结果中的来源标注保持一致。仅当检索结果中明确给出了页码时才标注页码，否则只写文件名，禁止编造页码。若未使用知识库内容则写"无"]

### 免责声明
以上信息仅供参考，不能替代专业医疗诊断。如有健康问题，请及时就医。
"""

    def register_tools(self):
        """注册所有技能工具"""
        self.register_all_skills()

    def format_user_input(self, input_data: Dict[str, Any]) -> str:
        """
        格式化用户输入

        Args:
            input_data: 输入数据

        Returns:
            格式化后的字符串
        """
        parts = []

        # 添加会话ID
        if input_data.get('session_id'):
            parts.append(f"[会话ID: {input_data['session_id']}]")

        # 添加上下文
        if input_data.get('context'):
            parts.append(f"[上下文: {input_data['context']}]")

        # 添加问题
        question = input_data.get('question') or input_data.get('query') or str(input_data)
        parts.append(question)

        return "\n".join(parts)

    async def post_process_result(self, result: Dict[str, Any], final_response: str) -> Dict[str, Any]:
        """
        后处理结果，提取建议和免责声明

        Args:
            result: 原始结果
            final_response: LLM 最终响应

        Returns:
            处理后的结果
        """
        # 提取核心建议
        suggestions = []
        suggestion_match = re.search(r'###\s*核心建议\s*\n(.*?)(?=###|\Z)', final_response, re.DOTALL)
        if suggestion_match:
            suggestion_text = suggestion_match.group(1).strip()
            # 提取列表项
            items = re.findall(r'[-*]\s*(.+)', suggestion_text)
            suggestions = [item.strip() for item in items]

        # 提取免责声明
        disclaimer = ""
        disclaimer_match = re.search(r'###\s*免责声明\s*\n(.*?)(?=###|\Z)', final_response, re.DOTALL)
        if disclaimer_match:
            disclaimer = disclaimer_match.group(1).strip()

        result['suggestions'] = suggestions
        result['disclaimer'] = disclaimer

        logger.debug(f"Consultation post-processing: {len(suggestions)} suggestions extracted")

        return result


async def consult(question: str, session_id: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷咨询函数

    Args:
        question: 用户问题
        session_id: 会话ID
        context: 上下文信息

    Returns:
        咨询结果
    """
    agent = ConsultationAgent()
    input_data = {
        'question': question,
        'session_id': session_id,
        'context': context
    }
    return await agent.process(input_data)
