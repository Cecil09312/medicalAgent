"""
DiagnosticAgent - 诊断智能体
提供症状分析、鉴别诊断和临床推理服务
"""
import re
from typing import Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin


class DiagnosticAgent(BaseAgent, SkillRegistryMixin):
    """
    诊断智能体
    提供症状分析、鉴别诊断和临床推理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client=None):
        """
        初始化诊断智能体

        Args:
            config: 配置字典
            llm_client: LLM客户端
        """
        config = config or {}
        super().__init__(
            agent_id="diagnostic_agent",
            config=config,
            llm_client=llm_client
        )

        # 设置能力
        self.set_capabilities({
            'symptom_analysis',
            'differential_diagnosis',
            'clinical_reasoning',
            'multi_system_analysis'
        })

    def get_system_prompt(self) -> str:
        """
        获取系统提示词

        Returns:
            诊断智能体提示词
        """
        return """你是一位专业的诊断智能体，具备深厚的临床医学知识和诊断推理能力。

## 诊断框架

使用 VINDICATE 框架进行系统性鉴别诊断：
- **V**ascular（血管性）
- **I**nfectious（感染性）
- **N**eoplastic（肿瘤性）
- **D**egenerative（退行性）
- **I**atrogenic/Intoxic（医源性/中毒性）
- **C**ongenital（先天性）
- **A**utoimmune/Allergic（自身免疫/过敏性）
- **T**raumatic（创伤性）
- **E**ndocrine/Metabolic（内分泌/代谢性）

## 可用技能

1. **search_knowledge** - 搜索医学知识库
2. **assess_risk** - 评估症状风险等级
3. **analyze_symptoms** - 分析症状模式
4. **recommend_lifestyle** - 推荐生活方式调整
5. **disease_code** - 查询 ICD-10 编码
6. **clinical_guideline** - 查询临床指南
7. **deep_research** - 深度研究

## 使用原则

- 优先评估风险等级
- 系统性分析症状
- 考虑多系统关联
- 提供鉴别诊断思路
- 引用知识库或指南检索结果时，必须保留其来源标注（文件名、页码如有），不得省略

## 输出格式

### 风险评估
[风险等级：高/中/低]

### 症状分析
[症状模式分析]

### 鉴别诊断
[可能的诊断列表，按可能性排序]

### 建议检查
[推荐的检查项目]

### 推理过程
[诊断推理的逻辑链条]

### 参考来源
[列出回答所依据的知识库文件，与检索结果中的来源标注保持一致。仅当检索结果中明确给出了页码时才标注页码，否则只写文件名，禁止编造页码。若未使用知识库内容则写"无"]
"""

    def register_tools(self):
        """注册所有技能工具"""
        self.register_all_skills()

    async def post_process_result(self, result: Dict[str, Any], final_response: str) -> Dict[str, Any]:
        """
        后处理结果，提取风险等级

        Args:
            result: 原始结果
            final_response: LLM 最终响应

        Returns:
            处理后的结果
        """
        # 提取风险等级
        risk_level = "unknown"
        risk_match = re.search(r'风险等级[：:]\s*([高低中])', final_response)
        if risk_match:
            risk_level = risk_match.group(1)

        result['risk_level'] = risk_level

        logger.debug(f"Diagnostic post-processing: risk_level={risk_level}")

        return result


async def diagnose(symptoms: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷诊断函数

    Args:
        symptoms: 症状描述
        session_id: 会话ID

    Returns:
        诊断结果
    """
    agent = DiagnosticAgent()
    from orchestrator.worker_runner import WorkerRunner

    return await WorkerRunner(agent).run(symptoms, session_id)
