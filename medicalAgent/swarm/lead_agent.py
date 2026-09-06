"""
Lead Agent
负责任务分析和分解，协调 Worker Agents
"""
import json
import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

from core.llm_client import LLMClient
from .shared_context import SharedContext, SubTask, TaskStatus


class LeadAgent:
    """Lead Agent - 任务协调者"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化 Lead Agent
        
        Args:
            llm_client: LLM 客户端（可选，默认创建新实例）
        """
        self.llm_client = llm_client or LLMClient()

    def _get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        Returns:
            系统提示词
        """
        return """你是医疗 Agent Swarm 的 Lead Agent，负责分析和分解用户问题。

你有 3 个 Worker Agents 可以协调：
1. **consultation_agent** - 咨询 Agent
   - 职责：生活方式建议、健康知识查询、风险评估、历史记录查询
   - 适用：简单健康问题、饮食运动建议、生活习惯指导

2. **diagnostic_agent** - 诊断 Agent
   - 职责：症状分析、疾病风险评估、临床指南查询
   - 适用：症状评估、高风险情况判断、需要医学推理的场景

3. **research_agent** - 研究 Agent
   - 职责：深度研究、临床指南查询、文献综述
   - 适用：需要最新研究证据、临床指南、系统综述的问题

**任务分配策略（重要：优先单 Agent，分解越多回答越慢）：**
- 绝大多数日常健康问题（症状咨询、用药、生活方式建议等）→ 只分解为 1 个任务，交给 consultation_agent
- 仅当问题同时涉及多个相互独立的方面（如"诊断分析 + 最新研究进展"）时，才分解为 2 个任务
- 只有真正复杂的综合案例才分解为 3 个任务，最多不超过 3 个
- 注意：单任务由该 Agent 直接回答，无需额外综合，速度最快，请优先采用

**输出格式（JSON，字段尽量简短）：**
```json
{
  "analysis": "一句话问题分析",
  "decomposition": [
    {
      "type": "任务类型",
      "description": "任务描述",
      "assigned_agent": "agent_id",
      "dependencies": []
    }
  ],
  "coordination_strategy": "一句话协作策略说明"
}
```

请根据用户问题，分析并分解为子任务。记住：优先只分解为 1 个任务。
"""

    async def assess_and_decompose(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        分析并分解问题
        
        Args:
            question: 用户问题
            context: 上下文信息（可选）
        
        Returns:
            分解结果字典
        """
        logger.info(f"Lead Agent analyzing question: {question[:100]}...")

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": question}
        ]

        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            messages.insert(1, {
                "role": "system",
                "content": f"上下文信息：\n{context_str}"
            })

        try:
            response = await self.llm_client.chat(messages=messages, temperature=0.3)
            
            # 尝试解析 JSON
            decomposition = self._parse_json_response(response)
            
            if not decomposition:
                # 如果解析失败，创建默认分解
                logger.warning("Failed to parse decomposition, using default")
                decomposition = {
                    "analysis": "问题分析失败，使用默认策略",
                    "decomposition": [
                        {
                            "type": "consultation",
                            "description": question,
                            "assigned_agent": "consultation_agent",
                            "dependencies": []
                        }
                    ],
                    "coordination_strategy": "单 Agent 处理"
                }
            
            logger.info(f"Decomposed into {len(decomposition.get('decomposition', []))} subtasks")
            return decomposition

        except Exception as e:
            logger.error(f"Failed to decompose question: {e}")
            return {
                "analysis": f"分析失败：{str(e)}",
                "decomposition": [],
                "coordination_strategy": "错误处理"
            }

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        解析 JSON 响应
        
        Args:
            response: LLM 响应文本
        
        Returns:
            解析后的字典，或 None
        """
        try:
            # 尝试提取 JSON 块
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.debug(f"JSON parse failed: {e}")
            return None

    def create_subtasks(
        self,
        decomposition: Dict[str, Any],
        shared_context: SharedContext
    ) -> List[SubTask]:
        """
        从分解结果创建 SubTask 对象
        
        Args:
            decomposition: 分解结果
            shared_context: 共享上下文
        
        Returns:
            创建的 SubTask 列表
        """
        subtasks = []
        decomposition_list = decomposition.get("decomposition", [])
        
        for i, task_def in enumerate(decomposition_list):
            task_id = f"task_{i}_{hash(task_def.get('description', '')) % 10000}"
            
            subtask = SubTask(
                id=task_id,
                type=task_def.get("type", "unknown"),
                description=task_def.get("description", ""),
                assigned_agent=task_def.get("assigned_agent", "consultation_agent"),
                dependencies=task_def.get("dependencies", [])
            )
            
            shared_context.add_subtask(subtask)
            subtasks.append(subtask)
        
        shared_context.task_decomposition = decomposition
        logger.info(f"Created {len(subtasks)} subtasks")
        
        return subtasks

    async def wait_for_completion(
        self,
        shared_context: SharedContext,
        timeout: int = 90
    ) -> bool:
        """
        等待所有子任务完成
        
        Args:
            shared_context: 共享上下文
            timeout: 超时时间（秒）
        
        Returns:
            是否成功完成（False 表示超时）
        """
        start_time = asyncio.get_event_loop().time()
        
        while not shared_context.is_all_subtasks_completed():
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.warning(f"Timeout after {timeout}s, some tasks incomplete")
                return False
            
            await asyncio.sleep(1)
        
        logger.info("All subtasks completed")
        return True

    async def synthesize_results(
        self,
        question: str,
        shared_context: SharedContext,
        timeout_occurred: bool = False
    ) -> str:
        """
        综合所有 Agent 的贡献生成最终答案
        
        Args:
            question: 原始问题
            shared_context: 共享上下文
            timeout_occurred: 是否发生超时
        
        Returns:
            综合后的最终答案
        """
        logger.info("Synthesizing results from all agents")

        completed_tasks = shared_context.get_all_completed_subtasks()
        
        contributions_text = ""
        for task in completed_tasks:
            if task.result:
                agent_answer = task.result.get("answer", "")
                contributions_text += f"\n\n**{task.assigned_agent}** ({task.type}):\n{agent_answer}"

        synthesis_prompt = f"""请综合以下多个 Agent 的贡献，为用户问题提供完整、连贯的答案。

**用户问题：** {question}

**各 Agent 贡献：**
{contributions_text}

**要求：**
1. 整合所有信息，避免重复
2. 保持逻辑连贯
3. 如有冲突，优先采信诊断和研究 Agent 的结论
4. 添加必要的免责声明
5. 使用通俗易懂的语言
6. 保留各 Agent 回答中的来源标注，在答案末尾统一汇总为"参考来源"节；页码以检索结果中给出的为准，未给出页码时只写文件名，禁止编造页码
"""

        if timeout_occurred:
            synthesis_prompt += "\n7. 注意：部分任务可能未完成，请基于已有信息提供答案，并说明信息可能不完整"

        messages = [
            {"role": "system", "content": "你是医疗 Swarm 系统的综合助手，负责整合多个 Agent 的输出。"},
            {"role": "user", "content": synthesis_prompt}
        ]

        try:
            # 限制综合答案的最大长度，避免过长生成拖慢响应
            final_answer = await self.llm_client.chat(messages=messages, temperature=0.5, max_tokens=2048)
            logger.info(f"Synthesis completed, answer length: {len(final_answer)}")
            return final_answer
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"综合结果失败：{str(e)}\n\n各 Agent 原始贡献：{contributions_text}"
