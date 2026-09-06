"""
评估集自动扩充
将专家审核通过的 Q&A 自动加入评估数据集
"""
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# 评估数据集路径
_EVAL_DATA_DIR = Path(__file__).parent.parent / "evaluation" / "data"


class EvalDatasetExpander:
    """评估集自动扩充器"""

    def __init__(self, eval_data_dir: Optional[Path] = None):
        self._data_dir = eval_data_dir or _EVAL_DATA_DIR

    def export_reviewed_cases(
        self,
        reviewed_items: List[Dict],
        min_confidence: int = 2,
    ) -> Dict[str, int]:
        """
        将已审核且通过的案例导出为评估用例

        Args:
            reviewed_items: 已审核项列表，每项包含:
                - question: 问题
                - expert_correction: 专家修正后的回答
                - trigger_reason: 触发原因
            min_confidence: 最少需要多少个审核案例才触发扩充

        Returns:
            各数据集新增案例数 {"agent": N, "rag": N, "swarm": N}
        """
        if len(reviewed_items) < min_confidence:
            logger.info(f"审核案例数 ({len(reviewed_items)}) 不足最小阈值 ({min_confidence})，跳过扩充")
            return {"agent": 0, "rag": 0, "swarm": 0}

        counts = {"agent": 0, "rag": 0, "swarm": 0}

        agent_cases = []
        rag_cases = []
        swarm_cases = []

        for item in reviewed_items:
            question = item.get("question", "")
            answer = item.get("expert_correction") or item.get("original_answer", "")
            if not question or not answer:
                continue

            # 分类到对应的评估数据集
            case_type = self._classify_case(question, item)

            if case_type == "swarm":
                swarm_cases.append(self._make_swarm_case(question, answer, item))
            elif case_type == "rag":
                rag_cases.append(self._make_rag_case(question, answer, item))
            else:
                agent_cases.append(self._make_agent_case(question, answer, item))

        # 追加到各数据集文件
        if agent_cases:
            self._append_to_json("agent_test_cases.json", agent_cases)
            counts["agent"] = len(agent_cases)

        if rag_cases:
            self._append_to_json("rag_test_cases.json", rag_cases)
            counts["rag"] = len(rag_cases)

        if swarm_cases:
            self._append_to_json("swarm_test_cases.json", swarm_cases)
            counts["swarm"] = len(swarm_cases)

        total = sum(counts.values())
        if total > 0:
            logger.info(f"评估集扩充完成: agent+{counts['agent']}, rag+{counts['rag']}, swarm+{counts['swarm']}")

        return counts

    def _classify_case(self, question: str, item: Dict) -> str:
        """根据问题内容分类"""
        # Swarm 特征：涉及多 Agent 协作
        swarm_keywords = ["指南", "研究", "进展", "最新", "同时", "并且", "还需要"]
        if any(kw in question for kw in swarm_keywords):
            return "swarm"

        # RAG 特征：知识检索类
        rag_keywords = ["什么是", "是什么", "有哪些", "区别", "编码", "ICD"]
        if any(kw in question for kw in rag_keywords):
            return "rag"

        return "agent"

    def _make_agent_case(self, question: str, answer: str, item: Dict) -> Dict:
        """构造 Agent 评估用例"""
        return {
            "id": f"flywheel_{uuid.uuid4().hex[:6]}",
            "question": question,
            "expected_tools": [],
            "expected_behavior": f"专家审核后的标准回答: {answer[:100]}...",
            "expected_answer": answer,
        }

    def _make_rag_case(self, question: str, answer: str, item: Dict) -> Dict:
        """构造 RAG 评估用例"""
        return {
            "id": f"flywheel_{uuid.uuid4().hex[:6]}",
            "question": question,
            "expected_answer": answer,
            "expected_contexts": [answer[:200]],
        }

    def _make_swarm_case(self, question: str, answer: str, item: Dict) -> Dict:
        """构造 Swarm 评估用例"""
        return {
            "id": f"flywheel_{uuid.uuid4().hex[:6]}",
            "question": question,
            "expected_subtask_count": 2,
            "expected_agents": ["consultation_agent"],
            "expected_coordination_mode": "swarm",
            "expected_behavior": f"专家审核后的标准回答: {answer[:100]}...",
        }

    def _append_to_json(self, filename: str, new_cases: List[Dict]):
        """追加到 JSON 文件"""
        filepath = self._data_dir / filename
        try:
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.extend(new_cases)

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"追加评估用例失败 ({filename}): {e}")
