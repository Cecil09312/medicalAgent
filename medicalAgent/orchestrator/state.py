"""
编排图状态定义
数据流:问题 -> 记忆 -> 分解 -> 并行 Worker -> 综合 -> 后处理
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def or_reducer(left: bool, right: bool) -> bool:
    """并行 Worker 分支的布尔合并:任一分支为 True 即为 True"""
    return bool(left or right)


class MedicalGraphState(TypedDict, total=False):
    """医疗编排图的全局状态(Send 分支会带上完整副本,worker 节点内含 current_task)"""

    question: str
    session_id: Optional[str]
    context: Optional[Dict[str, Any]]

    # load_context 节点写入
    memories: Optional[List[Dict[str, Any]]]

    # decompose 节点写入
    decomposition: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    mode: str  # "single" | "swarm"

    # Send 分支私有输入
    current_task: Dict[str, Any]

    # worker 节点并行合并(reducer 必须显式声明,否则并行更新冲突)
    worker_results: Annotated[List[Dict[str, Any]], operator.add]
    worker_timeout: Annotated[bool, or_reducer]

    # finalize/synthesize 节点写入,postprocess 节点更新
    result: Dict[str, Any]
