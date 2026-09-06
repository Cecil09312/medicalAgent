"""
审核队列管理
支持实时审核（阻塞等待）和异步审核（事后处理）
"""
import os
import json
import uuid
import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class ReviewItem:
    """审核项"""
    id: str
    question: str
    original_answer: str
    trigger_reason: str          # "high_risk" / "constraint_violation" / "random_sample" / "user_flag"
    review_mode: str             # "realtime" / "async"
    status: str = "pending"      # "pending" / "approved" / "corrected" / "rejected"
    expert_correction: Optional[str] = None
    expert_id: Optional[str] = None
    created_at: float = 0.0
    reviewed_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class ReviewResult:
    """审核结果"""
    action: str              # "approved" / "corrected" / "rejected" / "timeout"
    correction: Optional[str] = None
    expert_id: Optional[str] = None
    review_id: str = ""


class ReviewQueue:
    """审核队列管理器"""

    def __init__(self, data_path: Optional[str] = None):
        """
        初始化审核队列

        Args:
            data_path: 队列持久化路径（可选，默认从 .env 读取）
        """
        if data_path is None:
            data_path = os.getenv("REVIEW_DATA_PATH", "review/data/pending_reviews.json")

        self._data_path = Path(data_path)
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        self._items: Dict[str, ReviewItem] = {}
        self._realtime_events: Dict[str, asyncio.Event] = {}

        # 加载已有数据
        self._load()

    def _load(self):
        """从文件加载队列"""
        if self._data_path.exists():
            try:
                with open(self._data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item_data in data:
                    item = ReviewItem(**item_data)
                    self._items[item.id] = item
                logger.debug(f"加载了 {len(self._items)} 条审核记录")
            except Exception as e:
                logger.warning(f"加载审核队列失败: {e}")
                self._items = {}

    def _save(self):
        """持久化队列到文件（原子写入，避免进程崩溃导致 JSON 损坏）"""
        tmp_path = None
        try:
            data = [asdict(item) for item in self._items.values()]
            # 原子写入：先写同目录临时文件，写完成功后用 os.replace 原子替换目标文件，
            # 防止写入中途进程崩溃留下损坏的 JSON，导致下次启动 _load() 失败、数据清零
            tmp_path = Path(f"{self._data_path}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._data_path)
        except Exception as e:
            logger.error(f"保存审核队列失败: {e}")
            # 写入失败时清理残留的临时文件，避免遗留 .tmp 垃圾文件
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"清理审核队列临时文件失败: {cleanup_error}")

    def _create_item(
        self,
        question: str,
        answer: str,
        trigger_reason: str,
        review_mode: str,
    ) -> ReviewItem:
        """创建审核项"""
        item = ReviewItem(
            id=str(uuid.uuid4())[:8],
            question=question,
            original_answer=answer,
            trigger_reason=trigger_reason,
            review_mode=review_mode,
        )
        self._items[item.id] = item
        self._save()
        return item

    async def submit_realtime(
        self,
        question: str,
        answer: str,
        trigger_reason: str,
        timeout: Optional[int] = None,
    ) -> ReviewResult:
        """
        实时审核：阻塞等待专家审核结果

        Args:
            question: 用户问题
            answer: 原始回答
            trigger_reason: 触发原因
            timeout: 超时秒数（默认从 .env 读取）

        Returns:
            审核结果
        """
        if timeout is None:
            # 默认 15 秒：专家实时审核的合理响应窗口，避免无人审核时长时间阻塞回答
            timeout = int(os.getenv("REVIEW_REALTIME_TIMEOUT", "15"))

        item = self._create_item(question, answer, trigger_reason, "realtime")

        # 创建等待事件，并记录所属事件循环（专家审核可能在其他线程触发）
        event = asyncio.Event()
        loop = asyncio.get_running_loop()
        self._realtime_events[item.id] = (event, loop)

        logger.info(f"实时审核已提交 [review_id={item.id}], 等待专家审核 (超时={timeout}s)")

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"实时审核超时 [review_id={item.id}]")
            # 超时并非专家拒绝：转为异步待审核，保留在待办列表中供专家事后处理
            item.review_mode = "async"
            self._save()
            self._realtime_events.pop(item.id, None)
            return ReviewResult(action="timeout", review_id=item.id)

        # 获取结果
        result = ReviewResult(
            action=item.status if item.status in ("approved", "corrected", "rejected") else "approved",
            correction=item.expert_correction,
            expert_id=item.expert_id,
            review_id=item.id,
        )

        # 清理事件
        self._realtime_events.pop(item.id, None)
        return result

    def enqueue_async(
        self,
        question: str,
        answer: str,
        trigger_reason: str,
    ) -> str:
        """
        异步审核：入队，返回 review_id

        Args:
            question: 用户问题
            answer: 原始回答
            trigger_reason: 触发原因

        Returns:
            review_id
        """
        item = self._create_item(question, answer, trigger_reason, "async")
        logger.info(f"异步审核已入队 [review_id={item.id}, reason={trigger_reason}]")
        return item.id

    def _record_review_metrics(self, item: ReviewItem, action: str):
        """
        将审核结果接入飞轮指标

        审核耗时 = 专家处理时间 - 提交时间；指标记录失败不影响审核主流程
        """
        try:
            from flywheel.metrics_tracker import get_default_metrics
            metrics = get_default_metrics()
            review_time = max(0.0, item.reviewed_at - item.created_at)
            metrics.record_review(review_time)
            if action == "approved":
                metrics.record_approval()
            elif action == "corrected":
                metrics.record_correction()
            elif action == "rejected":
                metrics.record_rejection()
        except Exception as e:
            logger.warning(f"飞轮指标记录失败: {e}")

    def _notify_realtime(self, review_id: str):
        """
        唤醒等待该审核项的实时协程

        专家审核操作可能运行在其他线程（如 Gradio 线程池），
        asyncio.Event.set 非线程安全，需通过事件循环的 call_soon_threadsafe 调度
        """
        entry = self._realtime_events.get(review_id)
        if not entry:
            return
        event, loop = entry
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:
            # 事件循环已关闭（如等待方已退出），忽略
            logger.warning(f"实时审核事件循环已关闭，无法唤醒 [review_id={review_id}]")

    def submit_correction(
        self,
        review_id: str,
        correction: str,
        expert_id: str = "anonymous",
    ) -> bool:
        """
        专家提交修正

        Args:
            review_id: 审核项 ID
            correction: 修正后的回答
            expert_id: 专家 ID

        Returns:
            是否成功
        """
        item = self._items.get(review_id)
        if not item:
            logger.error(f"审核项不存在: {review_id}")
            return False

        if item.status != "pending":
            logger.warning(f"审核项已处理: {review_id} (status={item.status})")
            return False

        item.expert_correction = correction
        item.expert_id = expert_id
        item.status = "corrected"
        item.reviewed_at = time.time()
        self._save()

        # 接入飞轮指标
        self._record_review_metrics(item, "corrected")

        # 如果是实时审核，唤醒等待的协程
        self._notify_realtime(review_id)

        logger.info(f"专家修正已提交 [review_id={review_id}, expert={expert_id}]")
        return True

    def submit_approval(
        self,
        review_id: str,
        expert_id: str = "anonymous",
    ) -> bool:
        """
        专家批准（无需修正）

        Args:
            review_id: 审核项 ID
            expert_id: 专家 ID

        Returns:
            是否成功
        """
        item = self._items.get(review_id)
        if not item or item.status != "pending":
            return False

        item.status = "approved"
        item.expert_id = expert_id
        item.reviewed_at = time.time()
        self._save()

        # 接入飞轮指标
        self._record_review_metrics(item, "approved")

        self._notify_realtime(review_id)

        logger.info(f"专家已批准 [review_id={review_id}]")
        return True

    def submit_rejection(
        self,
        review_id: str,
        expert_id: str = "anonymous",
    ) -> bool:
        """专家拒绝"""
        item = self._items.get(review_id)
        if not item or item.status != "pending":
            return False

        item.status = "rejected"
        item.expert_id = expert_id
        item.reviewed_at = time.time()
        self._save()

        self._record_review_metrics(item, "rejected")

        self._notify_realtime(review_id)

        return True

    def get_pending(self, status: str = "pending") -> List[ReviewItem]:
        """获取指定状态的审核列表"""
        return [item for item in self._items.values() if item.status == status]

    def get_item(self, review_id: str) -> Optional[ReviewItem]:
        """获取单个审核项"""
        return self._items.get(review_id)

    def get_stats(self) -> Dict[str, int]:
        """获取审核统计"""
        stats = {"pending": 0, "approved": 0, "corrected": 0, "rejected": 0, "total": 0}
        for item in self._items.values():
            stats[item.status] = stats.get(item.status, 0) + 1
            stats["total"] += 1
        return stats


# ===================== 进程级默认队列（单例） =====================

_default_queue: Optional[ReviewQueue] = None


def get_default_queue() -> ReviewQueue:
    """
    获取进程级默认审核队列（单例）

    实时审核的等待事件保存在队列实例内存中，提问方与专家审核方必须共享
    同一实例，专家操作才能唤醒等待的协程，因此各模块统一使用此单例
    """
    global _default_queue
    if _default_queue is None:
        _default_queue = ReviewQueue()
    return _default_queue
