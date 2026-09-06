"""
飞轮指标追踪
追踪专家介入率、修正率、知识库增长等指标
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger


@dataclass
class FlywheelSnapshot:
    """飞轮指标快照"""
    timestamp: float = 0.0
    total_queries: int = 0
    total_reviews: int = 0
    total_corrections: int = 0
    total_approvals: int = 0
    total_rejections: int = 0
    intervention_rate: float = 0.0       # 专家介入率 = total_reviews / total_queries
    correction_rate: float = 0.0         # 修正率 = total_corrections / total_reviews
    avg_review_time_seconds: float = 0.0 # 平均审核耗时
    kb_expert_docs: int = 0              # 知识库中专家贡献文档数
    eval_cases_from_flywheel: int = 0    # 飞轮贡献的评估用例数

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FlywheelMetrics:
    """飞轮指标追踪器"""

    def __init__(self, metrics_path: Optional[str] = None):
        """
        初始化

        Args:
            metrics_path: 指标持久化路径
        """
        if metrics_path is None:
            metrics_path = str(Path(__file__).parent / "data" / "flywheel_metrics.json")
        self._metrics_path = Path(metrics_path)
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self._history: List[FlywheelSnapshot] = []
        self._current = FlywheelSnapshot()
        self._review_times: List[float] = []

        self._load()

    def _load(self):
        """加载历史指标（兼容旧格式：纯历史快照列表）"""
        if self._metrics_path.exists():
            try:
                with open(self._metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # 新格式：{"current": 当前累计, "history": 历史快照}
                    self._history = [FlywheelSnapshot(**s) for s in data.get("history", [])]
                    if data.get("current"):
                        self._current = FlywheelSnapshot(**data["current"])
                    elif self._history:
                        self._current = self._history[-1]
                else:
                    # 旧格式：纯历史快照列表
                    self._history = [FlywheelSnapshot(**s) for s in data]
                    if self._history:
                        self._current = self._history[-1]
            except Exception as e:
                logger.warning(f"加载飞轮指标失败: {e}")

    def _save(self):
        """持久化指标（同时保存当前累计值，进程重启后不丢失；原子写入，避免 JSON 损坏）"""
        tmp_path = None
        try:
            data = {
                "current": asdict(self._current),
                "history": [asdict(s) for s in self._history],
            }
            # 原子写入：先写同目录临时文件，写完成功后用 os.replace 原子替换目标文件，
            # 防止写入中途进程崩溃留下损坏的 JSON，导致下次启动 _load() 解析失败
            tmp_path = Path(f"{self._metrics_path}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._metrics_path)
        except Exception as e:
            logger.error(f"保存飞轮指标失败: {e}")
            # 写入失败时清理残留的临时文件，避免遗留 .tmp 垃圾文件
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"清理飞轮指标临时文件失败: {cleanup_error}")

    def record_query(self):
        """记录一次查询"""
        self._current.total_queries += 1
        self._recalculate()

    def record_review(self, review_time_seconds: float = 0.0):
        """记录一次审核"""
        self._current.total_reviews += 1
        if review_time_seconds > 0:
            self._review_times.append(review_time_seconds)
        self._recalculate()

    def record_correction(self):
        """记录一次修正"""
        self._current.total_corrections += 1
        self._recalculate()

    def record_approval(self):
        """记录一次批准"""
        self._current.total_approvals += 1
        self._recalculate()

    def record_rejection(self):
        """记录一次拒绝"""
        self._current.total_rejections += 1
        self._recalculate()

    def record_kb_ingest(self, doc_count: int = 1):
        """记录知识库注入"""
        self._current.kb_expert_docs += doc_count
        self._save()

    def record_eval_expansion(self, case_count: int = 1):
        """记录评估集扩充"""
        self._current.eval_cases_from_flywheel += case_count
        self._save()

    def _recalculate(self):
        """重新计算派生指标并立即持久化（进程重启不丢失）"""
        q = self._current.total_queries
        r = self._current.total_reviews
        c = self._current.total_corrections

        self._current.intervention_rate = r / q if q > 0 else 0.0
        self._current.correction_rate = c / r if r > 0 else 0.0

        if self._review_times:
            self._current.avg_review_time_seconds = sum(self._review_times) / len(self._review_times)

        self._save()

    def take_snapshot(self) -> FlywheelSnapshot:
        """
        拍摄当前快照并保存到历史

        Returns:
            当前快照
        """
        snapshot = FlywheelSnapshot(
            timestamp=time.time(),
            total_queries=self._current.total_queries,
            total_reviews=self._current.total_reviews,
            total_corrections=self._current.total_corrections,
            total_approvals=self._current.total_approvals,
            total_rejections=self._current.total_rejections,
            intervention_rate=self._current.intervention_rate,
            correction_rate=self._current.correction_rate,
            avg_review_time_seconds=self._current.avg_review_time_seconds,
            kb_expert_docs=self._current.kb_expert_docs,
            eval_cases_from_flywheel=self._current.eval_cases_from_flywheel,
        )
        self._history.append(snapshot)
        self._save()
        logger.debug(f"飞轮快照已保存: intervention_rate={snapshot.intervention_rate:.3f}")
        return snapshot

    def get_current(self) -> FlywheelSnapshot:
        """获取当前指标"""
        return self._current

    def get_history(self, last_n: int = 10) -> List[FlywheelSnapshot]:
        """获取最近 N 个快照"""
        return self._history[-last_n:]

    def summary(self) -> str:
        """生成摘要"""
        c = self._current
        lines = [
            "飞轮指标:",
            f"  总查询数:       {c.total_queries}",
            f"  总审核数:       {c.total_reviews}",
            f"  专家介入率:     {c.intervention_rate:.1%}",
            f"  修正率:         {c.correction_rate:.1%}",
            f"  平均审核耗时:   {c.avg_review_time_seconds:.1f}s",
            f"  知识库专家贡献: {c.kb_expert_docs} 文档",
            f"  飞轮评估用例:   {c.eval_cases_from_flywheel} 条",
        ]
        return "\n".join(lines)


# ===================== 进程级默认实例（单例） =====================

_default_metrics: Optional[FlywheelMetrics] = None


def get_default_metrics() -> FlywheelMetrics:
    """
    获取进程级默认飞轮指标实例（单例）

    指标的累计值保存在实例内存中，记录方（提问/审核）与展示方（指标页）
    必须共享同一实例，否则展示端永远看不到记录端的数据
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = FlywheelMetrics()
    return _default_metrics
