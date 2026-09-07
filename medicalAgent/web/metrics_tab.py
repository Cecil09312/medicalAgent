"""
飞轮指标仪表盘
展示专家介入率、修正率、审核耗时、知识库增长等指标

数据来源设计：
- 审核相关指标（审核数/批准/修正/拒绝/耗时）：实时从审核队列统计，
  与专家审核 Tab 同源，保证两处数据一致
- 总查询数/知识库贡献/评估用例：由 FlywheelMetrics 累计
- 历史快照：由定时器自动拍摄，仅在数据有变化时产生新行
"""
import sys
from pathlib import Path
from typing import List
from loguru import logger

import gradio as gr

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 自动快照间隔（秒）：定时同步数据，有变化时拍摄快照
SNAPSHOT_INTERVAL_SECONDS = 60


def _get_metrics():
    """获取 FlywheelMetrics 实例（进程级单例，与记录方共享数据）"""
    try:
        from flywheel.metrics_tracker import get_default_metrics
        return get_default_metrics()
    except Exception as e:
        logger.error(f"FlywheelMetrics 初始化失败: {e}")
        return None


def _collect_review_stats() -> dict:
    """
    从审核队列实时统计审核指标

    审核队列是持久化真源（与专家审核 Tab 同源），以此计算保证两处数据一致
    """
    try:
        from review.expert_interface import ExpertInterface
        items = ExpertInterface().list_all()
    except Exception as e:
        logger.warning(f"读取审核队列失败: {e}")
        return {"total_reviews": 0, "corrections": 0, "approvals": 0, "rejections": 0, "avg_time": 0.0}

    reviewed = [i for i in items if i.status in ("approved", "corrected", "rejected")]
    times = [
        (i.reviewed_at - i.created_at)
        for i in reviewed
        if i.reviewed_at and i.created_at
    ]
    return {
        "total_reviews": len(reviewed),
        "corrections": sum(1 for i in reviewed if i.status == "corrected"),
        "approvals": sum(1 for i in reviewed if i.status == "approved"),
        "rejections": sum(1 for i in reviewed if i.status == "rejected"),
        "avg_time": (sum(times) / len(times)) if times else 0.0,
    }


def _sync_metrics():
    """
    以审核队列为真源同步审核指标到 FlywheelMetrics

    审核次数/批准/修正/拒绝/耗时直接由队列记录计算（覆盖内存计数，消除不一致）；
    总查询数、知识库贡献、评估用例仍由 FlywheelMetrics 自身累计
    """
    metrics = _get_metrics()
    if metrics is None:
        return None

    stats = _collect_review_stats()
    c = metrics.get_current()
    c.total_reviews = stats["total_reviews"]
    c.total_corrections = stats["corrections"]
    c.total_approvals = stats["approvals"]
    c.total_rejections = stats["rejections"]
    c.avg_review_time_seconds = stats["avg_time"]
    c.intervention_rate = c.total_reviews / c.total_queries if c.total_queries > 0 else 0.0
    c.correction_rate = c.total_corrections / c.total_reviews if c.total_reviews > 0 else 0.0
    return metrics


def _snapshot_if_changed(metrics) -> bool:
    """数据较上一次快照有变化时才拍摄新快照（避免产生重复无意义行）"""
    history = metrics.get_history(1)
    c = metrics.get_current()

    # 无任何数据且无历史时，不产生全 0 的空快照
    if not history and c.total_queries == 0 and c.total_reviews == 0:
        return False

    if history:
        last = history[-1]
        unchanged = (
            last.total_queries == c.total_queries
            and last.total_reviews == c.total_reviews
            and last.total_corrections == c.total_corrections
            and last.total_approvals == c.total_approvals
            and last.total_rejections == c.total_rejections
            and last.kb_expert_docs == c.kb_expert_docs
            and last.eval_cases_from_flywheel == c.eval_cases_from_flywheel
        )
        if unchanged:
            return False

    metrics.take_snapshot()
    return True


def _build_metrics_markdown():
    """构建核心指标卡片 Markdown（展示前先同步审核队列真源数据）"""
    metrics = _sync_metrics()
    if metrics is None:
        return "⚠️ 飞轮指标系统不可用"

    snapshot = metrics.get_current()

    # 格式化指标（耗时超过 2 分钟时以 min 显示，更友好）
    intervention_pct = f"{snapshot.intervention_rate:.1%}"
    correction_pct = f"{snapshot.correction_rate:.1%}"
    avg_sec = snapshot.avg_review_time_seconds
    avg_time = f"{avg_sec:.1f}s" if avg_sec < 120 else f"{avg_sec / 60:.1f}min"

    md = f"""
| 指标 | 值 | 说明 |
|------|-----|------|
| 📊 专家介入率 | **{intervention_pct}** | 总查询中需要专家审核的比例（应随时间下降） |
| ✏️ 修正率 | **{correction_pct}** | 审核中需要修正的比例 |
| ⏱️ 平均审核耗时 | **{avg_time}** | 专家审核平均耗时 |
| 📚 知识库专家贡献 | **{snapshot.kb_expert_docs}** 篇 | 专家修正注入知识库的文档数 |
| 📝 飞轮评估用例 | **{snapshot.eval_cases_from_flywheel}** 条 | 飞轮自动扩充的评估用例数 |

---

**原始数据:** 总查询 {snapshot.total_queries} | 总审核 {snapshot.total_reviews} | 修正 {snapshot.total_corrections} | 批准 {snapshot.total_approvals} | 拒绝 {snapshot.total_rejections}
"""
    return md


def _build_history_table():
    """构建历史快照表格"""
    metrics = _get_metrics()
    if metrics is None:
        return []

    history = metrics.get_history(last_n=20)
    if not history:
        return []

    rows = []
    for snap in reversed(history):
        from datetime import datetime
        ts = datetime.fromtimestamp(snap.timestamp).strftime("%Y-%m-%d %H:%M")
        rows.append([
            ts,
            snap.total_queries,
            snap.total_reviews,
            f"{snap.intervention_rate:.1%}",
            f"{snap.correction_rate:.1%}",
            f"{snap.avg_review_time_seconds:.1f}s",
            snap.kb_expert_docs,
            snap.eval_cases_from_flywheel,
        ])
    return rows


# 趋势图缓存：历史快照数未变化时直接复用，避免重复渲染拖慢刷新
_trend_cache = {"count": -1, "fig": None}


def _build_trend_data():
    """构建介入率趋势数据（用于 Plot）"""
    metrics = _get_metrics()
    if metrics is None:
        return None

    history = metrics.get_history(last_n=20)

    # 快照数未变化时不更新组件：gr.update() 表示保持前端原样，
    # 避免 Gradio 对同一 Figure 重复做 matplotlib→plotly 转换与前端重绘（刷新慢的主因）
    if len(history) == _trend_cache["count"] and _trend_cache["fig"] is not None:
        return gr.update()

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        # matplotlib 默认字体不支持中文，需指定中文字体避免乱码方块
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        from datetime import datetime

        # 无历史快照时显示提示（快照由定时器每分钟自动拍摄）
        if not history:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5, 0.5,
                "暂无历史数据\n数据将随使用自动积累（每分钟拍摄一次快照）",
                ha="center", va="center", fontsize=13, color="gray",
            )
            ax.axis("off")
            fig.tight_layout()
            plt.close(fig)
            # 空数据提示图同样入缓存，避免每次刷新重复生成
            _trend_cache["count"] = len(history)
            _trend_cache["fig"] = fig
            return fig

        timestamps = [datetime.fromtimestamp(s.timestamp).strftime("%m-%d %H:%M") for s in history]
        intervention_rates = [s.intervention_rate * 100 for s in history]
        correction_rates = [s.correction_rate * 100 for s in history]

        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax1.set_xlabel("时间")
        ax1.set_ylabel("专家介入率 (%)", color="tab:blue")
        ax1.plot(timestamps, intervention_rates, "o-", color="tab:blue", label="介入率")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_ylim(0, max(intervention_rates + [10]) * 1.2)

        ax2 = ax1.twinx()
        ax2.set_ylabel("修正率 (%)", color="tab:orange")
        ax2.plot(timestamps, correction_rates, "s--", color="tab:orange", label="修正率")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0, max(correction_rates + [10]) * 1.2)

        # 不设置图内标题：Gradio Plot 组件的 label 已显示在左上角，避免重叠
        fig.tight_layout()
        plt.close(fig)

        _trend_cache["count"] = len(history)
        _trend_cache["fig"] = fig
        return fig
    except Exception as e:
        logger.warning(f"生成趋势图失败: {e}")
        return None


_DIMENSION_LABELS = {
    "disease_category": "疾病类别",
    "skill_name": "触发 Skill",
    "agent_type": "Agent 类型",
}


def _build_dimension_markdown() -> str:
    """构建维度下钻表（按疾病类别 / Skill / Agent 聚合，修正数降序 Top 10）"""
    metrics = _get_metrics()
    if metrics is None:
        return ""

    try:
        sections = []
        for dim, label in _DIMENSION_LABELS.items():
            rows = metrics.get_dimension_breakdown(dim, top_n=10)
            # 仅展示该维度有审核数据的取值；无数据时显示占位说明
            if not rows:
                sections.append(f"**{label}**：暂无审核数据（数据将随专家审核自动积累）\n")
                continue
            lines = [
                f"**{label}**（按修正数降序，定位薄弱环节）\n",
                "| 取值 | 审核数 | 修正数 | 修正率 |",
                "|------|--------|--------|--------|",
            ]
            for value, reviews, corrections, rate in rows:
                lines.append(f"| {value} | {reviews} | {corrections} | {rate:.1%} |")
            sections.append("\n".join(lines) + "\n")

        if all("暂无审核数据" in s for s in sections):
            return "暂无维度数据。专家审核若干回答后，此处将按疾病类别 / Skill / Agent 展示修正率下钻。"
        return "\n".join(sections)
    except Exception as e:
        logger.warning(f"构建维度下钻表失败: {e}")
        return ""


def _refresh_metrics():
    """刷新展示（同步队列数据，但不拍摄快照；快照由定时器负责）"""
    _sync_metrics()
    return (
        _build_metrics_markdown(),
        _build_history_table(),
        _build_trend_data(),
        _build_dimension_markdown(),
    )


def _auto_snapshot_and_refresh():
    """定时器回调：同步数据，有变化时拍摄快照，并刷新展示"""
    metrics = _sync_metrics()
    if metrics is not None:
        _snapshot_if_changed(metrics)
    return (
        _build_metrics_markdown(),
        _build_history_table(),
        _build_trend_data(),
        _build_dimension_markdown(),
    )


def create_metrics_tab():
    """创建飞轮指标 Tab"""
    # 核心指标
    gr.Markdown("##### 📈 核心指标")
    metrics_md = gr.Markdown(value=_build_metrics_markdown, elem_classes="medix-card medix-center")

    # 趋势图
    gr.Markdown("##### 📉 趋势图")
    trend_plot = gr.Plot(
        value=_build_trend_data,
        label="介入率 / 修正率趋势",
    )

    # 历史快照表
    gr.Markdown("##### 🗂 历史快照")
    history_table = gr.Dataframe(
        value=_build_history_table,
        headers=["时间", "总查询", "总审核", "介入率", "修正率", "审核耗时", "KB贡献", "评估用例"],
        interactive=False,
    )

    # 维度下钻（按疾病类别 / Skill / Agent 聚合，定位薄弱环节）
    gr.Markdown("##### 🔍 维度下钻")
    dimension_md = gr.Markdown(value=_build_dimension_markdown, elem_classes="medix-card")

    # 刷新按钮：仅刷新展示，不产生历史快照
    refresh_btn = gr.Button("🔄 刷新指标", variant="primary")
    refresh_btn.click(
        _refresh_metrics,
        outputs=[metrics_md, history_table, trend_plot, dimension_md],
    )

    # 定时器：每 60 秒自动同步一次，数据有变化才拍摄历史快照
    timer = gr.Timer(SNAPSHOT_INTERVAL_SECONDS)
    timer.tick(
        _auto_snapshot_and_refresh,
        outputs=[metrics_md, history_table, trend_plot, dimension_md],
    )
