"""
专家审核面板
展示待审核列表、审核详情、批准/修正/拒绝操作
"""
import sys
from pathlib import Path
from typing import List, Optional
from loguru import logger

import gradio as gr

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _get_expert_interface():
    """获取 ExpertInterface 实例"""
    try:
        from review.expert_interface import ExpertInterface
        return ExpertInterface()
    except Exception as e:
        logger.error(f"ExpertInterface 初始化失败: {e}")
        return None


def _load_pending_table():
    """加载待审核列表，返回 Dataframe 数据"""
    expert = _get_expert_interface()
    if expert is None:
        return []

    pending = expert.list_pending()
    if not pending:
        return []

    rows = []
    for item in pending:
        rows.append([
            item.id,
            item.trigger_reason,
            item.review_mode,
            item.question[:80] + ("..." if len(item.question) > 80 else ""),
            item.original_answer[:80] + ("..." if len(item.original_answer) > 80 else ""),
        ])
    return rows


def _load_stats_markdown():
    """加载审核统计 Markdown"""
    expert = _get_expert_interface()
    if expert is None:
        return "⚠️ 审核系统不可用"

    stats = expert.get_stats()
    return (
        f"**待审核:** {stats.get('pending', 0)} | "
        f"**已批准:** {stats.get('approved', 0)} | "
        f"**已修正:** {stats.get('corrected', 0)} | "
        f"**已拒绝:** {stats.get('rejected', 0)} | "
        f"**总计:** {stats.get('total', 0)}"
    )


def _show_detail(evt: gr.SelectData):
    """
    点击表格行时显示详情

    Args:
        evt: 选择事件，包含行索引

    Returns:
        (question_md, answer_md, review_id_state)
    """
    expert = _get_expert_interface()
    if expert is None:
        return "系统不可用", "", ""

    pending = expert.list_pending()
    if not pending:
        return "没有待审核项", "", ""

    row_idx = evt.index[0] if evt.index else 0
    if row_idx >= len(pending):
        return "请选择有效的审核项", "", ""

    item = pending[row_idx]
    detail = expert.get_detail(item.id)
    if not detail:
        return "无法获取详情", "", ""

    question_md = f"**问题:**\n\n{detail['question']}"
    answer_md = (
        f"**原始回答:**\n\n{detail['original_answer']}\n\n"
        f"---\n"
        f"**触发原因:** `{detail['trigger_reason']}` | "
        f"**审核模式:** `{detail['review_mode']}` | "
        f"**状态:** `{detail['status']}`"
    )

    return question_md, answer_md, item.id


def _approve_review(review_id: str, expert_id: str):
    """批准审核"""
    if not review_id:
        return "⚠️ 请先选择一个审核项", _load_stats_markdown(), _load_pending_table()

    expert = _get_expert_interface()
    if expert is None:
        return "⚠️ 审核系统不可用", _load_stats_markdown(), _load_pending_table()

    ok = expert.review(review_id, "approve", expert_id=expert_id or "web_expert")
    if ok:
        return f"✅ 已批准 `{review_id}`", _load_stats_markdown(), _load_pending_table()
    else:
        return f"⚠️ 批准失败（可能已处理）", _load_stats_markdown(), _load_pending_table()


def _submit_correction(review_id: str, correction: str, expert_id: str):
    """提交修正"""
    if not review_id:
        return "⚠️ 请先选择一个审核项", _load_stats_markdown(), _load_pending_table()
    if not correction.strip():
        return "⚠️ 请输入修正内容", _load_stats_markdown(), _load_pending_table()

    expert = _get_expert_interface()
    if expert is None:
        return "⚠️ 审核系统不可用", _load_stats_markdown(), _load_pending_table()

    ok = expert.review(
        review_id, "correct",
        correction=correction,
        expert_id=expert_id or "web_expert",
    )
    if ok:
        return f"✅ 已修正 `{review_id}`", _load_stats_markdown(), _load_pending_table()
    else:
        return f"⚠️ 修正失败（可能已处理）", _load_stats_markdown(), _load_pending_table()


def _reject_review(review_id: str, expert_id: str):
    """拒绝审核"""
    if not review_id:
        return "⚠️ 请先选择一个审核项", _load_stats_markdown(), _load_pending_table()

    expert = _get_expert_interface()
    if expert is None:
        return "⚠️ 审核系统不可用", _load_stats_markdown(), _load_pending_table()

    ok = expert.review(review_id, "reject", expert_id=expert_id or "web_expert")
    if ok:
        return f"✅ 已拒绝 `{review_id}`", _load_stats_markdown(), _load_pending_table()
    else:
        return f"⚠️ 拒绝失败（可能已处理）", _load_stats_markdown(), _load_pending_table()


def _refresh_all():
    """刷新所有数据"""
    return _load_stats_markdown(), _load_pending_table(), "", "", ""


def create_review_tab():
    """创建专家审核 Tab"""
    # 顶部：统计信息 + 刷新
    with gr.Row():
        with gr.Column(scale=4):
            stats_md = gr.Markdown(value=_load_stats_markdown, elem_classes="medix-card")
        refresh_btn = gr.Button("🔄 刷新", scale=1, min_width=100)

    # 待审核列表
    gr.Markdown("##### 📋 待审核列表")
    pending_table = gr.Dataframe(
        value=_load_pending_table,
        headers=["ID", "触发原因", "模式", "问题", "回答"],
        interactive=False,
        wrap=True,
    )

    # 详情面板
    gr.Markdown("##### 📄 审核详情")
    review_id_state = gr.State(value="")

    with gr.Row():
        question_detail = gr.Markdown("点击上方表格行查看详情", elem_classes="medix-card")
        answer_detail = gr.Markdown("", elem_classes="medix-card")

    # 表格点击事件
    pending_table.select(
        _show_detail,
        outputs=[question_detail, answer_detail, review_id_state],
    )

    # 操作区
    gr.Markdown("##### ⚙️ 审核操作")
    with gr.Row():
        expert_id_input = gr.Textbox(
            label="专家 ID",
            placeholder="输入您的 ID（可选）",
            scale=2,
        )
        approve_btn = gr.Button("✅ 批准", variant="primary", scale=1, min_width=100)
        reject_btn = gr.Button("❌ 拒绝", variant="stop", scale=1, min_width=100)

    with gr.Row():
        correction_input = gr.Textbox(
            label="修正内容",
            placeholder="输入修正后的回答...",
            lines=3,
            scale=3,
        )
        submit_correction_btn = gr.Button("📝 提交修正", variant="primary", scale=1, min_width=120)

    action_status = gr.Markdown("")

    # 绑定事件
    approve_btn.click(
        _approve_review,
        inputs=[review_id_state, expert_id_input],
        outputs=[action_status, stats_md, pending_table],
    )

    reject_btn.click(
        _reject_review,
        inputs=[review_id_state, expert_id_input],
        outputs=[action_status, stats_md, pending_table],
    )

    submit_correction_btn.click(
        _submit_correction,
        inputs=[review_id_state, correction_input, expert_id_input],
        outputs=[action_status, stats_md, pending_table],
    )

    refresh_btn.click(
        _refresh_all,
        outputs=[stats_md, pending_table, question_detail, answer_detail, action_status],
    )
