"""
用户对话界面
基于 Gradio Chatbot，调用多Agent编排系统处理问题
"""
import os
import sys
import time
import uuid
import random
import asyncio
from pathlib import Path
from loguru import logger

import gradio as gr

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 全局单例：多Agent编排系统
_coordinator = None
_short_term_memory = None
# 长期记忆在 _get_coordinator() 中懒初始化；模块级先置 None，
# 避免尚未提问（协调器未初始化）就点击"重置会话"时引用未定义变量
_long_term_memory = None
_session_id = str(uuid.uuid4())

# 触发实时审核的高风险关键词：统一使用医疗安全硬约束的配置（constraints/emergency.py），
# 与后端 WorkerRunner / 编排器的判定口径保持一致
def _hit_emergency_keyword(text: str) -> bool:
    """检测文本是否命中高危症状关键词（硬约束配置统一维护）"""
    try:
        from constraints.emergency import detect_emergency
        return detect_emergency(text) is not None
    except Exception:
        # 模块不可用时退回内置最小关键词集，保证高风险场景仍可触发审核
        fallback = ["胸痛", "昏迷", "休克", "大出血", "呼吸困难", "中风", "心梗"]
        return any(kw in text for kw in fallback)


# 异步审核随机采样率（不阻塞用户）
REVIEW_SAMPLE_RATE = 0.05

# 固定免责声明横幅（医疗合规要求：始终可见，不随对话滚动消失）
DISCLAIMER_BANNER_HTML = (
    "<div style='display:flex;align-items:center;gap:8px;padding:8px 14px;"
    "border:1px solid #f0c36d;border-left:4px solid #e8a33d;border-radius:8px;"
    "background:#fdf6e8;color:#7a5b1e;font-size:13px;line-height:1.5;'>"
    "⚕️ <b>免责声明：</b>本系统提供的信息不构成医疗诊断建议，不能替代执业医师诊疗。"
    "若出现胸痛、呼吸困难、大出血、意识障碍等急症表现，请立即拨打 120 或前往急诊科就医。"
    "</div>"
)

# 欢迎语：首次打开及重置会话时展示
WELCOME_MESSAGE = (
    "您好，欢迎使用 **AI智能医疗诊断系统**。\n\n"
    "我可以为您提供：\n"
    "- 🩺 **症状咨询** — 症状初步分析与就医建议\n"
    "- 🔬 **诊断辅助** — 结合病史、检验指标的诊断参考\n"
    "- 📚 **医学研究** — 医学文献与临床指南解读\n\n"
    "> ⚠️ 本系统提供的信息仅供参考，不能替代执业医师的诊断。\n\n"
    "点击下方快速提问卡片，或直接输入您的问题开始咨询。"
)


def _get_coordinator():
    """懒初始化编排器（单例）"""
    global _coordinator, _short_term_memory

    if _coordinator is not None:
        return _coordinator

    try:
        from core.llm_client import LLMClient
        from orchestrator.factory import build_orchestrator
        from memory.short_term import ShortTermMemory

        llm_client = LLMClient()

        # 短期记忆（后端由 .env 的 MEMORY_BACKEND 决定：memory / redis 持久化）
        _short_term_memory = ShortTermMemory()
        logger.info(f"短期记忆初始化完成 (backend={_short_term_memory.backend})")

        # 创建编排器（LangGraph 多Agent编排，Worker Agent 集合由工厂统一组装）
        _coordinator = build_orchestrator(
            llm_client=llm_client,
            short_term_memory=_short_term_memory,
        )

        logger.info(f"编排器初始化成功 (type={type(_coordinator).__name__})")

        # 长期记忆（Mem0 云服务，不可用时自动降级本地存储）
        global _long_term_memory
        from memory.long_term import LongTermMemory
        _long_term_memory = LongTermMemory()
        logger.info(f"长期记忆初始化完成 (mem0={_long_term_memory._use_mem0})")

        # 后台线程预热知识库（提前加载嵌入模型，约数十秒的一次性开销），
        # 避免首个提问承担模型加载时间；search_knowledge 内的锁保证不会重复初始化
        import threading

        def _warmup_kb():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "skill_search_knowledge",
                    _project_root / ".claude" / "skills" / "search-knowledge" / "script" / "search.py",
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.get_knowledge_base()
                logger.info("知识库预热完成")

                # 重排序模型预热（仅在 RERANK_ENABLED 开启时）：提前下载/加载
                # CrossEncoder，避免首个提问承担模型加载耗时；关闭时不导入该模块
                if os.getenv("RERANK_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on"):
                    from knowledge.reranker import get_default_reranker
                    get_default_reranker()
                    logger.info("重排序模型预热完成")
            except Exception as e:
                logger.warning(f"知识库预热失败: {e}")

        threading.Thread(target=_warmup_kb, daemon=True).start()
    except Exception as e:
        logger.error(f"编排器初始化失败: {e}")
        _coordinator = None

    return _coordinator


async def _run_review(question: str, answer: str) -> str:
    """
    提交实时审核，返回最终展示文本

    审核结果处理：
    - approved：原样展示
    - corrected：展示专家修正内容
    - rejected：提示回答被撤回
    - timeout：提示已转异步审核，原回答先行展示
    """
    from review.review_queue import get_default_queue

    queue = get_default_queue()

    async def _submit():
        return await queue.submit_realtime(
            question=question,
            answer=answer,
            trigger_reason="high_risk",
        )

    # 启动审核任务（不取消，便于超时后补入异步队列）
    worker = asyncio.create_task(_submit())
    result = await worker

    if result.action == "approved":
        return answer
    elif result.action == "corrected":
        return (
            f"⚠️ **专家已修正回答：**\n\n{result.correction}\n\n"
            f"---\n*原回答经专家审核修正 (审核编号: {result.review_id})*"
        )
    elif result.action == "rejected":
        return (
            "⚠️ **该回答未通过专家审核，已被撤回。**\n\n"
            "您的问题涉及高风险症状，请尽快就医或拨打急救电话。"
        )
    else:  # timeout：队列内部已自动转为异步审核项，无需再次入队
        return (
            f"{answer}\n\n---\n"
            f"*⏳ 实时审核超时，该回答已转入异步审核队列，专家稍后将复核。*"
        )


async def _process_question(user_message: str, history: list) -> str:
    """
    处理用户问题，返回回答文本

    Args:
        user_message: 用户输入
        history: 对话历史 (Gradio messages 格式)

    Returns:
        回答文本
    """
    global _session_id

    coordinator = _get_coordinator()
    if coordinator is None:
        return "⚠️ 系统初始化失败，请检查配置（.env 文件中的 LLM_API_KEY 等）。"

    start_time = time.time()

    try:
        # 检索长期记忆中的相似历史会话，注入为参考上下文（失败不影响主流程）
        llm_input = user_message
        if _long_term_memory is not None:
            try:
                memories = _long_term_memory.search_similar_sessions(user_message, top_k=2)
                refs = []
                for m in memories:
                    text = m.get("memory") or m.get("summary") or ""
                    if text:
                        refs.append(f"- {text}")
                if refs:
                    llm_input = (
                        "【历史相似会话参考】\n" + "\n".join(refs)
                        + "\n\n【本次问题】\n" + user_message
                    )
                    logger.info(f"已注入 {len(refs)} 条长期记忆")
            except Exception as e:
                logger.warning(f"长期记忆检索失败: {e}")

        result = await coordinator.process(
            question=llm_input,
            session_id=_session_id,
        )

        # 记录飞轮指标：每次提问计一次查询（失败不影响主流程）
        try:
            from flywheel.metrics_tracker import get_default_metrics
            get_default_metrics().record_query()
        except Exception as e:
            logger.warning(f"飞轮指标记录失败: {e}")

        elapsed = time.time() - start_time
        answer = result.get("answer", "抱歉，暂时无法回答您的问题。")
        # 空回答兜底：若回答为 None、空字符串或纯空白字符，替换为友好提示，避免向用户展示空内容
        if answer is None or not str(answer).strip():
            answer = "抱歉，暂时无法生成回答，请重试。"
        suggestions = result.get("suggestions", [])

        # 审核触发逻辑：高危症状关键词触发实时审核（需配置开启），否则 5% 随机采样异步审核
        realtime_enabled = os.getenv("REVIEW_REALTIME_ENABLED", "false").lower() == "true"
        hit_keyword = _hit_emergency_keyword(user_message)

        if hit_keyword and realtime_enabled:
            answer = await _run_review(user_message, answer)
        elif random.random() < REVIEW_SAMPLE_RATE:
            try:
                from review.review_queue import get_default_queue
                # 附加维度信息（疾病类别粗分类），供飞轮指标下钻
                dimensions = None
                try:
                    from flywheel.dimensions import classify_disease_category
                    dimensions = {"disease_category": classify_disease_category(user_message)}
                except Exception:
                    pass
                get_default_queue().enqueue_async(
                    question=user_message,
                    answer=answer[:1000],
                    trigger_reason="random_sample",
                    dimensions=dimensions,
                )
            except Exception as e:
                logger.warning(f"异步审核采样入队失败: {e}")

        # 格式化回答
        output_parts = [answer]

        if suggestions:
            output_parts.append("\n\n💡 **相关建议：**")
            for i, s in enumerate(suggestions, 1):
                output_parts.append(f"  {i}. {s}")

        output_parts.append(
            f"\n\n---\n⏱️ 耗时: {elapsed:.2f}s"
        )
        output_parts.append(
            "\n⚕️ *免责声明：以上信息仅供参考，不能替代专业医疗建议。如有健康问题，请咨询专业医生。*"
        )

        return "\n".join(output_parts)

    except Exception:
        # 使用 logger.exception 将完整异常堆栈写入日志，便于开发排查
        logger.exception("处理问题失败")
        elapsed = time.time() - start_time
        # 面向用户的提示不暴露原始异常文本，避免技术细节泄露
        return f"⚠️ 系统暂时无法处理您的问题，请稍后重试。\n\n⏱️ 耗时: {elapsed:.2f}s"


def _flag_last_answer(history: list) -> str:
    """
    将最近一次回答标记为有误，加入异步审核队列

    Args:
        history: 对话历史

    Returns:
        提示信息
    """
    if not history or len(history) < 2:
        return "⚠️ 没有可标记的回答。"

    # 找到最后一个用户问题和助手回答
    last_user_msg = None
    last_assistant_msg = None
    for msg in reversed(history):
        if msg["role"] == "user" and last_user_msg is None:
            last_user_msg = msg["content"]
        elif msg["role"] == "assistant" and last_assistant_msg is None:
            last_assistant_msg = msg["content"]
        if last_user_msg and last_assistant_msg:
            break

    if not last_user_msg or not last_assistant_msg:
        return "⚠️ 无法找到最近的对话内容。"

    try:
        from review.review_queue import get_default_queue
        queue = get_default_queue()
        review_id = queue.enqueue_async(
            question=last_user_msg,
            answer=last_assistant_msg[:1000],
            trigger_reason="user_flag",
        )
        return f"✅ 已标记有误，审核编号: `{review_id}`。专家将尽快审核。"
    except Exception as e:
        logger.warning(f"标记有误失败: {e}")
        return f"⚠️ 标记失败: {e}"


def _save_session_summary_async(history: list, session_id: str):
    """
    后台线程：用 LLM 生成本次会话摘要并写入长期记忆（Mem0）

    在会话重置/结束时触发，异步执行不阻塞界面响应
    """
    import threading

    def _job():
        try:
            # 提取纯文本对话
            lines = []
            for msg in history:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = (msg.get("content") or "")[:500]
                if content:
                    lines.append(f"{role}: {content}")
            if not any(l.startswith("用户:") for l in lines):
                return  # 无真实提问，不生成摘要
            dialogue = "\n".join(lines)[:3000]

            from core.llm_client import LLMClient
            llm = LLMClient()
            summary = asyncio.run(llm.chat(
                messages=[
                    {"role": "system", "content": "你是医疗会话摘要助手。将对话浓缩为一段不超过 200 字的摘要，包含：主要症状/问题、关键结论、给出的建议。只输出摘要正文。"},
                    {"role": "user", "content": dialogue},
                ],
                max_tokens=300,
            ))
            _long_term_memory.add_session_summary(
                summary=summary.strip(),
                metadata={"agent_id": "medical_assistant"},
                session_id=session_id,
            )
            logger.info(f"会话摘要已写入长期记忆 (session={session_id})")
        except Exception as e:
            logger.warning(f"会话摘要保存失败: {e}")

    threading.Thread(target=_job, daemon=True).start()


def _reset_session(history: list):
    """重置会话（清理旧会话短期记忆，再把本次会话摘要写入长期记忆，然后清空对话并展示欢迎语）"""
    global _session_id

    # 旧会话的短期记忆随重置一并清除（redis 后端即删除 stm:* 键，避免会话键无限累积；
    # memory 后端同样释放对应条目），失败不影响重置主流程
    if _short_term_memory is not None:
        try:
            _short_term_memory.clear_session(_session_id)
        except Exception as e:
            logger.warning(f"清除旧会话短期记忆失败: {e}")

    # 会话结束：有实质对话内容时生成摘要存入长期记忆
    if _long_term_memory is not None and history and len(history) >= 3:
        _save_session_summary_async(list(history), _session_id)

    _session_id = str(uuid.uuid4())
    logger.info(f"会话已重置，新 session_id: {_session_id}")
    return [{"role": "assistant", "content": WELCOME_MESSAGE}], "🧹 会话已重置。"


def _set_sending(sending: bool):
    """
    切换发送状态：发送中禁用输入与按钮，防止重复提交

    Args:
        sending: 是否正在发送

    Returns:
        (user_input 更新, submit_btn 更新)
    """
    return (
        gr.update(interactive=not sending),
        gr.update(value=("思考中..." if sending else "发送"), interactive=not sending),
    )


def _set_sending_true():
    """进入发送状态（供事件链直接调用）"""
    return _set_sending(True)


def _set_sending_false():
    """退出发送状态（供事件链直接调用）"""
    return _set_sending(False)


# 快速提问选项（图标，问题文本）
QUICK_QUESTIONS = [
    ("🌡️", "发烧了怎么办？"),
    ("🧠", "经常头痛怎么缓解？"),
    ("❤️", "心悸是什么原因？"),
    ("💊", "布洛芬的副作用有哪些？"),
    ("🦠", "感冒了要吃药吗？"),
    ("🏃", "如何保持健康的生活方式？"),
]


def create_chat_tab():
    """创建对话 Tab"""
    # 头像资源路径（用户 / AI 助手）
    _user_avatar = str(_project_root / "web" / "assets" / "user_avatar.png")
    _bot_avatar = str(_project_root / "web" / "assets" / "bot_avatar.png")

    # 固定免责声明横幅（医疗合规：置于对话区上方，始终可见）
    gr.HTML(DISCLAIMER_BANNER_HTML)

    # 兼容 Gradio 4.x-6.x：不同版本 Chatbot 支持的参数不同（5.x+ 移除 type，6.x 移除
    # show_copy_button 等），按构造签名过滤，仅传入当前版本支持的参数
    import inspect
    _supported = set(inspect.signature(gr.Chatbot.__init__).parameters)
    _chatbot_kwargs = {
        "label": "对话",
        "height": 440,
        "type": "messages",            # 4.x 需要；5.x+ 自动忽略（不在签名中会被过滤）
        "show_copy_button": True,       # 4.x/5.x；6.x 移除后自动过滤
        "avatar_images": (_user_avatar, _bot_avatar),
        "value": [{"role": "assistant", "content": WELCOME_MESSAGE}],
        "elem_classes": "medix-chatbot",
    }
    chatbot = gr.Chatbot(**{k: v for k, v in _chatbot_kwargs.items() if k in _supported})

    # 快速提问卡片网格（2 行 3 列，点击填充输入框）
    gr.Markdown("##### ✨ 快速提问")
    quick_btns = []
    for row_start in (0, 3):
        with gr.Row():
            for i in range(row_start, row_start + 3):
                icon, question = QUICK_QUESTIONS[i]
                btn = gr.Button(
                    f"{icon} {question}",
                    elem_classes="medix-quick-btn",
                )
                quick_btns.append((btn, question))

    # 输入区
    with gr.Row(equal_height=True):
        user_input = gr.Textbox(
            placeholder="Ask your medical question... 请描述您的症状或健康问题",
            lines=2,
            scale=9,
            show_label=False,
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=100)

    # 操作区：重置会话 + 标记有误
    with gr.Row():
        reset_btn = gr.Button("🧹 重置会话", variant="secondary", scale=1, min_width=120)
        flag_btn = gr.Button("🚩 标记有误", variant="secondary", scale=1, min_width=120)
    action_status = gr.Markdown("")

    # 绑定快速提问：点击后填充到输入框
    for btn, question in quick_btns:
        btn.click(lambda q=question: q, outputs=[user_input])

    # 对话函数（流式输出：先显示思考提示，答案就绪后分段追加，形成打字机效果）
    async def respond(message: str, history: list):
        if not message.strip():
            return

        # 添加用户消息，并先放入"思考中"占位回复（核心处理约需 10-20 秒，避免界面长时间无反馈）
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "🤔 正在检索医学知识库并分析您的问题，请稍候..."},
        ]
        yield history

        # 处理问题（核心链路保持不变，一次性拿到完整回答）
        answer = await _process_question(message, history)

        # 流式分段输出：逐块累加更新最后一条助手消息，模拟打字机效果。
        # 固定步长切片 + 短间隔，既保证打字节奏感，又避免单块等待过久
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            history[-1]["content"] = answer[:i + chunk_size]
            yield history
            await asyncio.sleep(0.03)

        # 确保最终内容完整（防止切片边界导致末尾遗漏）
        history[-1]["content"] = answer
        yield history

    # 绑定事件：发送时禁用输入与按钮，完成后恢复并清空输入框
    user_input.submit(
        _set_sending_true, outputs=[user_input, submit_btn], queue=False
    ).then(
        respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot],
    ).then(lambda: "", outputs=[user_input], queue=False).then(
        _set_sending_false, outputs=[user_input, submit_btn], queue=False
    )

    submit_btn.click(
        _set_sending_true, outputs=[user_input, submit_btn], queue=False
    ).then(
        respond,
        inputs=[user_input, chatbot],
        outputs=[chatbot],
    ).then(lambda: "", outputs=[user_input], queue=False).then(
        _set_sending_false, outputs=[user_input, submit_btn], queue=False
    )

    # 操作按钮事件
    reset_btn.click(_reset_session, inputs=[chatbot], outputs=[chatbot, action_status])
    flag_btn.click(_flag_last_answer, inputs=[chatbot], outputs=[action_status])
