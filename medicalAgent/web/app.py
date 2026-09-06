"""
Gradio 主应用
组装 3 个 Tab：用户对话、专家审核、飞轮指标
深色科技感风格：深蓝黑渐变背景 + 紫粉光晕 + 渐变霓虹标题
"""
import gradio as gr

from .chat_tab import create_chat_tab
from .review_tab import create_review_tab
from .metrics_tab import create_metrics_tab


# 强制暗色模式（Gradio 通过 URL 参数 __theme=dark 切换整套暗色组件）
FORCE_DARK_JS = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

CUSTOM_CSS = """
/* 隐藏 Gradio 自带 footer */
footer {display: none !important;}

/* 深色科技感背景：深蓝黑渐变 + 紫粉光晕 */
body, gradio-app {
    background:
        radial-gradient(ellipse 800px 500px at 85% 12%, rgba(167, 139, 250, 0.12), transparent 60%),
        radial-gradient(ellipse 600px 400px at 8% 88%, rgba(244, 114, 182, 0.08), transparent 60%),
        linear-gradient(135deg, #0A0E17 0%, #1A1F2D 55%, #161226 100%) !important;
}

/* 容器透明以透出深色渐变 */
.gradio-container {max-width: 1180px !important; width: 100% !important; margin: 0 auto !important; padding: 18px !important; background: transparent !important;}

/* Tab 内容区统一撑满宽度，避免对话 Tab 收缩 */
.gradio-container [role="tabpanel"], .gradio-container .tabitem {width: 100% !important; max-width: 100% !important; flex: 1 1 0% !important;}

/* 标题区：玻璃拟态卡片，文字居中 */
#medix-header {
    background: rgba(30, 36, 48, 0.55) !important;
    border: 1px solid rgba(139, 124, 240, 0.25) !important;
    border-radius: 18px !important;
    padding: 26px 30px !important;
    margin-bottom: 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
}
/* 渐变霓虹标题：蓝→紫→粉 */
#medix-header h1 {
    background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
}
#medix-header p {color: #9CA3AF !important; font-size: 0.95rem !important; margin: 0 !important;}
#medix-header h1, #medix-header p {text-align: center !important;}

/* Tab 导航 */
.gradio-container .tab-nav {border-bottom: 1px solid rgba(139, 124, 240, 0.2) !important; margin-bottom: 18px !important; gap: 4px !important;}
.gradio-container .tab-nav button {font-size: 0.98rem !important; font-weight: 500 !important; padding: 10px 18px !important; border-radius: 8px 8px 0 0 !important;}

/* 深色玻璃卡片 */
.medix-card {
    background: rgba(30, 36, 48, 0.75) !important;
    border: 1px solid rgba(139, 124, 240, 0.18) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    margin-bottom: 12px !important;
    color: #D1D5DB !important;
}

/* 快速提问卡片：深色 + 悬浮发光 */
.medix-quick-btn {
    background: rgba(30, 36, 48, 0.85) !important;
    border: 1px solid rgba(139, 124, 240, 0.25) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    color: #E5E7EB !important;
    transition: all 0.2s ease !important;
}
.medix-quick-btn:hover {
    border-color: rgba(167, 139, 250, 0.6) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 124, 240, 0.25) !important;
}

/* 飞轮指标卡片：表格与文字居中 */
.medix-center table {margin: 0 auto !important;}
.medix-center th, .medix-center td {text-align: center !important;}
.medix-center p, .medix-center li {text-align: center !important;}

/* 按钮统一最小高度，提升可点击性 */
.gradio-container button {min-height: 42px !important;}
.gradio-container .tab-nav button {min-height: 0 !important;}

/* 小节标题 */
.gradio-container h5 {color: #A78BFA !important; font-weight: 600 !important; letter-spacing: 0.02em;}

/* 免责声明 */
#medix-disclaimer {
    text-align: center !important;
    color: #6B7280 !important;
    font-size: 0.82rem !important;
    padding: 14px 0 4px !important;
    margin-top: 18px !important;
    border-top: 1px solid rgba(139, 124, 240, 0.15) !important;
}

/* ============ 对话气泡美化 ============ */
/* 消息行间距与呼吸感 */
.medix-chatbot .message-row {padding: 6px 10px !important;}

/* 头像：圆形 + 紫色发光描边 */
.medix-chatbot .avatar-container img,
.medix-chatbot .avatar-container div {
    border-radius: 50% !important;
    border: 2px solid rgba(167, 139, 250, 0.55) !important;
    box-shadow: 0 0 12px rgba(139, 124, 240, 0.35) !important;
}

/* 用户气泡：蓝紫渐变，右侧圆角收紧 */
.medix-chatbot .message.user {
    background: linear-gradient(135deg, rgba(96, 165, 250, 0.30), rgba(139, 124, 240, 0.34)) !important;
    border: 1px solid rgba(139, 124, 240, 0.45) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 12px 16px !important;
    box-shadow: 0 4px 14px rgba(96, 165, 250, 0.15) !important;
}

/* AI 气泡：深色玻璃拟态，左侧圆角收紧 */
.medix-chatbot .message.bot {
    background: rgba(30, 36, 48, 0.82) !important;
    border: 1px solid rgba(139, 124, 240, 0.22) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 12px 16px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.28) !important;
}

/* 气泡内文字行距优化 */
.medix-chatbot .message {line-height: 1.7 !important;}
"""

# 自定义主题：紫色系
CUSTOM_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)


def create_app() -> gr.Blocks:
    """
    创建 Gradio 应用

    Returns:
        gr.Blocks 应用实例
    """
    # 启动时初始化 Swarm 系统并触发知识库后台预热（嵌入模型加载约数十秒），
    # 避免首个提问才懒加载导致长时间等待
    from .chat_tab import _get_coordinator
    _get_coordinator()

    with gr.Blocks(
        title="AI智能医疗诊断系统",
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
        js=FORCE_DARK_JS,
    ) as app:
        with gr.Column(elem_id="medix-header"):
            gr.Markdown(
                """
# 🏥 AI智能医疗诊断系统
**多智能协作，集咨询、诊断、研究为一体的医疗助手**
"""
            )

        with gr.Tabs():
            with gr.TabItem("💬 对话"):
                create_chat_tab()

            with gr.TabItem("🔍 专家审核"):
                create_review_tab()

            with gr.TabItem("📊 飞轮指标"):
                create_metrics_tab()

        gr.Markdown(
            "*AI智能医疗诊断系统 — 以上信息仅供参考，不能替代专业医疗建议。*",
            elem_id="medix-disclaimer",
        )

    return app
