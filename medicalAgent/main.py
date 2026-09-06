"""
AI智能医疗诊断系统 - 主入口
交互式命令行界面
"""
import sys
import os
import asyncio
import uuid
import time
import argparse
from pathlib import Path
from loguru import logger

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass


def _setup_hf_offline_early():
    """
    早期设置 HuggingFace 离线模式（必须在 gradio / transformers 导入 huggingface_hub 之前执行）

    原因：huggingface_hub 仅在首次导入时读取一次 HF_HUB_OFFLINE / HF_ENDPOINT 环境变量
    并固化为模块常量。Web 模式下 web.app 第一行 import gradio 会先行导入 huggingface_hub，
    若到 milvus_kb / reranker 模块中才设置这些环境变量则不会生效，导致模型加载时
    访问 huggingface.co 产生长时间网络重试（约 5 分钟）后失败。

    策略：仅当所有待加载模型（嵌入模型必加载；重排模型仅 RERANK_ENABLED 开启时加载）
    均已缓存到本地时才启用离线模式，避免阻止模型的首次下载。
    """
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"

    def _is_cached(model_name: str) -> bool:
        # 与 milvus_kb._detect_model_cached 一致：检查 hub 缓存目录是否存在
        if not model_name:
            return False
        return (hub_dir / f"models--{model_name.replace('/', '--')}").exists()

    # 汇总本进程将要加载的全部模型
    needed_models = [os.getenv("KB_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")]
    rerank_enabled = os.getenv("RERANK_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
    if rerank_enabled:
        needed_models.append(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"))

    # 所需模型全部已缓存时启用离线模式，避免加载时访问 huggingface.co 检查更新
    if all(_is_cached(name) for name in needed_models):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # 下载未缓存模型时使用国内镜像（与 reranker.py 的意图一致，setdefault 不覆盖用户配置）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


_setup_hf_offline_early()


def setup_logger(verbose: bool = False):
    """
    配置日志系统

    Args:
        verbose: 是否启用详细日志
    """
    # 移除默认处理器
    logger.remove()

    # 添加控制台输出
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # 添加文件日志
    logger.add(
        "logs/medical_agent.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )

    logger.debug(f"Logger configured (verbose={verbose})")


async def interactive_mode():
    """
    交互式模式
    用户可以输入问题，系统返回答案
    """
    # 打印欢迎横幅
    print("\n" + "=" * 60)
    print("🏥  AI智能医疗诊断系统")
    print("=" * 60)
    print()
    print("💡 输入您的医疗问题，我将为您提供专业的信息和建议")
    print("📋 输入 'help' 查看帮助 | 输入 'exit' 或 'quit' 退出")
    print("-" * 60)
    print()

    # 生成会话ID
    session_id = str(uuid.uuid4())
    logger.info(f"New session started: {session_id}")

    # 尝试导入 process_with_swarm（如果可用）
    try:
        from swarm import process_with_swarm
        swarm_available = True
    except ImportError:
        logger.warning("Swarm module not available, using basic mode")
        swarm_available = False

    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n🧑 您: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() in ("exit", "quit", "q", "退出"):
                print("\n👋 感谢使用AI智能医疗诊断系统，再见！")
                break

            if user_input.lower() in ("clear", "清除"):
                print("\n" * 20)
                print("🧹 屏幕已清除")
                continue

            if user_input.lower() in ("help", "帮助"):
                print("\n📖 帮助信息：")
                print("  • 直接输入医疗问题进行咨询")
                print("  • 'exit' / 'quit' - 退出程序")
                print("  • 'clear' - 清除屏幕")
                print("  • 'help' - 显示此帮助信息")
                continue

            # 处理问题
            start_time = time.time()

            if swarm_available:
                try:
                    result = await process_with_swarm(user_input, session_id=session_id)

                    elapsed = time.time() - start_time

                    # 显示结果
                    print("\n" + "-" * 60)
                    print(f"🤖 AI智能医疗诊断系统 (Swarm模式 | 耗时: {elapsed:.2f}s)")
                    print("-" * 60)

                    if isinstance(result, dict):
                        answer = result.get("answer", "无回答")
                        suggestions = result.get("suggestions", [])

                        print(f"\n{answer}")

                        if suggestions:
                            print("\n💡 相关建议：")
                            for i, s in enumerate(suggestions, 1):
                                print(f"  {i}. {s}")
                    else:
                        print(f"\n{result}")

                except Exception as e:
                    logger.error(f"Swarm processing failed: {e}")
                    print(f"\n⚠️  处理失败: {e}")
                    print("💡 建议：请尝试简化问题或检查系统配置")

            else:
                # 基础模式：直接使用 LLM
                try:
                    from core.llm_client import LLMClient

                    client = LLMClient()
                    messages = [
                        {
                            "role": "system",
                            "content": "你是一个专业的医疗助手。请根据用户的问题提供准确、有用的医疗信息。注意：你的建议不能替代专业医生的诊断。",
                        },
                        {"role": "user", "content": user_input},
                    ]

                    response = await client.chat(messages)
                    elapsed = time.time() - start_time

                    print("\n" + "-" * 60)
                    print(f"🤖 AI智能医疗诊断系统 (基础模式 | 耗时: {elapsed:.2f}s)")
                    print("-" * 60)
                    print(f"\n{response}")

                except Exception as e:
                    logger.error(f"Basic processing failed: {e}")
                    print(f"\n⚠️  处理失败: {e}")

            # 免责声明
            print("\n⚕️  免责声明：以上信息仅供参考，不能替代专业医疗建议。")
            print("   如有健康问题，请咨询专业医生。")

        except KeyboardInterrupt:
            print("\n\n👋 检测到中断，正在退出...")
            break
        except EOFError:
            print("\n\n👋 输入结束，再见！")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n❌ 发生错误: {e}")


def launch_web():
    """启动 Web 前端"""
    from web.app import create_app

    app = create_app()

    print("\n" + "=" * 50)
    print("  AI智能医疗诊断系统 Web 前端")
    print("=" * 50)
    print("  启动中...")
    print("  地址: http://0.0.0.0:7860")
    print("  按 Ctrl+C 停止")
    print("=" * 50 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


def main():
    """
    主函数
    解析命令行参数并启动程序
    """
    parser = argparse.ArgumentParser(
        description="MediX 医疗Agent - 智能医疗助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py              # 启动 Web 前端 (默认)
  python main.py --cli        # 启动命令行交互模式
  python main.py -v           # 启用详细日志
        """,
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="启动命令行交互模式（默认启动 Web 前端）",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="启用详细日志输出",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logger(verbose=args.verbose)

    logger.info("Starting AI智能医疗诊断系统")

    if args.cli:
        # 运行命令行交互模式
        try:
            asyncio.run(interactive_mode())
        except KeyboardInterrupt:
            print("\n👋 再见！")
    else:
        # 运行 Web 前端
        try:
            launch_web()
        except KeyboardInterrupt:
            print("\n👋 再见！")


if __name__ == "__main__":
    main()
