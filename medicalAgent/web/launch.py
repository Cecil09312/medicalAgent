"""
MediX Web 前端启动脚本

运行方式:
    python web/launch.py

或从项目根目录:
    python -m web.launch
"""
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(_project_root / ".env")
except ImportError:
    pass


def main():
    """启动 Web 前端"""
    from web.app import create_app

    app = create_app()

    print("\n" + "=" * 50)
    print("  MediGenius 医疗智能助手 - Web 前端")
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


if __name__ == "__main__":
    main()
