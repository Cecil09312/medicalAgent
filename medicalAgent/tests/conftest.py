"""
pytest 全局配置：项目根目录入 sys.path + 安全的默认环境

单元/集成测试不依赖真实 LLM API 与外部服务（需要 LLM 的路径一律 mock）。
examples/test_all.py 仍保留为真实 API 的端到端冒烟脚本。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _safe_env(monkeypatch):
    """测试运行的默认环境：占位 API Key、确定性开关（个别测试自行覆盖）"""
    monkeypatch.setenv("LLM_API_KEY", "test-key-placeholder")
    monkeypatch.delenv("LLM_SMALL_MODEL_NAME", raising=False)
    monkeypatch.setenv("REVIEW_REALTIME_ENABLED", "true")
    monkeypatch.setenv("REVIEW_ASYNC_SAMPLE_RATE", "0.0")
    monkeypatch.delenv("ROUTE_LLM_CLASSIFY_ENABLED", raising=False)
