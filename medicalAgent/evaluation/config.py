"""
评估模块配置
定义各维度的指标阈值和 LLM Judge 配置
"""
import sys
import os
from pathlib import Path

# 加载 .env
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# LLM Judge 配置：优先使用 EVAL_LLM_* 环境变量，否则复用 LLM_* 配置
_LLM_JUDGE_MODEL = os.getenv("EVAL_LLM_MODEL_NAME") or os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
_LLM_JUDGE_TEMP = float(os.getenv("EVAL_LLM_TEMPERATURE", "0.1"))


def _setup_deepeval_llm_env():
    """
    把项目 LLM 配置映射为 deepeval 所需的 OpenAI 兼容环境变量（不覆盖已有设置）

    deepeval 内置指标（FaithfulnessMetric 等）默认通过 OpenAI SDK 调用 judge 模型，
    需要 OPENAI_API_KEY；该 SDK 同时支持 OPENAI_BASE_URL / OPENAI_MODEL_NAME 环境变量。
    不映射时 judge 调用会因缺少 OPENAI_API_KEY 而失败，导致所有指标得 0 分。

    优先级：用户显式设置的 OPENAI_* > EVAL_LLM_MODEL_NAME > LLM_*（复用主模型配置）
    """
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    judge_model = os.getenv("EVAL_LLM_MODEL_NAME") or os.getenv("LLM_MODEL_NAME")

    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
    if judge_model:
        os.environ.setdefault("OPENAI_MODEL_NAME", judge_model)

    # 单次指标评分超时（秒）：默认 180s 在 DashScope 偶发慢响应/同步串行评分时易超时，
    # 导致整批评分失败、指标全 0 分；调大为 600s 提升容错（不覆盖用户显式设置）
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")


_setup_deepeval_llm_env()

# 知识库配置（供评估模块使用）
KNOWLEDGE_CONFIG = {
    "top_k": int(os.getenv("KB_TOP_K", "5")),
    "chunk_size": int(os.getenv("KB_CHUNK_SIZE", "500")),
}

# 评估配置
EVAL_CONFIG = {
    "rag": {
        "min_faithfulness": 0.7,
        "min_relevancy": 0.7,
        "min_context_precision": 0.6,
        "min_context_recall": 0.6,
        "min_no_hallucination": 0.8,
    },
    "agent": {
        "min_tool_accuracy": 0.8,
        "min_answer_correctness": 0.6,
        "min_task_completion": 0.7,
        "min_response_quality": 0.7,
    },
    "swarm": {
        "min_decomposition_quality": 0.7,
        "min_agent_selection_accuracy": 0.8,
        "max_coordination_time": 120,
        "min_synthesis_quality": 0.7,
    },
    "performance": {
        "max_latency_p50": 15.0,
        "max_latency_p95": 30.0,
        "max_latency_p99": 60.0,
        "max_tokens_per_query": 5000,
        "min_throughput_qpm": 2,
    },
    "llm_judge": {
        "model": _LLM_JUDGE_MODEL,
        "temperature": _LLM_JUDGE_TEMP,
    },
}

# 评估数据目录
_EVAL_DIR = Path(__file__).parent
DATA_DIR = _EVAL_DIR / "data"
