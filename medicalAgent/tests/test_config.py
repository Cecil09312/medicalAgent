"""配置统一校验单元测试：必填项、格式、逻辑冲突与脱敏摘要"""
from core.config import build_config_summary, validate_config


def _clean_env(monkeypatch):
    """清空相关环境变量，逐项构造测试场景（防止宿主机全局环境变量渗透断言）"""
    for name in (
        "LLM_API_KEY", "LLM_MODEL_NAME", "LLM_BASE_URL",
        "LLM_SMALL_MODEL_NAME", "LLM_SMALL_BASE_URL", "LLM_SMALL_API_KEY",
        "ROUTE_LLM_CLASSIFY_ENABLED",
        "ROUTE_LLM_BOUNDARY_MIN", "ROUTE_LLM_BOUNDARY_MAX",
        "CIRCUIT_FAILURE_THRESHOLD", "CIRCUIT_COOLDOWN_SECONDS",
        "RERANK_ENABLED", "RERANK_CANDIDATE_K",
        "REVIEW_REALTIME_ENABLED",
        "MEMORY_BACKEND", "STM_SESSION_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_valid_config_passes(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-valid-key-123")
    assert validate_config() == []


def test_missing_api_key_fails(monkeypatch):
    _clean_env(monkeypatch)
    errors = validate_config()
    assert any("LLM_API_KEY 未配置" in e for e in errors)


def test_invalid_integer_fails(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("CIRCUIT_COOLDOWN_SECONDS", "abc")
    errors = validate_config()
    assert any("CIRCUIT_COOLDOWN_SECONDS" in e and "不是有效整数" in e for e in errors)


def test_invalid_bool_fails(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("REVIEW_REALTIME_ENABLED", "maybe")
    errors = validate_config()
    assert any("REVIEW_REALTIME_ENABLED" in e and "不是有效布尔值" in e for e in errors)


def test_classify_without_small_model_conflict(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("ROUTE_LLM_CLASSIFY_ENABLED", "true")
    errors = validate_config()
    assert any("LLM_SMALL_MODEL_NAME 为空" in e for e in errors)


def test_classify_with_small_model_ok(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("ROUTE_LLM_CLASSIFY_ENABLED", "true")
    monkeypatch.setenv("LLM_SMALL_MODEL_NAME", "qwen-turbo")
    assert not any("LLM_SMALL_MODEL_NAME" in e for e in validate_config())


def test_boundary_min_max_conflict(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("ROUTE_LLM_BOUNDARY_MIN", "60")
    monkeypatch.setenv("ROUTE_LLM_BOUNDARY_MAX", "15")
    errors = validate_config()
    assert any("必须小于" in e for e in errors)


def test_invalid_memory_backend_fails(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("MEMORY_BACKEND", "mysql")
    errors = validate_config()
    assert any("MEMORY_BACKEND" in e and "memory/redis" in e for e in errors)


def test_negative_stm_ttl_fails(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("MEMORY_BACKEND", "redis")
    monkeypatch.setenv("STM_SESSION_TTL_SECONDS", "-5")
    errors = validate_config()
    assert any("STM_SESSION_TTL_SECONDS" in e for e in errors)


def test_summary_masks_secrets(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-1234567890abcdef")
    summary = build_config_summary()
    assert "sk-1234567890abcdef" not in summary
    assert "sk-1***" in summary
    # 未配置的小模型 Key 显示占位符而非掩码
    assert "(复用主 Key)" in summary
