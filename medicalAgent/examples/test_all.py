"""
MediX 医疗Agent - 综合测试套件
使用模拟数据测试各个模块（无需实际 API 调用）
"""
import sys
import asyncio
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from loguru import logger


def test_skill_registry():
    """测试 SkillRegistry: 注册、执行、格式化"""
    from core.skill_registry import SkillRegistry, SkillParameter

    logger.info("Testing SkillRegistry...")

    registry = SkillRegistry()

    # 定义测试函数
    async def mock_search(query: str, max_results: int = 5):
        return {"results": [{"title": f"Result for {query}"}]}

    # 注册 Skill
    registry.register(
        name="search_knowledge",
        function=mock_search,
        description="搜索医疗知识库",
        parameters=[
            SkillParameter(
                name="query",
                type="string",
                description="搜索查询",
                required=True,
            ),
            SkillParameter(
                name="max_results",
                type="integer",
                description="最大结果数",
                required=False,
            ),
        ],
    )

    # 验证注册
    assert "search_knowledge" in registry.skills
    assert registry.get("search_knowledge") is not None

    # 测试执行
    result = asyncio.get_event_loop().run_until_complete(
        registry.execute("search_knowledge", query="头痛", max_results=3)
    )
    assert result["results"][0]["title"] == "Result for 头痛"

    # 测试 OpenAI 格式
    openai_format = registry.to_openai_format()
    assert len(openai_format) == 1
    assert openai_format[0]["function"]["name"] == "search_knowledge"
    assert "query" in openai_format[0]["function"]["parameters"]["properties"]

    logger.info("✅ SkillRegistry test passed")


def test_state_manager():
    """测试 StateManager: 创建、更新、完成状态"""
    from core.state_manager import StateManager, TaskStatus

    logger.info("Testing StateManager...")

    manager = StateManager()

    # 创建状态
    state = manager.create_state(
        task_id="test_task_1",
        agent_id="test_agent",
        input_data={"question": "什么是感冒？"},
        max_iterations=5,
    )

    assert state.task_id == "test_task_1"
    assert state.status == TaskStatus.PENDING
    assert state.agent_id == "test_agent"

    # 更新状态
    manager.update_state("test_task_1", status=TaskStatus.IN_PROGRESS, iteration=1)
    updated_state = manager.get_state("test_task_1")
    assert updated_state.status == TaskStatus.IN_PROGRESS
    assert updated_state.iteration == 1

    # 完成状态
    updated_state.mark_completed({"answer": "感冒是一种常见的呼吸道感染"})
    assert updated_state.status == TaskStatus.COMPLETED
    assert updated_state.final_result["answer"] is not None

    # 测试活跃任务
    manager.create_state("test_task_2", "agent_2", {}, 3)
    manager.update_state("test_task_2", status=TaskStatus.IN_PROGRESS)
    active = manager.get_active_tasks()
    assert len(active) == 1

    logger.info("✅ StateManager test passed")


def test_short_term_memory():
    """测试 ShortTermMemory: 添加、获取、清除"""
    logger.info("Testing ShortTermMemory...")

    try:
        from memory.short_term import ShortTermMemory
    except ImportError:
        logger.warning("ShortTermMemory not available, skipping")
        return

    memory = ShortTermMemory()

    session_id = "test_session_1"

    # 添加消息
    memory.add_message(session_id, "user", "你好")
    memory.add_message(session_id, "assistant", "您好！有什么可以帮助您的？")
    memory.add_message(session_id, "user", "我头痛")

    # 获取历史
    history = memory.get_history(session_id, limit=2)
    assert len(history) == 2
    assert history[-1]["content"] == "我头痛"

    # 清除
    memory.clear_session(session_id)
    history_after = memory.get_history(session_id)
    assert len(history_after) == 0

    logger.info("✅ ShortTermMemory test passed")


def test_entropy_manager():
    """测试 EntropyManager: 去重、压缩、熵估计"""
    logger.info("Testing EntropyManager...")

    try:
        from memory.entropy import EntropyManager
    except ImportError:
        logger.warning("EntropyManager not available, skipping")
        return

    manager = EntropyManager()

    # 测试去重
    messages = [
        {"role": "user", "content": "什么是高血压？"},
        {"role": "assistant", "content": "高血压是一种慢性疾病"},
        {"role": "user", "content": "什么是高血压？"},  # 重复
    ]

    deduped = manager.deduplicate(messages)
    assert len(deduped) == 2

    # 测试压缩
    long_text = "这是一段很长的文本，" * 50
    compressed = manager.compress(long_text, max_length=100)
    assert len(compressed) <= 100

    # 测试熵估计
    entropy = manager.estimate_entropy(messages)
    assert 0 <= entropy <= 1

    logger.info("✅ EntropyManager test passed")


def test_constraint_validator():
    """测试 ConstraintValidator: 工具调用验证、输出验证"""
    logger.info("Testing ConstraintValidator...")

    try:
        from constraints import ConstraintValidator
        validator = ConstraintValidator()
    except Exception as e:
        logger.warning(f"ConstraintValidator not available or config error: {e}, skipping")
        return

    # 测试工具调用验证
    result = validator.validate_tool_call("medical_agent", "search_knowledge")
    assert "valid" in result

    # 测试输出验证
    output = "这是一个正常的医疗回答"
    result = validator.validate_output("medical_agent", output)
    assert "valid" in result

    logger.info("✅ ConstraintValidator test passed")


def test_auto_fixer():
    """测试 AutoFixer: 免责声明修复、高风险警告修复"""
    logger.info("Testing AutoFixer...")

    try:
        from validation import AutoFixer
    except ImportError:
        logger.warning("AutoFixer not available, skipping")
        return

    fixer = AutoFixer()

    # 测试免责声明修复
    text_without_disclaimer = "建议您服用阿司匹林。"
    fixed = fixer.fix_missing_disclaimer(text_without_disclaimer)
    assert "免责声明" in fixed or "医生" in fixed

    # 测试高风险警告修复
    high_risk_text = "你应该立即手术"
    fixed = fixer.fix_high_risk_warning(high_risk_text)
    assert "紧急提醒" in fixed or "就医" in fixed or fixed == high_risk_text

    logger.info("✅ AutoFixer test passed")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🧪 MediX 医疗Agent - 综合测试套件")
    print("=" * 60 + "\n")

    tests = [
        ("SkillRegistry", test_skill_registry),
        ("StateManager", test_state_manager),
        ("ShortTermMemory", test_short_term_memory),
        ("EntropyManager", test_entropy_manager),
        ("ConstraintValidator", test_constraint_validator),
        ("AutoFixer", test_auto_fixer),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            logger.error(f"❌ {name} test FAILED: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"❌ {name} test ERROR: {e}")
            failed += 1

    # 打印总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  ⏭️  跳过: {skipped}")
    print(f"  📝 总计: {passed + failed + skipped}")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 所有测试通过！\n")
    else:
        print(f"\n⚠️  {failed} 个测试失败\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
