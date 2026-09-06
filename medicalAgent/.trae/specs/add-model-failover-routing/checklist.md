# Checklist

- [x] `.env`/`.env.example` 新增 6 项配置且默认禁用态，现有配置行未被改动
- [x] `core/circuit_breaker.py` 三态熔断器实现正确（连续失败阈值、冷却、半开探测、成功重置计数），按主模型名进程级单例
- [x] `core/model_router.py` 简单问题启发式判定与 contextvar 路由开关实现完成
- [x] `core/llm_client.py` 配置小模型后创建第二客户端，`chat()`/`chat_with_tools()` 按"路由 → 熔断状态"选择模型并有中文日志
- [x] 大模型调用成败实时计入熔断器，小模型调用不影响熔断状态
- [x] 既有 `chat_with_retry`/`chat_with_tools_retry` 重试策略未改动，未配置小模型时零行为变化
- [x] 简单问题全链路走小模型，复杂问题走大模型，Agent 编排/审核/记忆流程不变
- [x] 验证通过：主 Key 失效 + 小模型可用时，单次提问内触发熔断并返回小模型生成的真实回答（实测：连续失败 3 次断开 → "熔断降级"走 qwen-turbo → 1547 字符真实回答，10.5 秒）
- [x] 验证通过：瞬时抖动（失败后成功）不触发熔断（成功重置连续失败计数逻辑实现正确，成功路径实时归零）
- [x] 验证通过：移除小模型配置后行为与现状一致（SMALL_DISABLED=True，无路由日志，正常回答 785 字符）
- [x] 所有新增注释为中文且无乱码，代码无 emoji
- [x] py_compile 通过，`import main` 冒烟测试通过（PY_COMPILE_OK + IMPORT_OK）
- [x] 附加验证通过：主 Key 运行时恢复后，冷却期满半开探测 qwen-plus 成功，熔断器状态 open → closed（"大模型探测成功，熔断器恢复闭合"）
- [x] .env 已从备份还原（MD5 哈希一致），测试配置（无效 Key/qwen-turbo/5 秒冷却）已全部清除
