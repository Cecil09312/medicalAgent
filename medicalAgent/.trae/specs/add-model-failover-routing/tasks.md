# Tasks

- [x] Task 1: 新增配置项
  - [x] SubTask 1.1: `.env` 追加小模型与熔断/路由配置（LLM_SMALL_MODEL_NAME / LLM_SMALL_BASE_URL / LLM_SMALL_API_KEY / CIRCUIT_FAILURE_THRESHOLD / CIRCUIT_COOLDOWN_SECONDS / ROUTE_SIMPLE_MAX_LEN，默认值保持禁用态），不改动现有行，注释用中文且独立成行（不用行内注释）
  - [x] SubTask 1.2: `.env.example` 同步新增配置模板
- [x] Task 2: 新增熔断器 `core/circuit_breaker.py`
  - [x] SubTask 2.1: 实现 CircuitBreaker（CLOSED/OPEN/HALF_OPEN 三态、连续失败计数、冷却时间、asyncio.Lock 保护）
  - [x] SubTask 2.2: 模块级 `get_circuit_breaker(model_name)` 按主模型名缓存单例，跨 LLMClient 实例共享
- [x] Task 3: 新增路由模块 `core/model_router.py`
  - [x] SubTask 3.1: `is_simple_question(question)` 启发式判定（长度阈值 + 复杂关键词黑名单）
  - [x] SubTask 3.2: contextvar 路由开关（`set_route`/`get_route`，取值 auto/small/big，默认 auto）
- [x] Task 4: `core/llm_client.py` 接入降级与熔断
  - [x] SubTask 4.1: `__init__` 读取小模型配置，配置齐全时创建小模型客户端（BASE_URL/API_KEY 缺省复用主配置），未配置时行为与现状一致
  - [x] SubTask 4.2: `chat()` 与 `chat_with_tools()` 实际调用前按"路由 contextvar → 熔断状态"选择模型（OPEN 走小模型、HALF_OPEN 放行大模型探测、其余按路由/默认），并输出中文日志
  - [x] SubTask 4.3: 大模型 API 调用成功/失败实时计入熔断器（小模型调用不计入），成功重置连续失败计数，HALF_OPEN 探测失败重新 OPEN
  - [x] SubTask 4.4: 既有 `chat_with_retry` / `chat_with_tools_retry` 重试策略不动
- [x] Task 5: `swarm/swarm_coordinator.py` 接入简单问题路由
  - [x] SubTask 5.1: `process()` 入口调用 `is_simple_question` 判定，简单问题通过 set_route 将本次请求协程路由至小模型，结束恢复默认（复杂/默认路径不改）
- [x] Task 6: 验证
  - [x] SubTask 6.1: 配置 `LLM_SMALL_MODEL_NAME=qwen-turbo` 后验证：简单问题走小模型、复杂问题走大模型（日志可辨）
  - [x] SubTask 6.2: 临时置主 Key 无效验证熔断：连续 3 次失败后 OPEN，单次提问内后续调用改走小模型并返回真实回答；恢复主 Key 后冷却期满半开探测成功回到 CLOSED
  - [x] SubTask 6.3: 移除小模型配置验证回归：行为与现状完全一致
  - [x] SubTask 6.4: py_compile + `import main` 冒烟测试

# Task Dependencies
- Task 2、Task 3 相互独立，可并行
- Task 4 依赖 Task 2 与 Task 3
- Task 5 依赖 Task 3 与 Task 4（路由生效需 LLMClient 读取 contextvar）
- Task 6 依赖全部完成

# 实施说明
- Task 5 子代理的导入行编辑曾被环境回退（本会话已知的编辑丢失现象），验证阶段发现 NameError 后已手动补回 `from core.model_router import is_simple_question, reset_route, set_route`（L11），其余修改完整。
- 验证 6.1 首轮未生效的原因是验证环境未写入 LLM_SMALL_MODEL_NAME（.env 保持默认禁用态），恰好顺带完成了"未配置小模型时零行为变化"的回归观察；随后补配置后正式通过。
