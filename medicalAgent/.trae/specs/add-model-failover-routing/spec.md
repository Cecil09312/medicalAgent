# 大模型降级、熔断器与大小模型路由 Spec

## Why
当前系统仅配置单一模型（qwen-plus），已有的容错能力只有"指数退避重试 + 静态致歉兜底"：大模型服务完全不可用时用户只能收到致歉文案，且所有请求（包括极简单问题）都消耗大模型配额。用户要求补齐模型层容错体系：正常业务走大模型、简单问题路由小模型（省钱提速）、大模型连续失败熔断后自动降级到小模型、冷却后半开探测恢复。

## What Changes
- `.env` / `.env.example` 新增小模型与熔断/路由配置项（小模型未配置时全部新能力自动关闭，行为与现状完全一致）
- 新增 `core/circuit_breaker.py`：CLOSED/OPEN/HALF_OPEN 三态熔断器，按主模型名进程级单例（LLMClient 为每 Agent 独立实例化，共 7 处，必须共享计数）
- 新增 `core/model_router.py`：简单问题启发式判定（长度阈值 + 复杂关键词黑名单）+ 基于 contextvar 的请求级路由开关
- `core/llm_client.py` 接入：配置了小模型时创建第二组 client；底层 `chat()`/`chat_with_tools()` 按路由/熔断状态选择模型；大模型 API 调用成功/失败实时计入熔断器（小模型调用不计入）；既有 `chat_with_retry`/`chat_with_tools_retry` 重试逻辑不动
- `swarm/swarm_coordinator.py` `process()` 入口：简单问题将本次请求协程路由到小模型（子任务 gather 继承 context），复杂问题与默认路径走大模型
- 不改动 agents/、skills/、memory/ 等调用方代码（路由对下游透明）

**BREAKING**: 无。未配置小模型时行为与现状完全一致。

## Impact
- Affected specs: 与第 6 章（重试）、第 7 章（兜底）互补，不修改其已实施行为
- Affected code:
  - `.env` / `.env.example`（新增配置项，不动现有行）
  - `core/circuit_breaker.py`（新增）
  - `core/model_router.py`（新增）
  - `core/llm_client.py`（__init__ + chat + chat_with_tools 接入）
  - `swarm/swarm_coordinator.py`（process 入口路由）

## 约束（实现时必须遵守）
- 未配置小模型时零行为变化；既有重试、兜底、审核、指标等功能不受影响
- 注释一律中文，UTF-8，代码不出现 emoji；不编写测试脚本
- 熔断器必须跨 LLMClient 实例共享（模块级单例注册表，按主模型名缓存）
- contextvar 方案避免给深层调用链加参数（agents→agent_loop→llm_client 链路不改动签名）

---

## ADDED Requirements

### Requirement: 大小模型配置与降级客户端
LLMClient SHALL 支持可选的第二模型配置（`LLM_SMALL_MODEL_NAME`/`LLM_SMALL_BASE_URL`/`LLM_SMALL_API_KEY`）：小模型模型名未配置时 SHALL NOT 创建小模型客户端且所有请求固定使用大模型；配置后 SHALL 创建独立的小模型客户端（BASE_URL/API_KEY 缺省分别复用主配置）。

#### Scenario: 未配置小模型时零行为变化
- **WHEN** `.env` 未设置 `LLM_SMALL_MODEL_NAME`
- **THEN** 不创建小模型客户端，熔断与路由不生效，所有请求行为与现状一致

#### Scenario: 配置小模型后降级可用
- **WHEN** `.env` 设置 `LLM_SMALL_MODEL_NAME=qwen-turbo`（同 BASE_URL 与 API_KEY）
- **THEN** LLMClient 同时持有大/小模型两组调用参数

### Requirement: 熔断器（CLOSED/OPEN/HALF_OPEN）
系统 SHALL 提供按主模型名进程级单例的熔断器：大模型 API 调用连续失败达到阈值（默认 3 次，`CIRCUIT_FAILURE_THRESHOLD`）时进入 OPEN，冷却期内（默认 60 秒，`CIRCUIT_COOLDOWN_SECONDS`）所有请求改走小模型；冷却到期后进入 HALF_OPEN 并放行一次大模型探测，探测成功回到 CLOSED，探测失败重新 OPEN 并重置冷却。大模型调用成功 SHALL 重置连续失败计数；小模型调用的成败 SHALL NOT 影响熔断器状态。

#### Scenario: 大模型连续失败触发熔断
- **WHEN** 大模型连续 3 次 API 调用失败（如 Key 失效）
- **THEN** 熔断器进入 OPEN，后续请求（含复杂问题与各 Agent 调用）直接改走小模型，用户仍能获得真实回答

#### Scenario: 冷却后半开探测恢复
- **WHEN** 熔断 OPEN 超过冷却时间后有大模型探测请求
- **THEN** 放行一次大模型调用：成功则恢复 CLOSED（后续走大模型），失败则重新 OPEN 并重置冷却

#### Scenario: 瞬时抖动不误熔断
- **WHEN** 大模型偶发失败 1-2 次后恢复成功
- **THEN** 连续失败计数被成功调用重置，熔断器保持 CLOSED

### Requirement: 简单问题路由小模型
系统 SHALL 提供简单问题启发式判定（去除首尾空白后长度小于阈值 `ROUTE_SIMPLE_MAX_LEN` 默认 30，且不含内置复杂关键词如：研究、分析、比较、鉴别、原理、为什么、机制、方案、综述、设计 等），`SwarmCoordinator.process()` 入口对判定为简单的问题 SHALL 将该请求协程整体路由至小模型（通过 contextvar 传递，Agent/Skill 深层调用透明生效），复杂问题与默认路径 SHALL 使用大模型。路由仅影响"选择哪个模型"，SHALL NOT 改变既有 Agent 编排、审核与记忆流程。

#### Scenario: 简单问题走小模型
- **WHEN** 用户提问"感冒了怎么办"（短且无复杂关键词）
- **THEN** 本次请求全链路 LLM 调用使用小模型，回答正常生成并完成既有审核采样等流程

#### Scenario: 复杂问题走大模型
- **WHEN** 用户提问包含复杂关键词或超长的问题
- **THEN** 请求使用大模型，行为与现状一致

### Requirement: 熔断与重试协同
熔断计数 SHALL 在底层 `chat()`/`chat_with_tools()` 的 API 调用结果处实时记录（仅针对大模型调用）；既有 `chat_with_retry`/`chat_with_tools_retry` 的重试策略、指数退避与次数 SHALL 保持不变，重试的每次底层尝试 SHALL 依次计入熔断器（因此在同一次提问内即可达到阈值触发熔断，后续重试尝试自动改走小模型）。

#### Scenario: 单次提问内触发熔断并降级成功
- **WHEN** 大模型 Key 失效且小模型可用，用户提问
- **THEN** 首轮 3 次失败尝试触发 OPEN，后续重试尝试与兜底路径的调用自动改走小模型，最终返回由小模型生成的真实回答（而非静态致歉）

### Requirement: 路由与熔断可观测
模型选择与熔断状态变化 SHALL 输出中文日志（记录使用的模型名与熔断状态迁移），便于验证与排查；日志频率不影响请求延迟。

#### Scenario: 模型选择可验证
- **WHEN** 任一 LLM 调用发生
- **THEN** 日志中可辨认本次使用的模型名及触发原因（简单路由/熔断降级/默认）
