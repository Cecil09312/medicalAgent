# 核心链路代码优化 Spec

## Why
对代码库进行全面审查后，发现 5 个有实际价值的优化点：2 个正确性 Bug（审核队列单例被违反 + 缺失导入导致异步审核静默失效；深度研究使用随机向量检索知识库）、1 个可靠性缺陷（LLM 工具调用路径无重试）、1 个数据安全风险（JSON 持久化非原子写入，崩溃时会损坏审核队列与飞轮指标文件）、1 个重复加载问题（约束 YAML 每次提问重复读取、正则每次重复编译）。

## What Changes
- 修复 `core/agent_loop.py` 中实时审核直接实例化 `ReviewQueue()` 的问题，改为使用进程级单例 `get_default_queue()`
- 修复 `core/agent_loop.py` 中 `get_default_queue` 未导入导致的 NameError（被 try/except 静默吞掉，异步抽样审核从未生效）
- 修复 `research/deep_research_workflow.py` `_search_milvus()` 使用随机向量检索的问题，改为用 embedding 模型生成真实查询向量
- 在 `knowledge/milvus_kb.py` 增加公开的按文本检索方法（内部复用现有 `_embed` + `search`）
- 在 `core/llm_client.py` 为 `chat_with_tools` 增加与现有 `chat_with_retry` 一致的重试逻辑（保留原方法不动）
- 将 `review/review_queue.py` `_save()` 与 `flywheel/metrics_tracker.py` `_save()` 改为原子写入（临时文件 + os.replace）
- 在 `constraints/validator.py` 为 YAML 配置加载增加类级缓存、将输出验证中的正则模式预编译为模块级常量（验证逻辑本身不变）

**不改动**（已验证的正常功能，遵守"之前完成正确的功能尽量不要修改"约束）：
- web/ 全部页面逻辑（含指标快照 60 秒自动采集、趋势图缓存等既有优化）
- swarm/ 的路由与并行执行逻辑
- 审核"超时转异步不重复入队"逻辑
- 飞轮指标"记录即持久化"的语义

## Impact
- Affected specs: 无既有变更规格；本规格为首次优化迭代
- Affected code:
  - `core/agent_loop.py`（审核集成段，约 L285-L322）
  - `core/llm_client.py`（`chat_with_tools`）
  - `research/deep_research_workflow.py`（`_search_milvus`）
  - `knowledge/milvus_kb.py`（新增一个检索方法，不改动现有方法）
  - `review/review_queue.py`（`_save`）
  - `flywheel/metrics_tracker.py`（`_save`）
  - `constraints/validator.py`（配置加载与正则常量）

## 约束（实现时必须遵守）
- 修改函数实现时保留原有逻辑，仅在原基础上调整，不删除既有分支
- 注释一律使用中文，UTF-8 编码，代码中不出现 emoji
- 不新增 fallback 逻辑
- 不编写测试脚本、不新增说明文档

---

## ADDED Requirements

### Requirement: 审核队列单例统一使用
Agent Loop 的专家审核集成 SHALL 统一通过 `get_default_queue()` 获取审核队列，禁止直接实例化 `ReviewQueue`；同时 SHALL 正确导入 `get_default_queue`，使异步抽样审核真正生效。

#### Scenario: 实时审核可被专家操作唤醒
- **WHEN** Agent Loop 中触发实时审核（`submit_realtime`）且专家在 Web 审核页提交操作
- **THEN** 等待中的协程被同一队列实例上的事件唤醒，审核结果正确回填

#### Scenario: 异步抽样不再报 NameError
- **WHEN** Agent 输出命中异步抽样条件（`should_enqueue_async` 返回 True）
- **THEN** 审核项成功入队（审核页可见），日志中不再出现 "专家审核集成失败: name 'get_default_queue' is not defined"

### Requirement: 深度研究真实向量检索
`DeepResearchWorkflow._search_milvus()` SHALL 使用 embedding 模型将查询文本转为向量后再检索 Milvus，SHALL NOT 使用随机向量。`MedicalKnowledgeBase` SHALL 提供公开的按文本检索方法供其调用。

#### Scenario: 知识库检索返回相关结果
- **WHEN** 深度研究工作流对子查询 "糖尿病运动注意事项" 执行知识库检索
- **THEN** 返回的检索结果与查询语义相关（如运动相关文档），而非随机无关内容

### Requirement: 工具调用路径 LLM 重试
`LLMClient` SHALL 为 `chat_with_tools` 提供带指数退避的重试执行方式，Agent Loop 主循环的 LLM 调用 SHALL 使用该重试能力；既有 `chat` / `chat_with_retry` 的实现与行为保持不变。

#### Scenario: 瞬时 API 错误自动恢复
- **WHEN** `chat_with_tools` 调用遇到瞬时网络/API 错误
- **THEN** 按指数退避自动重试，重试期间不中断 Agent 循环；最终成功时返回正常 `LLMResponse`

### Requirement: JSON 持久化原子写入
`ReviewQueue._save()` 与 `FlywheelMetrics._save()` SHALL 采用"先写临时文件、再 `os.replace` 原子替换"的写入方式，保证任意时刻中断不会损坏既有数据文件。

#### Scenario: 写入中断不损坏数据
- **WHEN** 进程在保存 `pending_reviews.json` 或 `flywheel_metrics.json` 过程中被终止
- **THEN** 目标文件内容保持上一次完整写入的状态，下次启动 `_load()` 正常加载，不出现 JSON 解析失败导致的数据清零

### Requirement: 约束配置与正则缓存
`ConstraintValidator` SHALL 对 YAML 配置做进程级缓存（同一进程内多次实例化只读一次文件），并将 `validate_output` 中使用的正则模式预编译为模块级常量；验证规则与判定结果 SHALL 与现有行为完全一致。

#### Scenario: 重复实例化不重复读盘
- **WHEN** 一次提问过程中 SwarmCoordinator 为多个 Agent 创建 AgentLoop、进而创建多个 ConstraintValidator
- **THEN** YAML 文件仅在进程内首次被读取，后续实例复用缓存配置，验证行为不变

#### Scenario: 输出验证结果不变
- **WHEN** 使用与优化前相同的输入分别调用优化前后的 `validate_output`
- **THEN** 违规项与可自动修复项判定结果完全一致
