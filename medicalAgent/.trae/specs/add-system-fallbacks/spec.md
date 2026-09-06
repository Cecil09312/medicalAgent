# 系统兜底方案 Spec

## Why
全面盘点系统异常路径后发现：项目已有大量局部兜底（审核超时转异步、任务分解失败降级单 Agent、Worker 异常隔离、90 秒超时、Mem0 降级、LLM 瞬时错误重试、JSON 原子写入等），但仍存在 5 个真实缺口：LLM 连续失败时 Agent Loop 会以"每轮重试 3 次 × 最多 10 轮"的方式轰炸 API 导致响应极慢、Swarm 全部 Worker 失败后仍浪费一次 LLM 合成、process 主入口无统一兜底导致异常裸抛、前端错误提示泄露技术异常细节、空回答直接展示空白。

注：本次为用户明确要求的兜底建设，覆盖默认"不考虑 fallback"规则；仅补齐上述缺口，不重构既有兜底。

## What Changes
- Agent Loop 增加 LLM 连续失败快速兜底：连续 2 轮 LLM 调用失败即跳出主循环，直接进入既有"强制生成最终答案"兜底路径（保留既有逐轮异常日志）
- Swarm 模式增加全 Worker 失败兜底：贡献为空时跳过 LLM 合成，直接返回友好提示答案（超时但部分成功的路径不受影响）
- `SwarmCoordinator.process()` 增加顶层兜底：捕获未预期异常并返回结构化错误结果，CLI 与 Web 获得一致的失败行为（内部函数的 raise 语义保留）
- Web 前端顶层错误提示去技术化：面向用户展示通用友好提示，原始异常仅写日志；CLI 面向开发者保留细节
- 空回答兜底：回答为空白时替换为友好提示（前端已对 None 有默认值，但空字符串会漏过）

## Impact
- Affected specs: 无既有规格变更；与第 6 章优化方案（已实施）互补
- Affected code:
  - `core/agent_loop.py`（主循环异常处理段）
  - `swarm/lead_agent.py`（synthesize_results 空贡献检查）或 `swarm/swarm_coordinator.py`（_process_with_swarm 调用处，二选一，实施时按侵入最小原则确定）
  - `swarm/swarm_coordinator.py`（process 顶层兜底）
  - `web/chat_tab.py`（顶层 except 文案）
  - 空回答兜底落点：`web/chat_tab.py`（取回答处）

## 约束（实现时必须遵守）
- 保留既有函数逻辑与既有兜底分支，仅在原基础上增加，不删除不重构
- 注释一律中文，UTF-8，代码不出现 emoji
- 不编写测试脚本、不新增说明文档
- 不改动 web/ 其他已验证功能（审核、指标快照等）、不改审核超时转异步语义

---

## ADDED Requirements

### Requirement: LLM 连续失败快速兜底
Agent Loop 主循环 SHALL 统计 LLM 调用的连续失败次数，连续失败达到 2 次时 SHALL 终止主循环并直接进入既有"强制生成最终答案"兜底路径；该兜底路径自身再次失败时 SHALL 使用既有的静态致歉文案返回。SHALL NOT 修改既有成功路径、工具调用逻辑与 max_iterations 语义。

#### Scenario: LLM 服务完全不可用时快速返回
- **WHEN** LLM API 持续不可用（重试均失败）
- **THEN** Agent Loop 在连续 2 轮失败后不再继续调用 LLM，快速返回兜底答案，总耗时显著小于旧行为（旧行为为每轮 3 次重试 × 最多 10 轮）

#### Scenario: 瞬时单次失败不受影响
- **WHEN** 某轮 LLM 调用失败但下一轮成功
- **THEN** Agent Loop 正常继续执行，行为与优化前一致

### Requirement: Swarm 全 Worker 失败兜底
Swarm 模式在所有 Worker 均未产生有效贡献时 SHALL 跳过 LLM 结果合成，直接返回友好提示答案；部分 Worker 成功或超时但存在贡献时 SHALL 保持既有合成流程不变。

#### Scenario: 全部 Worker 失败
- **WHEN** Swarm 模式下所有 Worker 子任务均失败（无任何贡献）
- **THEN** 不调用 LLM 合成，直接返回"协作处理失败"类友好提示，并记录错误日志

#### Scenario: 部分成功时正常合成
- **WHEN** 至少一个 Worker 产生贡献（即使发生超时）
- **THEN** 仍走既有 LLM 合成流程，结果与优化前一致

### Requirement: process 主入口统一兜底
`SwarmCoordinator.process()` SHALL 捕获处理过程中的未预期异常，并返回包含友好 answer 与错误信息的结构化结果，SHALL NOT 向调用方（CLI/Web）抛出裸异常；各内部函数现有的异常处理行为保持不变。

#### Scenario: 未预期异常返回结构化结果
- **WHEN** 处理链路中出现未被内层捕获的异常
- **THEN** process 返回 {"answer": 友好提示, "error": 异常信息, ...} 结构，CLI 与 Web 端正常渲染，不出现裸 traceback

### Requirement: 前端错误提示去技术化
Web 对话页顶层异常兜底 SHALL 向用户展示通用友好提示（不包含原始异常文本、堆栈或内部路径），完整异常信息 SHALL 仅写入日志；耗时提示等既有格式保留。

#### Scenario: 异常时用户看到友好提示
- **WHEN** 处理失败且异常到达前端顶层兜底
- **THEN** 用户看到的错误消息不含原始异常字符串（如 API Key 错误、堆栈信息），日志中可查到完整异常

### Requirement: 空回答兜底
当处理结果中的回答为空字符串或纯空白时，系统 SHALL 以友好提示文案替代后展示；回答非空时展示行为不变。

#### Scenario: 空回答被替换
- **WHEN** 下游返回的 answer 为 "" 或纯空白字符
- **THEN** 前端展示"抱歉，暂时无法生成回答，请重试"类提示，而非空白内容
