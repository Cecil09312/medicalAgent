# RAG 检索重排序（Rerank）Spec

## Why

当前知识库检索为"bge-small-zh 向量嵌入 + Milvus FLAT/COSINE 单阶段检索"，直接取 top_k=5。单阶段向量检索存在两个固有问题：嵌入向量对查询与文档的语义匹配是"双塔"独立编码，无法做查询-文档的交叉注意力精排；top_k 较小时，相关但表述差异大的文档容易漏检。

引入**重排序（Rerank）**是 RAG 的标准两阶段检索方案：第一阶段向量检索召回较大候选集（保证召回率），第二阶段用 Cross-Encoder 交叉编码器对"查询-文档"配对逐一精排（保证精度），取前 N 条送入 LLM。本项目已有 sentence-transformers 依赖，本地 CrossEncoder 方案无新增依赖、可离线运行、无 API 费用。

## What Changes

- 新增 `knowledge/reranker.py`：
  - `Reranker` 类（进程级单例，`__new__` + 类级 `threading.Lock` 双重检查，风格与 `MedicalKnowledgeBase` 一致）
  - 基于 `sentence_transformers.CrossEncoder` 加载 `BAAI/bge-reranker-base`（默认，可经 `RERANK_MODEL` 更换为 bge-reranker-v2-m3 等）
  - 复用 milvus_kb 的 HuggingFace 本地缓存检测逻辑；已缓存时 `local_files_only=True` 并启用离线环境变量，未缓存时显式关闭离线模式以允许自动下载
  - `rerank(query, documents, top_k)`：对候选文档构造 `[query, text]` 配对，`model.predict` 打分，按分数降序，写入 `rerank_score` 字段后截断返回 top_k
  - `get_default_reranker()` 进程级单例入口
- 修改 `knowledge/milvus_kb.py` 的 `search()`（保留既有全部逻辑，仅在末尾增加可选精排段）：
  - 模块级读取 `RERANK_ENABLED`（默认 `false`）与 `RERANK_CANDIDATE_K`（默认 10）
  - 开启时：Milvus 召回 `limit=max(top_k, RERANK_CANDIDATE_K)` 条候选 → 懒加载 `get_default_reranker()`（方法内 import，关闭时零开销、不加载模型）→ 精排返回 top_k 条
  - 关闭时：召回 limit、返回结构与现状完全一致
  - `doc_type` 过滤、`search_by_text()` 透传逻辑不变；重排序对所有 `search()` 调用方统一生效
- 修改 `.env.example` 与 `.env`：新增 `RERANK_ENABLED`（false）、`RERANK_MODEL`（BAAI/bge-reranker-base）、`RERANK_CANDIDATE_K`（10）三项配置及中文注释
- 修改 `web/chat_tab.py` 后台预热线程：当 `RERANK_ENABLED=true` 时，在预热嵌入模型的同一 daemon 线程中顺带初始化 reranker 单例（提前下载/加载模型，避免首个提问长等待）；关闭时不触碰

**不改动**（遵守"之前完成正确的功能尽量不要修改"）：
- 三个可用 Skill（search-knowledge / analyze-symptoms / clinical-guideline）、深度研究工作流、RAG 评估的调用代码——它们经统一入口自动获得精排能力
- 嵌入模型、Milvus schema/索引、分块逻辑、`add_documents()`、`search_by_text()` 透传语义
- 审核、飞轮、Swarm 编排、流式显示等无关模块

## Impact

- Affected specs: 无（新能力）；与既有 `optimize-core-pipeline`（search_by_text 语义检索）衔接，不冲突
- Affected code:
  - `knowledge/reranker.py`（新增，约 120 行）
  - `knowledge/milvus_kb.py`（`search()` 末尾增加精排段 + 模块级配置常量，约 15 行）
  - `web/chat_tab.py`（预热线程增加约 5 行，开关守卫）
  - `.env.example`、`.env`（新增 3 个配置项）
- 依赖：无新增（`sentence-transformers==5.3.0` 已包含 `CrossEncoder`）
- 模型：首次开启时自动下载 `BAAI/bge-reranker-base`（约 280MB）至 HF 缓存，之后离线可用
- 性能：开启后单次检索增加 10 对文本的 CrossEncoder 推理（CPU 约 0.2-1 秒），以小幅耗时换取检索精度

## 约束（实现时必须遵守）

- 修改函数实现时保留原有逻辑，仅在原基础上增加分支，不删除既有代码路径
- 注释一律使用中文，UTF-8 编码，代码中不出现 emoji
- 不新增 fallback 逻辑：开启重排序后模型加载失败按异常正常抛出（由既有顶层兜底统一处理），不静默降级
- 不编写测试脚本、不新增说明文档
- 默认关闭态下系统行为与现状逐字节一致

---

## ADDED Requirements

### Requirement: 重排序开关默认关闭且关闭态行为不变

系统 SHALL 提供 `RERANK_ENABLED` 环境变量控制重排序，默认值为 `false`。关闭时 `MedicalKnowledgeBase.search()` 的召回数量、返回字段、执行路径 SHALL 与当前完全一致，且进程中 SHALL NOT 加载 CrossEncoder 模型。

#### Scenario: 默认关闭时检索行为不变
- **WHEN** 未设置 `RERANK_ENABLED` 或设为 `false`，调用 `kb.search("糖尿病饮食", top_k=5)`
- **THEN** Milvus 召回 limit=5，返回结果不含 `rerank_score` 字段，进程中不存在 Reranker 实例

### Requirement: 开启后两阶段检索生效

`RERANK_ENABLED=true` 时，`search()` SHALL 先以向量检索召回 `max(top_k, RERANK_CANDIDATE_K)` 条候选，再经 CrossEncoder 对查询与每条候选文本配对打分，按精排分数降序返回 top_k 条；每条结果 SHALL 额外包含 `rerank_score` 字段。

#### Scenario: 候选池扩大并精排截断
- **WHEN** `RERANK_ENABLED=true`、`RERANK_CANDIDATE_K=10`，调用 `kb.search(query, top_k=5)`
- **THEN** Milvus 召回 10 条候选，精排后返回 5 条，返回顺序按 rerank_score 降序

#### Scenario: doc_type 过滤在精排前生效
- **WHEN** 开启重排序并调用 `kb.search(query, top_k=1, doc_type="clinical_guideline")`
- **THEN** 候选池仅包含 clinical_guideline 类型文档，精排在过滤后的候选集上进行

### Requirement: Reranker 单例与模型缓存

`Reranker` SHALL 为进程级单例（`get_default_reranker()` 统一入口），模型加载 SHALL 复用 HuggingFace 本地缓存检测：模型已缓存时以 `local_files_only=True` 离线加载，未缓存时允许自动下载。多次获取 SHALL 只加载一次模型。

#### Scenario: 重复获取不重复加载
- **WHEN** 同一进程内多次调用 `get_default_reranker()`
- **THEN** CrossEncoder 模型只加载一次，返回同一实例

### Requirement: 统一入口对全部调用方生效

重排序 SHALL 在 `MedicalKnowledgeBase.search()` 内部实现，`search_by_text()` 与全部既有调用方（search-knowledge / analyze-symptoms / clinical-guideline 三个 Skill、深度研究工作流 `_search_milvus`、RAG 评估）SHALL 无需任何修改即自动获得精排能力。

#### Scenario: Skill 与深度研究自动受益
- **WHEN** 开启重排序后，经 Skill 或深度研究工作流发起知识库检索
- **THEN** 返回结果为精排后的 top_k 条，调用方代码零改动

### Requirement: 配置与预热

`.env.example` 与 `.env` SHALL 包含 `RERANK_ENABLED`、`RERANK_MODEL`、`RERANK_CANDIDATE_K` 三项配置（默认 false / BAAI/bge-reranker-base / 10）及中文注释。Web 启动预热线程 SHALL 在开关开启时顺带初始化 reranker 单例，关闭时不触碰。

#### Scenario: 开启时启动即预热
- **WHEN** `RERANK_ENABLED=true` 启动 Web 服务
- **THEN** 后台 daemon 线程完成嵌入模型预热后同时加载 reranker 模型，首个提问不再承担模型下载/加载耗时

---

## Acceptance Criteria

### AC-1: 关闭态零影响

- **Type**: `rule`
- **Given**: 不设置或设置 `RERANK_ENABLED=false`
- **When**: 执行知识库检索并检查进程内模块加载情况
- **Then**: Milvus 召回 limit 等于 top_k；返回字典无 `rerank_score` 键；`knowledge.reranker` 模块未被导入（sys.modules 中不存在）；`search()` 既有代码路径完整保留
- **Pass Condition**: 关闭态冒烟测试中上述四项全部成立
- **Evidence**: 自验证命令输出 + 代码审查

### AC-2: 开启态两阶段检索正确

- **Type**: `rule`
- **Given**: `RERANK_ENABLED=true`，reranker 可正常加载
- **When**: 调用 `kb.search(query, top_k=5)`（候选池默认 10）
- **Then**: Milvus 召回 10 条；返回 5 条；每条含 `rerank_score`；返回顺序按 rerank_score 降序；doc_type 过滤仍生效；search_by_text 透传生效
- **Pass Condition**: 以注入假 CrossEncoder（predict 返回可控分数）的方式验证排序与截断逻辑全部成立（不依赖 280MB 模型下载）
- **Evidence**: 自验证脚本输出（注入假模型，断言顺序/数量/字段）

### AC-3: Reranker 单例与缓存逻辑

- **Type**: `rule`
- **Given**: `knowledge/reranker.py`
- **When**: 审查实现并多次调用 `get_default_reranker()`
- **Then**: 单例（`__new__` + Lock 双重检查）；缓存检测复用 HF 路径逻辑；已缓存→local_files_only + 离线环境变量；未缓存→清除离线标志允许下载
- **Pass Condition**: 代码审查逐项通过；py_compile 通过
- **Evidence**: 代码审查记录

### AC-4: 调用方零改动与配置完备

- **Type**: `rule`
- **Given**: 变更完成
- **When**: 检查全部 search 调用方与配置文件
- **Then**: 三个 Skill、deep_research_workflow.py、evaluation/rag_eval.py 无改动；`.env.example` 与 `.env` 含三个新配置项且默认值正确；chat_tab 预热段有开关守卫；requirements.txt 无新增依赖
- **Pass Condition**: 逐项检查全部成立
- **Evidence**: 文件 diff 检查

### AC-5: 代码质量

- **Type**: `rubric`
- **Dimension**: 与既有代码风格的一致性与可维护性
- **Scale**: 1-5
- **Anchors**: 1 = 破坏既有逻辑或风格冲突明显；3 = 功能正确但风格/注释与既有代码不一致；5 = 严格保留既有分支、单例/懒加载/日志风格与 milvus_kb 一致、中文注释清晰无 emoji
- **Pass Threshold**: >= 4
- **Evidence**: 独立审查者评分与评语

## Open Questions

- 无（选型已经用户确认：本地 bge-reranker、默认关闭、统一入口生效）。
