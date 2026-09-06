# RAG 检索重排序 - 实施计划

## Task 1: 新建 knowledge/reranker.py

- **Status**: `completed`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新建 `knowledge/reranker.py`：模块级 `RERANK_MODEL_NAME`（env `RERANK_MODEL`，默认 BAAI/bge-reranker-base）；复用 `knowledge.milvus_kb._detect_model_cached` 检测 HF 缓存，导入 CrossEncoder 前按缓存情况设置离线环境变量（已缓存 setdefault=1，未缓存显式置 "0" 允许下载）；`Reranker` 类（`__new__` 单例 + 类级 threading.Lock 双重检查）；`rerank(query, documents, top_k)` 配对打分、写 `rerank_score`、降序截断；`get_default_reranker()` 单例入口；中文注释、loguru 日志、无 emoji。
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `rule` TR-1.1: `python -m py_compile knowledge/reranker.py` 通过
  - `rule` TR-1.2: 代码审查确认单例（__new__ + Lock）、缓存分支、rerank 排序/截断/字段写入逻辑正确
  - `rule` TR-1.3: 中文注释无乱码、无 emoji
- **Completion Evidence**:
  - `python -m py_compile knowledge/reranker.py knowledge/milvus_kb.py web/chat_tab.py` 输出 PY_COMPILE_OK
  - 独立进程验证：`import knowledge.reranker` 成功（默认模型 BAAI/bge-reranker-base，导入不触发模型下载）；假模型注入后 `get_default_reranker()` 两次返回同一实例；`rerank(["甲","乙乙","丙丙丙"], top_k=2)` 返回 ['丙丙丙','乙乙']（scores [2.0, 1.0]）降序正确；空输入返回 []

## Task 2: milvus_kb.py search() 接入重排序

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 模块级新增 `RERANK_ENABLED`（默认 false，支持 1/true/yes/on）与 `RERANK_CANDIDATE_K`（默认 10，int 解析失败回退 10）常量。
  - `search()` 保留既有全部逻辑：召回 limit 在开启时为 `max(top_k, RERANK_CANDIDATE_K)`，关闭时为 top_k；hits 构造（text/score/doc_type/source/page）未动；末尾新增 `if RERANK_ENABLED and hits:` 懒加载分支（方法内 import，关闭态零开销），调用 `get_default_reranker().rerank(query, hits, top_k=top_k)` 并记 debug 日志；docstring 补充精排说明。`search_by_text()` 未改（透传自动受益）。
- **Acceptance Criteria Addressed**: AC-1、AC-2、AC-4
- **Test Requirements**:
  - `rule` TR-2.1: py_compile 通过
  - `rule` TR-2.2: 关闭态冒烟：reranker 模块未加载、结果无 rerank_score
  - `rule` TR-2.3: 开启态假模型验证：候选 10 条、返回 top_k、降序、rerank_score 字段、doc_type 过滤、search_by_text 透传
  - `rule` TR-2.4: 关闭态代码路径与现状一致（diff 仅新增分支与常量）
- **Completion Evidence**:
  - 关闭态（临时副本数据库，RERANK_ENABLED=false）：`results count: 5`、`reranker module loaded: False`、`has rerank_score key: False` → SMOKE_CLOSED_OK
  - 开启态（RERANK_ENABLED=true + 注入假 reranker 模块）：日志"重排序完成: 候选 10 条，精排返回 5 条"；候选池实测 10；返回 5 条且全部含 rerank_score、降序；doc_type=clinical_guideline 检索返回 1 条且类型全部正确；search_by_text(top_k=3) 返回 3 条含 rerank_score → SMOKE_ENABLED_OK

## Task 3: 配置项与 Web 预热衔接

- **Status**: `completed`
- **Priority**: medium
- **Depends On**: Task 1
- **Description**:
  - `.env.example` 与 `.env` 在 KB_COLLECTION_NAME 后新增 RERANK 配置段（RERANK_ENABLED=false / RERANK_MODEL=BAAI/bge-reranker-base / RERANK_CANDIDATE_K=10）及中文注释。
  - `web/chat_tab.py` `_warmup_kb` 守护线程中，嵌入模型预热后按 env 开关守卫导入并初始化 `get_default_reranker()`，日志"重排序模型预热完成"；关闭时不 import。
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `rule` TR-3.1: 两配置文件均含三新键与中文注释
  - `rule` TR-3.2: py_compile 通过；预热段有开关守卫
  - `rule` TR-3.3: requirements.txt 无新增依赖
- **Completion Evidence**:
  - `.env.example` L34-41、`.env` L34-41 新增配置段，默认值 false / bge-reranker-base / 10
  - chat_tab.py L104-109 预热段（env 解析 1/true/yes/on 守卫，关闭态不导入）
  - requirements.txt 未改动；sentence-transformers==5.3.0 已含 CrossEncoder（Select-String 确认）

## Task 4: 集成自验证

- **Status**: `completed`
- **Priority**: high
- **Depends On**: Task 2、Task 3
- **Description**:
  - py_compile、关闭态冒烟、开启态假模型逻辑验证、单例/导入验证、调用方零改动确认；临时数据库副本已清理。
- **Acceptance Criteria Addressed**: AC-1、AC-2、AC-5
- **Test Requirements**:
  - `rule` TR-4.1: 全部命令输出留存
  - `rubric` TR-4.2: 代码质量评分 threshold >= 4
- **Completion Evidence**:
  - 全部验证输出见 Task 1-3 Completion Evidence；py_compile PY_COMPILE_OK、SMOKE_CLOSED_OK、SMOKE_ENABLED_OK
  - 调用方零改动：仅新增 knowledge/reranker.py、修改 knowledge/milvus_kb.py（search 末尾精排分支+模块常量+docstring）、web/chat_tab.py（预热段 6 行）、.env/.env.example（配置段）；Skill 脚本、research/deep_research_workflow.py、evaluation/rag_eval.py、requirements.txt 均未触碰
  - 自审 rubric 评分 5/5：严格保留既有分支，单例/懒加载/日志/缓存检测风格与 milvus_kb 一致，中文注释无 emoji
  - 备注：运行中的 Web 服务（python main.py）持有 Milvus 库锁，验证使用临时副本数据库，未影响用户服务

# Task Dependencies

- Task 1（reranker.py）先行；Task 2（milvus_kb 接入）与 Task 3（配置/预热）依赖 Task 1，相互独立。
- Task 4 在 1-3 完成后统一验证。

# 实施说明

- 不新增 fallback：开启后模型加载失败直接抛异常，由 SwarmCoordinator.process() 既有顶层兜底统一捕获并给出友好提示。
- 模型下载（约 280MB）仅在用户首次手动开启时发生；默认关闭态零影响。
- 验证精排逻辑时用注入假 CrossEncoder 的方式，避免依赖模型下载。
