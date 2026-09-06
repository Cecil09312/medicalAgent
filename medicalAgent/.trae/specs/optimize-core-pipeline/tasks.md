# Tasks

- [x] Task 1: 修复 Agent Loop 审核队列单例问题与缺失导入
  - [x] SubTask 1.1: `core/agent_loop.py` 顶部补充导入 `get_default_queue`（保留现有 `ReviewQueue` 导入，因其仍被其他逻辑引用时需确认后处理；若确认不再使用可保留不动）
  - [x] SubTask 1.2: 将实时审核段（约 L298）的 `ReviewQueue()` 改为 `get_default_queue()`，注释用中文说明单例原因
- [x] Task 2: 修复深度研究随机向量检索
  - [x] SubTask 2.1: `knowledge/milvus_kb.py` 新增公开方法 `search_by_text(query, top_k)`，内部复用现有 `_embed` 与 `search`，不改动现有方法
  - [x] SubTask 2.2: `research/deep_research_workflow.py` `_search_milvus()` 改用 `kb.search_by_text(...)`，移除随机向量逻辑，注释用中文
- [x] Task 3: `chat_with_tools` 增加重试能力
  - [x] SubTask 3.1: `core/llm_client.py` 新增重试版工具调用方法（如 `chat_with_tools_retry`，指数退避与 `chat_with_retry` 保持一致），既有 `chat`/`chat_with_retry`/`chat_with_tools` 实现不动
  - [x] SubTask 3.2: `core/agent_loop.py` 主循环与 max_iterations 兜底两处 LLM 调用改用重试版方法
- [x] Task 4: JSON 持久化原子写入
  - [x] SubTask 4.1: `review/review_queue.py` `_save()` 改为临时文件 + `os.replace`，注释用中文
  - [x] SubTask 4.2: `flywheel/metrics_tracker.py` `_save()` 同样改为原子写入，保持"记录即持久化"语义不变
- [x] Task 5: 约束配置与正则缓存
  - [x] SubTask 5.1: `constraints/validator.py` 为 YAML 加载增加类级缓存（首次读盘，之后复用），保留现有异常处理行为
  - [x] SubTask 5.2: 将 `validate_output` 中的免责声明等正则模式预编译为模块级 `re.compile` 常量，验证逻辑与判定结果保持不变

# Task Dependencies
- Task 1、Task 4、Task 5 相互独立，可并行
- Task 2 的 SubTask 2.2 依赖 SubTask 2.1
- Task 3 的 SubTask 3.2 依赖 SubTask 3.1

# 实施说明
- Task 2 实施时发现两处与预估值不同但必要的调整：
  1. `milvus_kb.search()` 首参本就是查询文本（内部调用 `_embed`），故 `search_by_text` 直接透传文本，避免重复嵌入计算
  2. `deep_research_workflow.get_knowledge_base()` 原返回的是 research/knowledge_base.py 的内存型 KnowledgeBase（随机向量），若不改为返回 `MedicalKnowledgeBase`，`search_by_text` 会 AttributeError；该方法仅被 `_search_milvus()` 调用，影响面封闭。同时在 `_search_milvus` 内将字典结果转换为 `Document`，保持 `run()` 中 `r.content / r.score` 的消费方式不变
