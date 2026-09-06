# Checklist

- [x] `core/agent_loop.py` 中不再直接实例化 `ReviewQueue`，实时审核与异步抽样均使用 `get_default_queue()`
- [x] `core/agent_loop.py` 正确导入了 `get_default_queue`，异步抽样路径不再产生 NameError
- [x] `knowledge/milvus_kb.py` 新增 `search_by_text` 方法且未改动既有方法行为
- [x] `research/deep_research_workflow.py` `_search_milvus` 不再使用随机向量，改用文本语义检索
- [x] `core/llm_client.py` 提供带指数退避的 `chat_with_tools` 重试方法，既有 `chat`/`chat_with_retry` 未被修改
- [x] `core/agent_loop.py` 两处 LLM 调用（主循环 + 兜底）已使用重试版方法
- [x] `review/review_queue.py` `_save()` 与 `flywheel/metrics_tracker.py` `_save()` 均为原子写入（临时文件 + os.replace）
- [x] 飞轮指标"记录即持久化"语义保持不变
- [x] `constraints/validator.py` YAML 配置有进程级缓存，正则模式预编译，验证判定结果与优化前一致
- [x] 所有新增/修改注释为中文且无乱码，代码无 emoji
- [x] 未修改 web/、swarm/ 路由、审核超时转异步、指标快照采集等既有正常功能（文件修改时间核查确认仅 7 个目标文件变更）
- [x] 服务可通过 `python main.py` 正常启动，无 import 错误（py_compile 全部通过 + 全模块导入冒烟测试 IMPORT_ALL_OK + MAIN_IMPORT_OK）
