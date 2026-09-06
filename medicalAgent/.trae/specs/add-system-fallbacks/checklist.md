# Checklist

- [x] `core/agent_loop.py` 连续 2 次 LLM 失败后跳出主循环并进入既有兜底答案路径，单次瞬时失败行为不变
- [x] Swarm 全部 Worker 失败（零贡献）时跳过 LLM 合成并返回友好提示；部分成功时合成流程与优化前一致
- [x] `SwarmCoordinator.process()` 捕获未预期异常并返回结构化错误结果，不再向 CLI/Web 抛裸异常
- [x] Web 前端顶层错误提示不含原始异常文本，完整异常仅写入日志，耗时显示保留
- [x] 空字符串/纯空白回答被替换为友好提示，非空回答展示不变
- [x] 既有兜底（审核超时转异步、分解失败降级单 Agent、Worker 异常隔离、90 秒超时、Mem0 降级）全部未被改动（另修复 swarm_coordinator 缺失的 get_default_queue 导入，使既有异步抽样集成真正生效）
- [x] 所有新增注释为中文且无乱码，代码无 emoji
- [x] py_compile 通过，`import main` 冒烟测试通过（PY_COMPILE_OK + IMPORT_OK）
