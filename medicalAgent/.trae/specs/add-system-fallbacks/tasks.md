# Tasks

- [x] Task 1: LLM 连续失败快速兜底
  - [x] SubTask 1.1: `core/agent_loop.py` 主循环内增加连续失败计数（连续 2 次失败跳出主循环），保留既有逐轮异常日志与 `iteration >= max_iterations` 判断
  - [x] SubTask 1.2: 确认跳出后进入既有"强制生成最终答案"兜底路径（该路径已有静态致歉文案兜底，无需新写）
- [x] Task 2: Swarm 全 Worker 失败兜底
  - [x] SubTask 2.1: 在 `_process_with_swarm` 合成前检查 `shared_context.agent_contributions` 是否全部为空；为空则跳过 `synthesize_results`，直接返回友好提示答案（中文），并记录错误日志
  - [x] SubTask 2.2: 确认部分成功/超时但有贡献的路径不受影响（不改变既有 timeout_occurred 合成逻辑）
- [x] Task 3: process 主入口统一兜底
  - [x] SubTask 3.1: `swarm/swarm_coordinator.py` `process()` 主体包 try/except，异常时返回 {"answer": 友好提示, "error": str(e), "warning": "system_error"} 结构并记录日志；既有审核集成 try/except 保留
- [x] Task 4: 前端错误提示去技术化 + 空回答兜底
  - [x] SubTask 4.1: `web/chat_tab.py` 顶层 except 改为面向用户的通用提示（原始异常仅 logger.error），保留耗时显示
  - [x] SubTask 4.2: 取回答处（`result.get("answer", ...)`）增加空白判断，空/纯空白回答替换为友好提示

# Task Dependencies
- Task 1、Task 2、Task 3 相互独立，可并行（Task 3 改 swarm_coordinator.py 的 process 方法，Task 2 改 _process_with_swarm 方法，同文件不同函数，串行执行避免冲突，故 Task 2 与 Task 3 建议由同一子代理顺序完成）
- Task 4 独立，可并行
- 全部任务完成后统一验证

# 实施说明
- 实施过程中发现一处同类既有 Bug 并已一并修复：`swarm_coordinator.py` 顶部审核导入区补充了 `get_default_queue` 导入（原 L109 调用处因未导入，每次触发 NameError 被内层 try/except 吞掉为 warning，协调器的异步抽样审核从未实际入队）。该修复使 process() 内既有的异步抽样集成真正生效，与上一轮"审核队列单例修复"同类。
- Task 4 顶层 except 升级为 logger.exception，完整堆栈入日志；用户文案不含原始异常。
