# AI智能诊断医疗Agent 设计规格文档

## 1. 项目概述

AI智能诊断医疗Agent 是一个多智能体协作医疗助手系统，采用 **Skills-Agent 两层架构**：

- **Skills 层**: 7 个原子 Skill 自包含，直接转换为 OpenAI function calling 格式
- **Agent 层**: 3 个专业 Agent 通过 Agent Loop 自主选择调用 Skill
- **Swarm 层**: SwarmCoordinator 智能路由，支持单 Agent 和多 Agent 并行两种模式

```
用户问题 -> SwarmCoordinator(智能路由)
  -> 简单问题: 单Agent(ConsultationAgent)
  -> 复杂问题: LeadAgent(分解) -> 3个Worker并行执行 -> LeadAgent(汇总)
```

## 2. 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 语言 | Python 3.10+ | 异步支持 (asyncio) |
| LLM | OpenAI 兼容 API | openai.AsyncOpenAI |
| 向量数据库 | Milvus Lite | 本地存储，无需部署 |
| Embedding | BAAI/bge-small-zh-v1.5 | 512维，Sentence Transformers |
| 长期记忆 | Mem0 | 云服务，可选 |
| 短期记忆 | 内存 / Redis | 可配置切换 |
| 日志 | loguru | 结构化日志 |
| 搜索 | DuckDuckGo | DeepResearch 网络搜索 |
| 评估 | DeepEval | RAG/Agent/Swarm/性能评估 |
| Web前端 | Gradio 4.x | 对话界面 + 审核面板 + 指标仪表盘 |

## 3. 目录结构

```
c:\直播课\medicalAgent\
├── .env                               # 环境配置文件 (python-dotenv)
├── .env.example                       # 配置模板
├── main.py                            # 主入口(交互式对话)
├── requirements.txt                   # 依赖列表
├── setup.py                           # 安装脚本
├── spec.md                            # 本设计规格文档
│
├── core/                              # 核心引擎
│   ├── __init__.py
│   ├── llm_client.py                  # LLM客户端(OpenAI兼容, function calling, 大小模型选择)
│   ├── circuit_breaker.py             # 熔断器(三态状态机, 按模型名进程级单例)
│   ├── model_router.py                # 大小模型路由(contextvar请求级开关 + 简单问题启发式)
│   ├── skill_registry.py              # Skill注册表(注册/执行/转OpenAI format)
│   ├── skill_loader.py                # 动态加载Skills(扫描.claude/skills/)
│   ├── agent_loop.py                  # Agent Loop(Think-Act-Observe循环)
│   └── state_manager.py               # 状态管理
│
├── agents/                            # Agent实现
│   ├── __init__.py
│   ├── base_agent.py                  # Agent基类(Skill注册+Swarm协作)
│   ├── skill_registry_mixin.py        # Skill统一注册混入
│   ├── consultation_agent.py          # 健康咨询Agent
│   ├── diagnostic_agent.py            # 症状诊断Agent
│   └── research_agent.py              # 医学研究Agent
│
├── swarm/                             # Swarm协调器
│   ├── __init__.py
│   ├── events.py                      # 事件驱动通信
│   ├── shared_context.py              # 共享环境(信息素/黑板系统)
│   ├── lead_agent.py                  # 任务分解+结果汇总
│   └── swarm_coordinator.py           # 智能路由+Swarm入口
│
├── memory/                            # 记忆管理
│   ├── __init__.py
│   ├── short_term.py                  # 短期记忆(会话级, 内存/Redis)
│   ├── long_term.py                   # 长期记忆(Mem0跨会话)
│   ├── entropy_manager.py             # 熵管理器(去重/压缩)
│   ├── agent_identity.py              # Agent身份管理
│   └── session_summary.py             # 会话总结
│
├── knowledge/                         # Milvus知识库
│   ├── __init__.py
│   ├── milvus_kb.py                   # 知识库封装(单例)
│   ├── data/
│   │   └── documents/                 # 医学知识文档(txt)
│   └── scripts/
│       ├── __init__.py
│       └── import_hardcoded_data.py   # 数据导入脚本
│
├── constraints/                       # 约束系统(Harness)
│   ├── __init__.py
│   ├── agent_constraints.yaml         # Agent能力边界
│   ├── swarm_constraints.yaml         # Swarm协作规则
│   └── validator.py                   # 运行时约束验证器
│
├── validation/                        # 输出验证和修复
│   ├── __init__.py
│   └── auto_fixer.py                  # 自动修复器
│
├── research/                          # DeepResearch模块
│   ├── __init__.py
│   ├── deep_research_workflow.py      # 深度研究工作流
│   ├── evidence_synthesizer.py        # 证据综合器
│   ├── knowledge_base.py              # 本地知识库(可选)
│   └── web_search.py                  # 网络搜索(DuckDuckGo)
│
├── evaluation/                        # 评估系统 (DeepEval)
│   ├── __init__.py
│   ├── config.py                      # 评估配置(指标阈值/LLM Judge)
│   ├── datasets.py                    # 评估数据集定义
│   ├── rag_eval.py                    # RAG知识库评估
│   ├── agent_eval.py                  # Agent能力评估
│   ├── swarm_eval.py                  # Swarm协作评估
│   ├── performance_eval.py            # 端到端性能评估
│   ├── run_eval.py                    # 统一评估入口
│   └── data/                          # 评估数据集
│       ├── rag_test_cases.json
│       ├── agent_test_cases.json
│       └── swarm_test_cases.json
│
├── review/                              # 专家审核模块
│   ├── __init__.py
│   ├── review_trigger.py               # 审核触发判断(风险/置信度/规则)
│   ├── review_queue.py                  # 审核队列(实时+异步, 进程级单例, 原子写入)
│   ├── expert_interface.py             # 专家审核接口(Web/CLI共用)
│   └── data/
│       └── pending_reviews.json         # 审核队列持久化(原子写入)
│
├── flywheel/                            # 数据飞轮模块
│   ├── __init__.py
│   ├── kb_enhancer.py                   # 知识库增强(专家修正->Milvus)
│   ├── prompt_optimizer.py             # Prompt/约束优化建议
│   ├── eval_expander.py                 # 评估集自动扩充
│   ├── metrics_tracker.py              # 飞轮指标追踪(进程级单例, 记录即持久化)
│   └── data/
│       └── flywheel_metrics.json        # 指标累计值 + 历史快照(原子写入)
│
├── web/                                 # Gradio Web 前端
│   ├── __init__.py
│   ├── app.py                           # Gradio 主应用 (3个Tab, 标题/主题/暗色模式)
│   ├── chat_tab.py                      # 用户对话界面(流式打字机输出)
│   ├── review_tab.py                    # 专家审核面板
│   ├── metrics_tab.py                   # 飞轮指标仪表盘(队列真源同步 + 60秒自动快照)
│   ├── launch.py                        # 启动辅助模块
│   └── assets/                          # 头像资源(bot_avatar.png / user_avatar.png)
│
├── .claude/skills/                    # 7个原子Skills
│   ├── search-knowledge/script/search.py
│   ├── assess-risk/script/risk.py
│   ├── analyze-symptoms/script/symptoms.py
│   ├── recommend-lifestyle/script/lifestyle.py
│   ├── disease-code/script/code.py
│   ├── clinical-guideline/script/guideline.py
│   └── deep-research/script/research.py
│
└── examples/
    └── test_all.py                    # 测试套件
```

## 4. 模块设计

### 4.1 核心引擎 (core/)

#### .env - 环境配置

所有配置通过 `.env` 文件管理，使用 `python-dotenv` 加载：

```bash
# LLM API 配置
LLM_API_KEY=your-api-key
LLM_MODEL_NAME=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=8192

# 知识库 / 向量数据库配置
KB_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
KB_EMBEDDING_DIM=512
KB_SIMILARITY_METRIC=COSINE
KB_CHUNK_SIZE=500            # 文本分块最大字符数
KB_CHUNK_OVERLAP=50          # 相邻块重叠字符数
KB_BATCH_SIZE=64             # 嵌入批处理大小
KB_TOP_K=5                   # 默认检索结果数
KB_DB_PATH=knowledge/data/milvus_lite.db
KB_COLLECTION_NAME=medical_knowledge

# Mem0 配置 (长期记忆，可选)
MEM0_API_KEY=

# Redis 配置 (短期记忆持久化，可选)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# 评估 LLM Judge (可选，不设置则复用 LLM 配置)
# EVAL_LLM_MODEL_NAME=gpt-4o-mini
# EVAL_LLM_TEMPERATURE=0.1

# 专家审核配置
REVIEW_ENABLED=true                # 审核总开关
REVIEW_REALTIME_ENABLED=false      # 实时审核开关（默认 false：高风险问题也走异步，不阻塞回答）
REVIEW_REALTIME_TIMEOUT=15         # 实时审核等待超时秒数，超时自动转异步
REVIEW_ASYNC_SAMPLE_RATE=0.05      # 异步审核随机抽样比例
REVIEW_DATA_PATH=review/data/pending_reviews.json

# 大模型熔断/降级（详见第 8 章，小模型名为空时全部禁用）
LLM_SMALL_MODEL_NAME=              # 小模型名（如 qwen-turbo），空=禁用
LLM_SMALL_BASE_URL=                # 空=复用主 BASE_URL
LLM_SMALL_API_KEY=                 # 空=复用主 API_KEY
CIRCUIT_FAILURE_THRESHOLD=3        # 大模型连续失败熔断阈值
CIRCUIT_COOLDOWN_SECONDS=60        # 熔断冷却秒数
ROUTE_SIMPLE_MAX_LEN=30            # 简单问题长度阈值
```

#### core/llm_client.py - LLM客户端

- 基于 `openai.AsyncOpenAI`，支持 OpenAI 兼容 API
- `chat()`: 异步文本对话
- `chat_with_tools()`: 支持 function calling，返回 `LLMResponse(content, tool_calls, finish_reason)`
- `chat_with_retry()` / `chat_with_tools_retry()`: 3 次指数退避重试（1s/2s/4s）
- `_pick_route()`: 按"请求级路由 contextvar → 熔断器状态"选择大/小模型（详见第 8 章）；未配置小模型时始终走主模型

#### core/circuit_breaker.py - 熔断器

- CLOSED / OPEN / HALF_OPEN 三态状态机；按主模型名进程级单例（LLMClient 多处实例化计数共享）
- 连续失败达阈值 → OPEN，冷却期内请求降级小模型；冷却后半开探测单次放行，成功回 CLOSED、失败重新 OPEN
- `asyncio.Lock` 保护并发状态迁移

#### core/model_router.py - 模型路由

- `is_simple_question()`: 长度阈值（`ROUTE_SIMPLE_MAX_LEN`，默认 30）+ 复杂关键词黑名单（研究/分析/比较/鉴别/原理等）
- `set_route()/reset_route()/get_route()`: 基于 contextvar 的请求级路由开关（auto/small/big），asyncio 子任务自动继承，零侵入调用链
- `route_scope()`: 上下文管理器便捷用法

#### core/skill_registry.py - Skill注册表

- `SkillParameter` 数据类: name, type, description, required, enum
- `SkillRegistry`: register(), execute(), get(), to_openai_format()
- 支持 async/sync Skill 函数

#### core/skill_loader.py - 动态加载

- `discover_skills(project_root)`: 扫描 `.claude/skills/` 目录
- 自动发现 `script/` 下的可调用函数
- 读取 SKILL.md 元数据

#### core/agent_loop.py - Agent循环引擎

- Think-Act-Observe 循环
- 集成短期记忆（消息历史注入）
- 集成约束验证（tool call 前验证 + 输出后验证/修复）
- 硬性限制: `max_tool_calls` 防止过多调用
- 超时/异常时强制生成最终答案

#### core/state_manager.py - 状态管理

- `TaskStatus` 枚举: PENDING, IN_PROGRESS, COMPLETED, FAILED
- `StateManager`: create_state(), 跟踪迭代次数和中间结果

---

### 4.2 知识库 (knowledge/)

#### knowledge/milvus_kb.py - Milvus知识库(单例)

- `MedicalKnowledgeBase`: 单例模式（`__new__` 实现，避免重复加载嵌入模型）
- Milvus Lite 本地存储 (`knowledge/data/milvus_lite.db`)，索引类型 FLAT
- Embedding: `BAAI/bge-small-zh-v1.5` (512维, COSINE)，模型本地缓存检测
- `add_documents()`: 文本分块 -> 向量化 -> 插入，支持 source/page 元数据
- `search()`: 语义检索（传入向量）+ 可选类型过滤
- `search_by_text()`: 文本语义检索（内部完成"查询文本 -> 嵌入向量"，供深度研究工作流复用，避免重复嵌入）

#### 医学知识文档

- 生活方式建议 (01-05): 高血压/糖尿病/感冒/通用健康/急诊症状
- ICD-10编码 (10-12): 心血管/内分泌/感染类
- 临床指南 (20-22): 高血压/糖尿病指南 txt + 糖尿病指南 PDF（pypdf 逐页解析，页码写入 `page` 字段，检索引用形如"指南.pdf，第 3 页"）

#### 数据导入脚本

- 硬编码所有医学知识数据
- 调用 `MedicalKnowledgeBase.add_documents()` 批量导入

---

### 4.3 原子Skills (.claude/skills/)

每个 Skill 包含 `SKILL.md`(元数据) + `script/`(实现):

| Skill | 函数名 | 数据源 | 功能 |
|-------|--------|--------|------|
| search-knowledge | `search_knowledge` | Milvus | 语义检索医学知识 |
| assess-risk | `assess_risk` | 规则引擎 | 高危症状识别/风险分级 |
| analyze-symptoms | `analyze_symptoms` | 规则引擎 | 多系统症状模式分析 |
| recommend-lifestyle | `recommend_lifestyle` | Milvus | 饮食/运动/用药建议 |
| disease-code | `disease_code` | Milvus | ICD-10编码查询 |
| clinical-guideline | `clinical_guideline` | Milvus | 临床指南检索 |
| deep-research | `deep_research` | 网络搜索 | 深度研究+证据综合 |

---

### 4.4 Agent实现 (agents/)

#### agents/base_agent.py - Agent基类

- `BaseAgent(ABC)`: agent_id, config, llm_client, loop, skill_registry
- 抽象方法: `get_system_prompt()`, `register_tools()`
- Swarm协作: capabilities, shared_context, process_subtask()

#### agents/skill_registry_mixin.py - 统一Skill注册

- `SkillRegistryMixin.register_all_skills()`: 自动发现并注册所有Skills
- 从函数签名推断参数类型

#### 三个Worker Agent

| Agent | 职责 | System Prompt 特点 |
|-------|------|-------------------|
| ConsultationAgent | 健康咨询 | 通俗易懂 |
| DiagnosticAgent | 症状诊断 | 鉴别诊断推理 |
| ResearchAgent | 医学研究 | 循证医学 |

---

### 4.5 记忆系统 (memory/)

#### memory/short_term.py - 短期记忆(单例)

- `ConversationHistory` 数据类: session_id, messages, timestamps
- `ShortTermMemory`: 支持 memory/redis 两种存储
- `add_message()`, `get_recent_messages()`, `get_history()` (OpenAI格式)
- 集成熵管理: 自动去重(MD5) + 压缩(早期消息摘要)

#### memory/long_term.py - 长期记忆(Mem0)

- `LongTermMemory`: Mem0云服务集成
- `add_session_summary()`: 保存会话总结
- `search_similar_sessions()`: 向量相似度搜索历史案例
- 优雅降级: Mem0不可用时系统继续工作

#### memory/entropy_manager.py - 熵管理器

- 消息去重(MD5哈希), 历史压缩, 熵估算
- `auto_clean()`, `deduplicate_sessions()`

#### memory/agent_identity.py + session_summary.py

- Agent身份文件管理, 会话总结生成和存储

---

### 4.6 Swarm协调 (swarm/)

#### swarm/events.py - 事件系统

- `EventType` 枚举: SWARM_STARTED, TASK_DECOMPOSED, SUBTASK_STARTED 等
- `Event` 数据类: type, source_agent, target_agent, data

#### swarm/shared_context.py - 共享环境

- `SharedContext`: 黑板系统, 所有Agent可读写
- `SubTask` 数据类: id, type, description, assigned_agent, status
- `Contribution` 数据类: agent_id, subtask_id, result
- 任务管理: add_subtask(), get_subtasks_for_agent(), complete_subtask()

#### swarm/lead_agent.py - Lead Agent

- `assess_and_decompose()`: LLM分析问题 -> 分解子任务 -> 分配Agent
- `create_subtasks()`: 创建SubTask并发布到SharedContext
- `synthesize_results()`: 汇总所有Agent贡献 -> LLM生成最终答案

#### swarm/swarm_coordinator.py - Swarm入口

- 智能路由: 1个子任务->单Agent, 2+个子任务->Swarm
- 统一记忆检索: 短期记忆(会话历史) + 长期记忆(相似案例)
- Worker并行执行, 90秒超时
- 结果汇总 + 记忆保存
- 便捷函数: `process_with_swarm()`

---

### 4.7 约束系统 (constraints/ + validation/)

#### constraints/agent_constraints.yaml

- 每个Agent的 allowed_tools, forbidden_actions, output_constraints
- 通用安全规则: 严重症状必须建议就医, 不延误急救

#### constraints/swarm_constraints.yaml

- max_agents_per_task, timeout_seconds
- task_decomposition_rules: 任务复杂度分类
- agent_selection_rules: 高风险症状/指南/生活方式关键词路由规则
- quality_control, conflict_resolution

#### constraints/validator.py - 约束验证器

- `validate_tool_call()`: 验证工具调用是否允许
- `validate_output()`: 验证输出(免责声明/高危警告/禁止行为)
- `validate_task_decomposition()`: 验证任务分解合理性
- `get_required_agents()`: 根据问题关键词推荐 Agent

#### validation/auto_fixer.py - 自动修复器

- 缺少免责声明 -> 自动添加
- 高危症状 -> 自动添加就医提醒

---

### 4.8 DeepResearch (research/)

- `web_search.py`: DuckDuckGo搜索 + 医学网站白名单
- `evidence_synthesizer.py`: LLM驱动的证据综合
- `deep_research_workflow.py`: 查询规划 -> 并行搜索 -> 证据综合 -> 质量验证

---

### 4.9 主入口和测试

#### main.py - 交互式对话入口

- 交互式命令行, 调用 `process_with_swarm()`
- 显示协作模式/执行时间/建议/免责声明

#### examples/test_all.py - 测试套件

- 核心功能测试: Agent Loop, Skill调用, 记忆系统, Swarm协作
- 约束系统测试: 验证/修复

## 5. 关键设计决策

1. **配置集中化**: 所有配置通过 `.env` 文件管理，支持环境变量覆盖
2. **模块解耦**: 各模块通过接口通信，便于后续替换组件
3. **约束驱动**: Harness Engineering 理念，约束验证 + 自动修复贯穿全链路
4. **Skill 即函数**: 原子 Skill 自包含，自动转换为 OpenAI function calling 格式
5. **优雅降级**: Mem0/Redis 等外部依赖不可用时，系统仍可正常运行
6. **熵管理**: 记忆系统自动去重、压缩，控制上下文窗口膨胀

---

### 4.10 评估系统 (evaluation/)

基于 **DeepEval** 框架，覆盖 RAG / Agent / Swarm / 性能 四个评估维度。

#### evaluation/config.py - 评估配置

定义各维度的指标阈值和 LLM Judge 配置：
- RAG: min_faithfulness=0.7, min_relevancy=0.7, min_context_precision=0.6
- Agent: min_tool_accuracy=0.8, min_task_completion=0.7
- Swarm: min_decomposition_quality=0.7, max_coordination_time=120s
- Performance: max_latency_p95=30s, max_tokens_per_query=5000

#### evaluation/rag_eval.py - RAG知识库评估

使用 DeepEval 内置指标:
- **Faithfulness**: 答案是否忠实于检索到的上下文
- **Answer Relevancy**: 答案与问题的相关程度
- **Context Precision**: 检索上下文的排序精度
- **Context Recall**: 期望答案的关键信息是否被检索到
- **Hallucination**: 检测答案中的幻觉内容

核心类: `RAGEvaluator.evaluate(test_cases) -> RAGEvalResult`

#### evaluation/agent_eval.py - Agent能力评估

使用 DeepEval 指标 + 自定义指标:
- **Tool Correctness**: Agent 是否选择了正确的 Skill
- **Answer Correctness**: 最终答案与期望答案的语义相似度
- **Task Completion**: LLM-as-Judge 判断任务是否完成
- **Response Quality**: 自定义医疗领域质量评分（完整性/准确性/安全性）

核心类: `AgentEvaluator.evaluate(test_cases) -> AgentEvalResult`

#### evaluation/swarm_eval.py - Swarm协作评估

自定义指标（DeepEval 自定义 Metric 接口）:
- **Decomposition Quality**: LLM-as-Judge 评估任务分解合理性
- **Agent Selection Accuracy**: 是否选择了正确的 Agent 组合
- **Coordination Efficiency**: 并行执行效率
- **Synthesis Quality**: 结果汇总质量

核心类: `SwarmEvaluator.evaluate(test_cases) -> SwarmEvalResult`

#### evaluation/performance_eval.py - 端到端性能评估

纯指标采集，不依赖 DeepEval:
- **Latency**: 响应延迟（P50/P95/P99）
- **Token Usage**: Token 消耗统计
- **Throughput**: 单位时间处理请求数 (QPM)
- **Tool Call Count**: 平均工具调用次数

核心类: `PerformanceEvaluator.evaluate(test_cases, iterations) -> PerfEvalResult`

#### evaluation/run_eval.py - 统一入口

```python
async def run_evaluation(dimensions: List[str] = None, report_format: str = "console"):
    """运行评估，dimensions 可选: rag/agent/swarm/performance/all"""
```

- 支持按维度运行或全部运行
- 输出格式: console（终端表格）/ json（详细报告）
- 汇总各维度评分，与阈值对比，给出不达标项警告
- 命令行: `python -m evaluation.run_eval -d rag agent -f json -o report.json`

---

### 4.11 专家审核系统 (review/)

支持实时审核（同步阻塞）和异步审核（事后处理）两种模式。**审核队列必须使用进程级单例 `get_default_queue()`**：实时审核的等待事件保存在队列实例内存中，提问方（chat_tab/agent_loop/swarm）与专家审核方（review_tab）共享同一实例，专家操作才能跨线程唤醒等待协程。

#### review/review_trigger.py - 审核触发判断

实时审核触发条件（满足任一即触发，需 `REVIEW_REALTIME_ENABLED=true` 开启，默认关闭走异步）:
- 高风险症状 (`risk_level=HIGH`)
- 约束不可自动修复的违规
- 涉及敏感操作（诊断/处方类内容）
- Agent 被强制终止（低置信度）
- 回答包含高风险关键词（胸痛/昏迷/休克/大出血/呼吸困难/中风/心梗）

异步审核触发条件:
- 随机抽样（`REVIEW_ASYNC_SAMPLE_RATE`，默认 5%）
- 用户主动标记（"🚩 标记有误"按钮，reason=user_flag）

#### review/review_queue.py - 审核队列

- `ReviewItem`: 审核项数据结构（id/问题/回答/触发原因/模式/状态 pending/approved/corrected/rejected/专家修正）
- `get_default_queue()`: 进程级单例（全系统统一入口）
- `submit_realtime()`: 实时审核，`asyncio.Event` + `call_soon_threadsafe` 跨线程唤醒；超时（默认 15 秒，`REVIEW_REALTIME_TIMEOUT`）自动转为异步待审核（mode=async, status=pending），原回答先行展示，**不重复入队**
- `enqueue_async()`: 异步审核，入队后直接返回 review_id
- `submit_approval()` / `submit_correction()` / `submit_rejection()`: 专家三种操作，已处理项重复操作返回 False；每次操作联动飞轮指标
- 持久化到 `review/data/pending_reviews.json`，**原子写入**（临时文件 + `os.replace`），崩溃不损坏

#### review/expert_interface.py - 专家接口

- `list_pending()` / `list_all()`: 列出待审核/全部审核项
- `get_detail()`: 审核项详情
- `review()`: 审核操作（approve/correct/reject）
- `get_stats()`: 审核统计（pending/approved/corrected/rejected/total）
- `run_cli()`: 命令行交互界面

---

### 4.12 数据飞轮 (flywheel/)

形成 "系统回答 -> 专家审核 -> 修正反馈 -> 系统改进" 的闭环。

#### flywheel/kb_enhancer.py - 知识库增强

- 专家修正 -> 向量化 -> 注入 Milvus 知识库
- 标记 `doc_type="expert_reviewed"`, `source="flywheel"`
- 支持单条和批量注入

#### flywheel/prompt_optimizer.py - Prompt/约束优化建议

- 分析专家修正模式（触发原因分布、修正幅度、按 Agent 分组）
- 识别系统性问题（如某类问题总缺免责声明）
- 生成优化建议（Prompt 修改 + 约束规则修改）
- `generate_prompt_patch()`: 生成可审阅的补丁文本

#### flywheel/eval_expander.py - 评估集自动扩充

- 专家审核通过的 Q&A 自动加入评估数据集
- 根据问题内容分类到 agent/rag/swarm 测试集
- 可配置最小置信阈值

#### flywheel/metrics_tracker.py - 飞轮指标追踪

核心指标:
- `intervention_rate`: 专家介入率（= 总审核数 / 总查询数，0%-100%，应随时间下降）
- `correction_rate`: 修正率（= 修正数 / 审核数）
- `avg_review_time_seconds`: 平均审核耗时
- `kb_expert_docs`: 知识库专家贡献文档数
- `eval_cases_from_flywheel`: 飞轮贡献的评估用例数

机制:
- `get_default_metrics()`: 进程级单例，记录方（提问/审核）与展示方（指标页）共享同一实例
- **记录即持久化**：每次 record_query/record_review/record_approval 等立即原子写入 `flywheel/data/flywheel_metrics.json`（含 current 累计值 + history 快照），进程重启不丢数据
- 审核动作（批准/修正/拒绝）由 `ReviewQueue._record_review_metrics()` 自动联动记录

---

### 4.13 Web 前端 (web/)

基于 **Gradio** 框架，提供 3 个 Tab 页面的 Web 界面。运行方式: `python main.py`（默认 Web 模式；`python main.py --cli` 为命令行交互；`-v` 详细日志），访问 `http://localhost:7860`。浏览器标签页标题与页面头部均为"AI智能医疗诊断系统"。

#### web/app.py - 主应用

- `create_app()`: 创建 `gr.Blocks` 应用，组装 3 个 Tab（💬 对话 / 🔍 专家审核 / 📊 飞轮指标）
- 紫色系自定义主题（`gr.themes.Soft`）+ 自定义 CSS + 暗色模式 JS
- 启动时初始化 Swarm 系统并后台预热知识库嵌入模型，避免首个提问长等待

#### web/chat_tab.py - 用户对话界面

- `gr.Chatbot(type="messages")` 对话历史展示，配置用户/AI 头像（`web/assets/`）
- **流式打字机输出**：发送后先显示"🤔 思考中"占位，答案就绪后按固定步长逐段累加渲染（详见第 9 章）
- 调用 `SwarmCoordinator.process()` 处理问题；懒初始化 Swarm 系统（Agents + Coordinator + 记忆单例）
- 显示执行耗时 + 建议 + 免责声明；错误提示去技术化（完整堆栈仅入日志）
- "🚩 标记有误" 按钮 → `ReviewQueue.enqueue_async(reason="user_flag")`，返回审核编号
- "🧹 重置会话" 按钮 → 后台 LLM 生成会话摘要写入长期记忆，生成新 session_id
- 快速提问卡片（6 个常见问题）点击填充输入框；发送中禁用输入与按钮防重复提交

#### web/review_tab.py - 专家审核面板

- `gr.Dataframe` 展示审核列表（ID/触发原因/模式/问题/回答）
- 点击行查看详情（问题 + 原始回答 + 触发原因 + 状态）
- 操作区：专家 ID 输入框 + ✅ 批准 / ✏️ 提交修正（文本框）/ ❌ 拒绝
- 统计信息 `gr.Markdown`（待审核/已批准/已修正/已拒绝/总计）实时更新
- 刷新按钮从审核队列文件重新加载所有数据

#### web/metrics_tab.py - 飞轮指标仪表盘

- 核心指标表格：介入率、修正率、审核耗时、KB贡献、评估用例数
- **审核指标以审核队列文件为真源**：每次展示前 `_sync_metrics()` 从队列统计覆盖审核计数，保证与专家审核 Tab 数据一致
- `gr.Plot` 趋势图（介入率 + 修正率双轴折线图，matplotlib 中文字体处理，Figure 缓存避免重复渲染）
- `gr.Dataframe` 历史快照表（最近 20 条）；无数据时显示"暂无历史数据"提示
- "🔄 刷新指标"按钮仅刷新展示、不产生快照；`gr.Timer` 每 60 秒自动同步，**仅在数据变化时**拍摄快照

#### web/launch.py - 启动辅助模块

- 供 `main.py` 调用的启动辅助（.env 加载、sys.path 设置）；实际 Web 启动入口为 `main.py`

---

## 6. 代码优化方案（2026-09-05 代码审查）

对全部代码库审查后确定 5 项有实际价值的优化。原则：保留既有函数逻辑，仅在原基础上修改；不引入 fallback；不改动 web/、swarm/ 路由、审核超时转异步、指标快照采集等已验证功能。

### 6.1 优化项清单

| # | 优化项 | 位置 | 优先级 |
|---|--------|------|--------|
| 1 | 审核队列单例修复 + 缺失导入 | core/agent_loop.py | 高（Bug） |
| 2 | 深度研究随机向量检索修复 | research/deep_research_workflow.py + knowledge/milvus_kb.py | 高（Bug） |
| 3 | chat_with_tools 重试 | core/llm_client.py + core/agent_loop.py | 中高 |
| 4 | JSON 持久化原子写入 | review/review_queue.py + flywheel/metrics_tracker.py | 中 |
| 5 | 约束配置与正则缓存 | constraints/validator.py | 中 |

### 6.2 详细方案

#### 优化 1：审核队列单例修复 + 缺失导入（Bug）

**问题**：
- `core/agent_loop.py` 实时审核段直接 `ReviewQueue()` 实例化，违反"进程级单例 `get_default_queue()`"约束，专家在审核页的操作无法唤醒 Agent Loop 中等待的协程
- 同文件异步抽样段调用了 `get_default_queue()` 但从未导入，每次触发抛出 NameError 并被 try/except 静默吞掉（仅留下 "专家审核集成失败" 警告日志），异步抽样审核从未生效

**方案**：
- 顶部补充 `from review.review_queue import get_default_queue`
- 实时审核段 `ReviewQueue()` 改为 `get_default_queue()`

**预期**：实时审核可被专家操作唤醒；异步抽样入队成功，不再出现 NameError 警告。

#### 优化 2：深度研究随机向量检索修复（Bug）

**问题**：`research/deep_research_workflow.py` `_search_milvus()` 使用 `random.uniform(-1, 1)` 生成随机查询向量检索 Milvus（代码注释自述"实际应用中应使用 embedding 模型"），导致深度研究的知识库检索结果完全随机、无效。

**方案**：
- `knowledge/milvus_kb.py` 新增公开方法 `search_by_text(query, top_k)`，内部复用现有 `_embed()` 与 `search()`，不改动既有方法
- `_search_milvus()` 改用 `kb.search_by_text(...)`，移除随机向量逻辑

**预期**：深度研究的知识库检索返回与查询语义相关的结果。

#### 优化 3：chat_with_tools 重试

**问题**：`LLMClient.chat` 有 `chat_with_retry`（3 次指数退避），但 Agent 主循环使用的 `chat_with_tools` 无任何重试，瞬时 API/网络错误直接中断回答。

**方案**：
- `core/llm_client.py` 新增重试版工具调用方法（指数退避策略与 `chat_with_retry` 一致），既有 `chat` / `chat_with_retry` / `chat_with_tools` 实现不动
- `core/agent_loop.py` 主循环与 max_iterations 兜底两处 LLM 调用改用重试版方法

**预期**：瞬时错误自动恢复，不中断 Agent 循环。

#### 优化 4：JSON 持久化原子写入

**问题**：`ReviewQueue._save()` 与 `FlywheelMetrics._save()` 均直接 `open(path, "w")` 覆盖写入，进程在写入中途崩溃/断电会留下半截 JSON；下次启动 `_load()` 解析失败将内存清空，导致审核队列历史或飞轮指标丢失。

**方案**：两处 `_save()` 改为"先写同目录临时文件，再 `os.replace()` 原子替换"。飞轮指标"记录即持久化"语义保持不变。

**预期**：任意时刻中断，数据文件保持上一次完整状态。

#### 优化 5：约束配置与正则缓存

**问题**：`ConstraintValidator.__init__` 每次实例化都读两个 YAML；SwarmCoordinator 每次提问为各 Agent 新建 AgentLoop → 重复读盘。`validate_output` 每次调用都重新构造正则模式列表重复编译。

**方案**：
- YAML 加载增加类级缓存（进程内首次读盘，之后复用），保留现有异常处理行为
- 免责声明等正则模式预编译为模块级 `re.compile` 常量，验证逻辑与判定结果不变

**预期**：一次提问过程中的多次验证器实例化只读一次盘，验证结果与优化前完全一致。

### 6.3 验证方式

- `python main.py` 启动无 import 错误
- 对话页提问 → 审核页可见异步抽样入队记录，日志无 NameError
- 深度研究类问题检索结果与查询语义相关
- 强杀进程后重启，审核队列与飞轮指标数据完好

---

## 7. 系统兜底方案（2026-09-05 兜底盘点，已实施完成并验证）

### 7.1 既有兜底盘点（已具备，不重复建设）

| 场景 | 既有兜底行为 |
|------|-------------|
| 专家实时审核无人响应 | 超时自动转异步待办，原回答先行展示 |
| 任务分解 LLM 失败/解析失败 | 降级为默认单 Agent（consultation）处理 |
| Agent Loop 达到最大迭代 | 强制 LLM 生成最终答案；再失败则静态致歉文案 |
| Swarm Worker 异常 | 子任务标记 FAILED，不影响其他 Worker 并行执行 |
| Swarm 执行超时 | 90 秒超时标记，合成时提示"信息可能不完整" |
| 结果合成失败 | 返回原始贡献文本兜底 |
| 记忆读写失败 | 警告日志，不影响主流程 |
| Mem0 不可用 | 自动降级本地存储 |
| LLM 瞬时错误 | chat_with_retry / chat_with_tools_retry 指数退避重试 |
| 知识库检索失败 | Skill 返回错误信息，Agent 基于通用知识继续回答 |
| 审核队列/飞轮指标 JSON 写入中断 | 原子写入（第 6 章已完成），数据不损坏 |
| 飞轮指标记录失败 | 警告日志，不影响回答主流程 |

### 7.2 缺口与新增兜底方案

| # | 兜底项 | 位置 | 优先级 |
|---|--------|------|--------|
| 1 | LLM 连续失败快速兜底 | core/agent_loop.py | 高 |
| 2 | Swarm 全 Worker 失败兜底 | swarm/swarm_coordinator.py（合成前检查） | 高 |
| 3 | process 主入口统一兜底 | swarm/swarm_coordinator.py | 中 |
| 4 | 前端错误提示去技术化 | web/chat_tab.py | 中 |
| 5 | 空回答兜底 | web/chat_tab.py | 中 |

#### 兜底 1：LLM 连续失败快速兜底（高）

**缺口**：LLM 服务完全不可用时，Agent Loop 主循环的逐轮异常处理会在未达 max_iterations 前持续循环——每轮 chat_with_tools_retry 重试 3 次 × 最多 10 轮，叠加指数退避等待，用户需等待极长时间才能收到失败反馈。

**方案**：主循环内增加连续失败计数，连续 2 轮 LLM 调用失败即跳出主循环，直接进入既有"强制生成最终答案"兜底路径（该路径再失败时已有静态致歉文案）。单次瞬时失败后下一轮成功的行为保持不变。

#### 兜底 2：Swarm 全 Worker 失败兜底（高）

**缺口**：所有 Worker 子任务失败时贡献为空，synthesize_results 仍会发起一次 LLM 合成调用——既浪费调用，且合成结果基于空内容不可控。

**方案**：合成前检查贡献是否全部为空，为空则跳过 LLM 合成，直接返回友好提示答案并记录错误日志。部分成功或超时但存在贡献的路径不受影响。

#### 兜底 3：process 主入口统一兜底（中）

**缺口**：`SwarmCoordinator.process()` 无顶层异常兜底，链路中未预期的异常会裸抛给调用方，CLI 与 Web 各自处理、行为不一致。

**方案**：process 主体包 try/except，异常时返回 {"answer": 友好提示, "error": 异常信息, "warning": "system_error"} 结构化结果。内部函数既有异常处理行为不变。

#### 兜底 4：前端错误提示去技术化（中）

**缺口**：Web 对话页顶层 except 将原始异常文本（可能含 API Key 错误、内部路径等技术细节）直接展示给终端用户。

**方案**：面向用户展示通用友好提示，完整异常仅写入日志；耗时提示等既有格式保留。CLI 面向开发者，保留现状。

#### 兜底 5：空回答兜底（中）

**缺口**：`result.get("answer", 默认文案)` 仅对 None 生效，下游返回空字符串或纯空白时前端会展示空白内容。

**方案**：取回答处增加空白判断，空白回答替换为"抱歉，暂时无法生成回答，请重试"类提示。

### 7.3 验证方式

- 临时将 .env 中 LLM_API_KEY 改为无效值后提问：应在短时间内收到友好失败提示（而非长时间无响应），前端提示不含 API 错误细节，日志中有完整异常
- 恢复正确 API Key 后提问：行为与现在完全一致（瞬时单次失败不受影响）
- py_compile 通过，`import main` 冒烟测试通过

### 7.4 实施与验证结果（2026-09-05 已完成）

全部 5 项兜底已实施并完成实机验证：

| 兜底项 | 实施位置 | 实测结果 |
|--------|---------|---------|
| 1. LLM 连续失败快速兜底 | `AgentLoop.run()`（core/agent_loop.py）：主循环新增连续失败计数，LLM 调用单独 try/except，连续 2 轮失败跳出并进入既有"强制生成最终答案"路径，外层 except 与工具执行逻辑一行未动 | 无效 API Key 下提问 10.5 秒返回静态致歉（旧行为预计 60 秒以上），日志确认仅执行 2 轮主循环 |
| 2. Swarm 全 Worker 失败兜底 | `SwarmCoordinator._process_with_swarm()`（swarm/swarm_coordinator.py）：合成前检查零贡献，跳过 `synthesize_results` 直接返回友好提示；部分成功/超时但有贡献的路径不变 | 结构审查通过，零贡献判定不触发 defaultdict 键插入副作用 |
| 3. process 主入口统一兜底 | `SwarmCoordinator.process()`：主体包 try/except，异常返回 `{"answer": 友好提示, "error", "warning": "system_error"}` 结构化结果 | monkeypatch 模拟未预期异常：0.3 秒返回结构化结果，用户文案不含 RuntimeError 等技术细节 |
| 4. 前端错误提示去技术化 | `web/chat_tab.py` `_process_question()` 顶层 except：用户只见通用提示，`logger.exception` 完整堆栈入日志，耗时显示保留 | 与兜底 3 联合覆盖，用户可见文案无 API Key/堆栈细节 |
| 5. 空回答兜底 | `web/chat_tab.py` 取回答处：None/空串/纯空白统一替换为友好提示 | 代码验证通过，非空回答展示不变 |
| 附带修复 | swarm/swarm_coordinator.py 补充 `get_default_queue` 导入（原 NameError 被内层 try/except 吞掉，协调器异步抽样审核从未实际入队，与第 6 章优化 1 同类问题） | 修复后 process() 内既有的异步抽样集成真正生效 |

验证环境安全：`.env` 测试后已从备份还原（MD5 哈希一致），py_compile 与 `import main` 冒烟测试通过。

---

## 8. 大模型降级、熔断器与大小模型路由（2026-09-05，已实施完成并验证）

### 8.1 现状与目标

现状：系统仅配置单一模型（qwen-plus），模型层容错只有"指数退避重试 + 静态致歉兜底"（第 6/7 章已完成）——大模型完全不可用时用户只能收到致歉文案，且简单问题也消耗大模型配额。

目标模型层容错体系：
- **正常业务用大模型**：默认所有请求走 qwen-plus（大模型），质量优先
- **简单问题路由小模型**：短且无复杂关键词的问题（如"感冒了怎么办"）整请求路由到小模型（如 qwen-turbo），省钱提速
- **熔断器**：大模型连续失败达到阈值 → OPEN，冷却期内全部请求改走小模型；冷却后半开探测恢复
- **大模型降级**：熔断期间小模型顶替，用户仍获得真实回答而非致歉

### 8.2 配置设计（.env 新增，默认禁用态）

```bash
LLM_SMALL_MODEL_NAME=            # 小模型名，空=禁用降级与路由（如 qwen-turbo）
LLM_SMALL_BASE_URL=              # 空=复用主 BASE_URL
LLM_SMALL_API_KEY=               # 空=复用主 API_KEY
CIRCUIT_FAILURE_THRESHOLD=3      # 大模型连续失败熔断阈值
CIRCUIT_COOLDOWN_SECONDS=60      # 熔断冷却秒数
ROUTE_SIMPLE_MAX_LEN=30          # 简单问题长度阈值
```

小模型未配置时所有新能力自动关闭，行为与现状完全一致。

### 8.3 架构设计

| 组件 | 位置 | 职责 |
|------|------|------|
| 熔断器 | core/circuit_breaker.py（新增） | CLOSED/OPEN/HALF_OPEN 三态；按主模型名进程级单例（LLMClient 每 Agent 独立实例化共 7 处，计数必须共享）；asyncio.Lock 保护 |
| 路由模块 | core/model_router.py（新增） | is_simple_question 启发式（长度阈值 + 复杂关键词黑名单：研究/分析/比较/鉴别/原理/为什么/机制/方案/综述/设计等）+ contextvar 请求级路由开关（auto/small/big） |
| 模型选择 | core/llm_client.py（修改） | __init__ 配置齐全时创建小模型客户端；chat()/chat_with_tools() 调用前按"路由 contextvar → 熔断状态"选模型：简单路由→小模型；OPEN→小模型；HALF_OPEN→放行大模型探测；默认→大模型。大模型调用成败实时计入熔断器（小模型调用不计入）；chat_with_retry/chat_with_tools_retry 重试策略不动 |
| 入口路由 | swarm/swarm_coordinator.py（修改） | process() 入口对简单问题设置本次请求协程的路由 contextvar（子任务 gather 继承，Agent/Skill 深层透明生效），结束恢复默认 |

### 8.4 关键机制

- **熔断与重试协同**：计数在底层 API 调用结果处实时记录，重试的每次尝试依次计入——大模型 Key 失效时同一次提问内即可触发 OPEN，后续重试与兜底路径自动改走小模型，最终返回小模型生成的真实回答（而非静态致歉）
- **防误熔断**：大模型调用成功即重置连续失败计数，瞬时抖动（失败 1-2 次后恢复）不触发熔断
- **半开恢复**：OPEN 超过冷却期后放行一次大模型探测，成功回 CLOSED，失败重新 OPEN 并重置冷却
- **可观测**：模型选择与熔断状态迁移输出中文日志（模型名 + 触发原因：简单路由/熔断降级/默认）
- **零侵入调用链**：路由经 contextvar 传递，agents→agent_loop→llm_client 链路不改动任何函数签名

### 8.5 验证方式

- 配置 `LLM_SMALL_MODEL_NAME=qwen-turbo` 后：简单问题走小模型、复杂问题走大模型（日志可辨模型名）
- 主 Key 临时置无效 + 小模型 Key 有效：连续 3 次失败后熔断 OPEN，单次提问内后续调用改走小模型并返回真实回答；恢复主 Key 后冷却期满半开探测成功回 CLOSED
- 瞬时抖动（失败后成功）不触发熔断
- 移除小模型配置：行为与现状完全一致
- py_compile 通过，`import main` 冒烟测试通过

### 8.6 实施与验证结果（2026-09-05 已完成）

| 组件 | 实施位置 | 实测结果 |
|------|---------|---------|
| 熔断器 | core/circuit_breaker.py（新增）：三态状态机、按主模型名进程级单例（LLMClient 每 Agent 独立实例化共 7 处，计数跨实例共享）、asyncio.Lock 保护、半开探测单次放行 | 单例缓存生效；连续失败 3 次断开日志正确；成功重置计数逻辑正确 |
| 路由模块 | core/model_router.py（新增）：`is_simple_question()` 启发式（长度阈值 30 + 复杂关键词黑名单）+ contextvar 路由开关（auto/small/big） | 简单/复杂问题判定正确，默认 auto |
| 模型选择 | core/llm_client.py：`__init__` 创建小模型客户端（BASE_URL/API_KEY 缺省复用主配置）、`_pick_route()` 按"路由 → 熔断状态"选模型、`chat()`/`chat_with_tools()` 接入计数；`chat_with_retry`/`chat_with_tools_retry` 重试策略一行未动 | 大模型调用成败实时计入熔断器（成功记录置于响应解析前，解析异常不误计失败）；未配置小模型时不创建客户端且不触碰熔断器 |
| 入口路由 | swarm/swarm_coordinator.py `process()` 入口：简单问题 set_route("small")，try/finally 恢复默认 | 路由仅影响模型选择，Agent 编排/审核/记忆流程不变 |

实测数据：

- 简单问题"感冒了怎么办"→ qwen-turbo，4.2 秒完成（对比大模型 18.1 秒，提速约 77%）
- 复杂问题 → qwen-plus 多 Agent 协作，行为与现状一致
- 主 Key 失效 + 小模型可用：同一次提问内连续失败 3 次熔断 → 自动降级 qwen-turbo → 10.5 秒返回 1547 字符真实回答（非静态致歉）
- 主 Key 运行时恢复 + 冷却期满（测试用 5 秒冷却）：半开探测 qwen-plus 成功 → 熔断器 open → closed（日志"大模型探测成功，熔断器恢复闭合"）
- 未配置小模型（默认态）：`client_small=None`，无模型路由日志，正常回答 785 字符，行为与现状完全一致
- `.env` 已从备份还原（MD5 哈希一致），测试配置（无效 Key/qwen-turbo/5 秒冷却）全部清除；py_compile 与 `import main` 冒烟测试通过

实施备注：Task 5 子代理对 swarm_coordinator.py 的导入行编辑曾被环境回退，验证阶段发现 NameError 后已手动补回 `from core.model_router import ...`，其余修改完整。

当前启用状态：`.env` 中 `LLM_SMALL_MODEL_NAME` 为空（新能力全部处于默认禁用态）。需要启用时填写小模型名（如 `qwen-turbo`，BASE_URL/API_KEY 留空复用主配置）并重启服务即可。

---

## 9. 对话流式显示（2026-09-05，已实施完成并验证）

### 9.1 需求与现状

需求：AI 回复以流式方式显示。

现状：Web 对话页 `respond()` 一次性 await 完整回答（核心链路 Swarm 编排 + LLM 合成约 10-20 秒）后整条追加到对话历史，期间用户面对无反馈空白。

### 9.2 设计原则

- **核心链路零改动**：`SwarmCoordinator.process()`、Agent Loop、LLM 调用、审核/指标逻辑全部不动，仍一次性返回完整答案
- **仅改前端展示层**：`web/chat_tab.py` 的 `respond()` 由普通 async 函数改为异步生成器（Gradio 原生支持 generator 流式更新 Chatbot）
- 不做 LLM token 级真流式：Swarm 多 Agent 并行 + 结果合成架构下，最终答案在 LeadAgent 合成阶段才生成，token 级流式需重构编排层，收益不匹配；采用"思考占位 + 分段打字机"方案

### 9.3 实施方案（web/chat_tab.py respond 函数）

1. 用户消息入历史后，立即追加一条 assistant 占位消息"🤔 正在检索医学知识库并分析您的问题，请稍候..."并 `yield`，界面即时反馈
2. await 既有的 `_process_question()` 获取完整回答（逻辑一行未动）
3. 答案按固定步长（8 字符）切片，逐段累加更新最后一条 assistant 消息的 content，每段间隔 0.03 秒 yield（约 800 字回答 3 秒播完）
4. 结束后强制以完整答案再 yield 一次，防止切片边界导致末尾遗漏
5. 流式结束后历史消息结构（`[{role, content}]`）与原实现完全一致，"标记有误""重置会话"不受影响

### 9.4 实施与验证结果（2026-09-05 已完成）

| 验证点 | 实测结果 |
|--------|---------|
| 思考占位 | 发送后约 2 秒内出现"🤔 正在检索..."提示，输入框与发送按钮正确禁用 |
| 打字机效果 | 答案开始出现后文字长度逐段递增（连续截图确认），非整块出现 |
| 内容完整性 | 最终回答完整：饮食建议 6 项 + 耗时 18.51s + 免责声明，无截断、无消息重复 |
| 状态恢复 | 流式结束后输入框/按钮正确恢复可用 |
| 回归影响 | 标记有误、重置会话、审核采样、飞轮指标记录行为不变；py_compile 通过 |

---

## 10. 优化方案（2026-09-07 项目分析报告改进建议，已实施完成并验证）

> 来源：项目分析报告第 7 节"潜在问题与改进建议"提出的 6 个方向，本章将其转化为可落地的实施方案。各方案相互独立、可分批实施；全部遵循本项目既有的兼容原则——**新能力默认禁用，未启用时行为与现状完全一致**。10.1-10.6 已于 2026-09-07 实施完成并通过 80 项自动化测试验证（10.5），详见 10.8 实施结果。

### 10.1 医疗合规与责任边界（优先级：高）

**问题**：AI 诊断涉及医疗安全，当前免责声明与高危提示属于软性提示（仅依赖 Agent Prompt 与既有免责声明正则校验），缺少"高危症状必须转人工/就医"的硬约束。误答高危急症（如胸痛、大出血）可能延误就医，存在合规与安全风险。

**方案**：

1. **高危症状硬约束规则**
   - `constraints/agent_constraints.yaml` 新增"高危症状硬约束"规则：当用户问题或回答中命中高危关键词（初版清单：胸闷、胸痛、呼吸困难、咯血、大出血、意识障碍、昏迷、抽搐、剧烈头痛、药物过量、自杀倾向等，清单可配置化放入同一 YAML）时，约束级别设为 **must**（阻断级），而非现有软性校验
   - 复用既有 `constraints/validator.py`（ConstraintValidator）执行校验；校验不通过时由 `validation/auto_fixer.py`（AutoFixer）将回答**强制改写**为"您描述的症状可能属于急症，请立即拨打 120 或前往最近的急诊科就诊"类标准话术，原回答仅作为参考附于结构化字段（不直接展示给用户）
2. **高危强制实时审核**
   - 高危关键词命中时强制走实时（阻塞）专家审核，复用 `review/review_trigger.py` 既有的 `high_risk` 触发原因，不新增触发机制
3. **前端免责声明固定展示**
   - `web/chat_tab.py` 对话页顶部固定展示免责声明横幅（"本系统提供的信息不构成医疗诊断建议，不能替代执业医师诊疗"），不随对话滚动消失

**预期**：高危急症场景下用户必然收到"立即就医"引导，且必须有专家介入；普通咨询行为不变。

**验证方式**：
- 输入"我父亲突然胸口剧痛伴呼吸困难"→ 回答为标准就医引导话术，审核页出现 high_risk 实时审核记录
- 输入普通咨询问题 → 行为与现状一致，无新增审核延迟
- 免责声明横幅在对话页始终可见

### 10.2 模型路由准确率提升（优先级：中）

**问题**：`core/model_router.py` 的 `is_simple_question()` 仅靠"长度阈值（默认 30）+ 复杂关键词黑名单"判断，处于临界区间的问题（短但复杂、长但简单）存在误判，误路由到大模型浪费成本、误路由到小模型损失质量。

**方案**：分级路由策略，不替换现有启发式：

1. **第一级（保留现状）**：现有启发式快速判定，明确简单（短且无复杂词）与明确复杂（含研究/分析/比较等关键词）的问题直接路由，零额外成本
2. **第二级（新增，可选）**：仅对**临界问题**（如长度落在 15~60 区间且不含复杂关键词）发起一次小模型轻量意图分类调用（单轮、限定输出 `simple|complex`），按分类结果路由
   - 环境变量 `ROUTE_LLM_CLASSIFY_ENABLED`（默认 `false`）与 `ROUTE_LLM_BOUNDARY_MIN/MAX`（默认 15/60）控制，未启用时行为与现状完全一致
   - 分类失败（超时/异常）时回退第一级启发式结果，不阻断主流程
3. **路由可观测**：每次路由判定输出日志（问题摘要 + 判定级别 + 最终路由 + 触发原因），便于后续统计误判率并迭代阈值

**预期**：临界问题路由准确率提升；启用分类后单次提问最多增加一次小模型调用（约 1 秒内）；默认关闭时零影响。

**验证方式**：
- 构造 20 条临界问题集（人工标注 simple/complex），对比启用前后的路由一致率
- 关闭开关 → 路由日志与现状完全一致
- 分类接口异常（模拟超时）→ 回退启发式，回答不中断

### 10.3 知识库多源导入与更新机制（优先级：中）

**问题**：知识库依赖 `knowledge/data/documents` 内置静态 txt 语料，量少且无更新流程，缺少权威医学资料（临床指南、文献）的自动入库机制，长期会导致检索质量受限、知识陈旧。

**方案**：新增统一入库管道 `knowledge/importer.py`：

1. **多格式解析**：支持 txt / markdown / pdf（复用 requirements 中已有的 pypdf），统一走既有 chunk（500/overlap 50）与 `_embed()` 入库链路，入库参数复用 milvus_kb 既有配置
2. **文档源清单管理**：新增 `knowledge/data/sources.yaml` 文档源清单，每个条目记录：来源名称、文件路径/URL、权威等级（指南/教材/科普）、版本与更新日期等元数据；元数据随 chunk 一并写入 Milvus 字段，检索结果可标注出处
3. **增量入库**：按文件内容 hash 去重，重复文档跳过，变更文档先删旧 chunk 再入库；提供 CLI 入口（`python -m knowledge.importer --sync`）
4. **入库质量门禁**：每次入库后自动抽样运行 `evaluation/rag_eval`，检索质量（相关性命中率）低于阈值时告警并暂停启用新语料（新 chunk 打标 `pending`，告警人工确认后转正）

**预期**：新增权威指南文档 30 分钟内完成解析、入库与质量验证；重复入库零冗余；检索结果可溯源。

**验证方式**：
- 放入一份新 PDF 指南 → `--sync` 后检索相关症状能命中该指南内容且标注出处
- 重复执行 `--sync` → 全部跳过，无重复 chunk
- 故意放入低质量文档 → rag_eval 抽样告警，新 chunk 处于 pending 不参与检索

### 10.4 飞轮指标维度下钻（优先级：中）

**问题**：`flywheel/metrics_tracker.py` 仅统计整体专家介入率、修正率等指标，无法定位薄弱环节——不知道哪类疾病、哪个 Skill、哪个 Agent 出错最多，飞轮改进缺少靶向依据。

**方案**：

1. **维度字段扩展**：指标记录结构（`record_*` 系列）新增可选维度字段：`disease_category`（疾病类别，可由 disease-code Skill 或关键词粗分类）、`skill_name`（本次回答实际调用的 Skill）、`agent_type`（咨询/诊断/研究）；字段缺省为 `unknown`，写入端（swarm_coordinator / review_queue）在既有记录点顺带填充，不新增采集链路
2. **聚合统计**：metrics_tracker 新增按维度聚合接口（各维度的介入率、修正率、平均审核耗时 Top-N）
3. **仪表盘下钻视图**：`web/metrics_tab.py` 飞轮指标仪表盘新增按维度聚合表格（"修正率最高的疾病类别 Top 10"、"被修正最多的 Skill"），辅助定位薄弱环节
4. **历史数据兼容**：旧记录无维度字段按 `unknown` 聚合，不迁移、不报错

**预期**：能直接回答"哪类问题最常被专家修正"，为知识库增强（kb_enhancer）与 Prompt 优化（prompt_optimizer）提供靶向输入。

**验证方式**：
- 制造多轮含 Skill 调用与专家修正的对话 → 仪表盘维度聚合表数据正确
- 仅有旧数据启动 → 聚合表正常显示（unknown 行），无报错
- 指标 JSON 持久化格式向后兼容，旧文件可直接加载

### 10.5 测试体系补齐（优先级：高）

**问题**：测试仅集中在 `examples/test_all.py`（依赖真实 LLM API 的端到端冒烟脚本），无单元/集成测试体系，回归成本高：每次改动（如第 6-9 章优化）只能靠手工验证，模型路由、熔断器、约束校验等纯逻辑模块完全无覆盖。

**方案**：新建 `tests/` 目录，与 examples 分工（examples 保留真实 API 冒烟，tests 面向 CI 可重复运行）：

1. **单元测试**（不依赖外部服务，monkeypatch/mock LLM 客户端）：
   - `tests/test_model_router.py`：`is_simple_question()` 边界用例（空文本、超长、复杂关键词、临界长度）、contextvar 路由继承
   - `tests/test_circuit_breaker.py`：三态状态机迁移（CLOSED→OPEN→HALF_OPEN→CLOSED）、阈值/冷却/成功重置、按模型名单例
   - `tests/test_constraint_validator.py`：各约束规则命中/未命中、AutoFixer 改写
   - `tests/test_review_queue.py`：入队、状态流转、原子写入持久化
2. **集成回归测试**（mock LLM 返回固定脚本）：`tests/test_swarm_pipeline.py` 覆盖 Swarm 编排主链路——简单问题单 Agent 路径、复杂问题分解并行路径、全 Worker 失败兜底、异常统一兜底（覆盖第 7 章各兜底项）
3. **医疗场景回归用例集**：`tests/cases/medical_regression.yaml` 固化 20+ 真实医疗问答场景（含高危症状场景，联动 10.1 验证），并将飞轮沉淀的评估用例（eval_expander 产物）定期导入该用例集
4. **工程化**：requirements.txt 增加 `pytest` 与 `pytest-asyncio`；提供 `pytest tests/ -v` 运行命令；单元测试目标耗时 < 30 秒

**预期**：核心纯逻辑模块 100% 用例覆盖，主链路改动可一键回归；高危场景进入自动化回归。

**验证方式**：
- `pytest tests/ -v` 全绿，耗时 < 30 秒，无需真实 API Key
- 人为在 model_router 引入 bug → 对应单元测试失败
- 第 6-9 章已实施的兜底行为均有对应集成断言

### 10.6 配置统一校验（优先级：低）

**问题**：`.env` 配置项多（LLM 主/小模型、熔断、路由、Milvus、Mem0、Redis 等，见第 8.2 节），各模块分散读取、静默回退默认值，误配置（如填错 Key、漏填依赖项）只在运行期以隐蔽方式暴露，排障成本高。

**方案**：新增 `core/config.py` 集中配置管理：

1. **pydantic Settings 统一建模**：将散落在各模块的 `os.getenv` 收敛为单一 Settings 类（llm、llm_small、circuit_breaker、route、milvus、memory 等分组），各模块改为从 Settings 读取（保持字段名与现有 .env 变量名不变，零迁移成本）
2. **启动时统一校验**：main.py 启动入口调用校验，输出可读错误清单，例如：
   - 缺失必填项：`LLM_API_KEY 未配置`
   - 冲突项：`ROUTE_LLM_CLASSIFY_ENABLED=true 但 LLM_SMALL_MODEL_NAME 为空`
   - 格式错误：`CIRCUIT_COOLDOWN_SECONDS=abc 不是有效整数`
   - 校验失败时快速失败（明确报错退出），替代现在的静默回退
3. **生效配置摘要**：启动时打印脱敏配置摘要（Key 仅显示前 4 位 + ***），方便确认运行环境
4. **渐进式收敛**：config.py 先只做"启动校验 + 摘要"（只读 .env，不改各模块读取逻辑），模块级迁移作为后续低风险重构分批进行

**预期**：误配置在启动 1 秒内以人话报错，而非运行期隐蔽故障；配置项一处可查。

**验证方式**：
- 清空 LLM_API_KEY 启动 → 明确报错"LLM_API_KEY 未配置"并退出
- 填入非法数字配置 → 启动时报格式错误
- 正常配置启动 → 打印脱敏摘要，系统行为与现状一致

### 10.7 实施优先级与依赖关系

| 优先级 | 方案 | 依赖 | 预估工作量 |
|--------|------|------|-----------|
| P0（高） | 10.1 医疗合规硬约束 | 无，复用既有 ConstraintValidator / review_trigger | 1-2 天 |
| P0（高） | 10.5 测试体系补齐 | 10.1 的回归验证建议在测试完成后补充高危用例 | 2-3 天 |
| P1（中） | 10.2 路由准确率提升 | 无（可选启用小模型分类） | 1 天 |
| P1（中） | 10.3 知识库导入更新 | 无，复用 pypdf 与 rag_eval | 2-3 天 |
| P1（中） | 10.4 飞轮维度下钻 | 建议在 10.5 之后实施，聚合逻辑可同步获得测试覆盖 | 1-2 天 |
| P2（低） | 10.6 配置统一校验 | 无，建议最后实施以覆盖前述新增配置项 | 1-2 天 |

实施原则：
- 各方案相互独立，可单独立项实施与验证
- 全部遵循"默认禁用 / 未启用时行为与现状完全一致"的兼容原则（同第 8 章）
- 每项实施完成后按第 7-9 章惯例在本章追加"实施与验证结果"小节

### 10.8 实施与验证结果（2026-09-07 已完成）

全部 6 项方案已实施，验证方式：`pytest tests/ -v`（80 项测试，耗时 < 1 秒，无需真实 API Key）、全量 py_compile、`import main` 与 Web 组件冒烟。

| 方案 | 实施位置 | 要点 |
|------|---------|------|
| 10.1 合规硬约束 | `constraints/agent_constraints.yaml`（emergency_constraints 段：16 个高危关键词 + 就医引导判定正则，均可配置）、`constraints/emergency.py`（新增：detect_emergency / has_urgent_care_guidance / emergency_response）、`constraints/validator.py`（validate_output 前置阻断级校验，对未配置约束的 Agent 同样生效）、`validation/auto_fixer.py`（emergency_guidance 强制改写）、`review/review_trigger.py`（条件 0：问题命中高危关键词 → 强制实时审核，复用 high_risk 原因；agent_loop 已传入 question）、`swarm/swarm_coordinator.py`（process 出口：高危问题且回答无就医引导且未经专家修正时，answer 强制替换为标准就医话术，原文保留在 reference_answer 字段）、`web/chat_tab.py`（对话页顶部固定免责声明横幅；高危检测统一走 emergency.py 配置） | 高危问题必然收到"立即拨打 120 / 急诊"引导；实时审核由 AgentLoop 强制触发，协调器只做兜底改写，不重复阻塞 |
| 10.2 路由二级分类 | `core/model_router.py`（新增 is_boundary_question / _classify_by_llm / classify_question：临界区间默认 15~60 字，小模型单轮分类限输出 simple/complex，5 秒超时；直接构建独立 AsyncOpenAI 客户端，不计入熔断器）、`swarm/swarm_coordinator.py`（process 入口改用 classify_question） | 环境变量 `ROUTE_LLM_CLASSIFY_ENABLED` 默认 false；关闭/失败/未配置小模型时行为与现状完全一致（测试覆盖三种回退路径） |
| 10.3 知识库导入管道 | `knowledge/importer.py`（新增：sources.yaml 清单解析、txt/md/pdf 解析、MD5 增量去重、删旧入库、入库后检索验证；CLI `python -m knowledge.importer --sync / --list`）、`knowledge/data/sources.yaml`（新增：清单模板含权威等级/版本元数据，默认条目 enabled=false）、`knowledge/milvus_kb.py`（新增 delete_by_source）、`knowledge/__init__.py`（MedicalKnowledgeBase 改懒加载，importer 与测试不再连带加载 pymilvus/嵌入模型） | 与方案的偏差：入库质量门禁未引入 LLM 评估（会引入 deepeval+LLM 依赖），改为"检索验证"——入库后用首个分块做语义检索，确认新语料可命中，未命中输出告警；pending 标记机制暂缓 |
| 10.4 飞轮维度下钻 | `flywheel/dimensions.py`（新增：9 类疾病关键词粗分类、build_review_dimensions、top_dimensions）、`flywheel/metrics_tracker.py`（维度统计随 record_review/approval/correction/rejection 可选传入，持久化于指标 JSON 的 dimensions 字段，get_dimension_breakdown 下钻接口）、`review/review_queue.py`（ReviewItem 增加 dimensions 字段，实时/异步入队透传，旧数据兼容 None）、`core/agent_loop.py`（结果附 skills_used）、`swarm/swarm_coordinator.py`（process 出口附加 review_dimensions）、`web/metrics_tab.py`（新增"维度下钻"区：按疾病类别/Skill/Agent 的审核数/修正数/修正率表，修正数降序 Top 10） | 维度缺失按 unknown 聚合，旧指标文件与旧审核记录直接兼容（均有测试覆盖） |
| 10.5 测试体系 | `tests/`（新增 10 个测试文件 + conftest.py，共 84 项）、`pytest.ini`（asyncio_mode=auto）、`requirements.txt`（pytest / pytest-asyncio） | 覆盖：model_router 启发式/contextvar/边界/分类回退、circuit_breaker 三态全迁移、validator 软+硬约束、emergency 检测与强制审核、AgentLoop 集成（高危强制审核/专家实时修正替换回答/skills_used/普通问题不审核，stub LLM 无真实 API）、review_queue 流程/实时唤醒/维度/持久化/旧格式、metrics 维度聚合/持久化/旧格式、Swarm 主链路兜底（全 Worker 失败/系统异常/硬约束改写/含引导不改写/普通问题不变）、importer 解析/hash 去重/缺文件/清单状态/缺 pymilvus 降级、config 校验规则。真实 LLM 冒烟仍由 examples/test_all.py 承担 |
| 10.6 配置统一校验 | `core/config.py`（新增：validate_config 校验必填/数值/布尔/逻辑冲突/边界，print_config_summary 脱敏摘要——Key 仅显示前 4 位）、`main.py`（启动入口接入：校验失败以可读清单退出，通过则打印摘要）、`.env.example`（补充 ROUTE_LLM_CLASSIFY_ENABLED / ROUTE_LLM_BOUNDARY_MIN/MAX 说明） | 校验项：LLM_API_KEY 必填、10 个数值项格式、3 个布尔项格式、`开启二级分类但未配小模型` 冲突、临界区间 min<max。渐进式第一步：只读校验，各模块 getenv 逻辑未动 |

实测结果：
- `pytest tests/`：84 passed，约 1.2 秒，无外部依赖；连跑两轮真实数据文件（审核队列/飞轮指标）MD5 不变（测试隔离完备）
- 全量 py_compile 通过；`import main`、Web 三 Tab 完整构建（`create_app()`）冒烟通过
- 端到端链路实测（stub LLM）：高危问题 → AgentLoop 强制实时审核（超时转异步）→ Agent 层自动补免责声明 → 协调器硬约束改写为"拨打 120"话术且原文保留在 reference_answer → 维度信息附加；专家在审核窗口内提交修正 → 回答被替换为专家修正内容；已含就医引导的高危回答与普通问题回答均不被误改
- `python main.py --cli`（无 Key）：可读错误"LLM_API_KEY 未配置"并退出码 1（旧行为是提问时隐晦报错）；配置占位 Key 后正常进入交互并打印脱敏摘要（Key 显示 sk-v***）
- `python -m knowledge.importer --list` 清单状态正确；本机未安装 pymilvus 时 `--sync` 对 enabled 条目返回明确的 failed 报告而非崩溃（真实 Milvus 入库需在依赖齐全的环境执行）
- 存量数据向后兼容：真实 review/data/pending_reviews.json（6 条旧记录）与 flywheel/data/flywheel_metrics.json（旧格式无 dimensions 字段）经新代码加载正常，旧记录按 unknown 维度聚合
- 行为兼容性：未配置小模型/未开启新开关时，路由、审核、飞轮、前端行为与实施前一致（回归测试断言覆盖）

验证过程中的附带修复（2026-09-07）：
1. **Gradio 6.x 兼容**：本机 gradio 6.26 已移除 `Chatbot(type=...)`/`show_copy_button` 参数，导致 `create_app()` 构建失败（属既有代码与新版 gradio 的不兼容）。`web/chat_tab.py` 改为按构造签名过滤参数，4.x-6.x 均可构建。遗留已知项：gradio 6 将 theme/css/js 迁至 launch()，当前仅产生警告、不影响功能，样式主题如需完全生效需后续调整 `web/app.py`
2. **测试数据隔离加固**：`tests/test_review_queue.py` 中触发专家批准的用例原先会经进程级指标单例把测试数据写入真实 flywheel_metrics.json，已改为模块级 autouse 隔离，并将被污染的真实数据文件还原至 HEAD 版本（恢复后 reviews=1/approvals=1，与提交一致）

---

## 11. 短期记忆 Redis 持久化接入与过期策略（2026-09-07，已实施完成并验证）

### 11.1 背景与问题

对短期记忆 Redis 持久化功能实测（16 项断言全过，功能本身正常）后发现两个接入层问题：

| # | 问题 | 影响 |
|---|------|------|
| 1 | Redis 后端只能通过构造函数 `ShortTermMemory(backend="redis")` 启用，主链路（chat_tab / examples）均为无参构造，`.env` 中无后端开关 | 配好 REDIS_HOST/PORT/DB 也切不到 Redis，功能实际不可达 |
| 2 | Redis 会话键 `stm:*` 写入无 TTL，且 Web 端"重置会话"仅更换 session_id 不清理旧键 | 长期运行会话键无限累积（泄漏式增长） |

### 11.2 实施方案

| 项 | 实施位置 | 内容 |
|----|---------|------|
| 后端环境变量开关 | `memory/short_term.py` | 构造参数 `backend` 改为可选（None），缺省读取 `MEMORY_BACKEND`（默认 memory）；主链路无参构造即可由 .env 切换。`__init__` 按解析后的 `self.backend` 判断是否初始化 Redis |
| 会话键滑动过期 | `memory/short_term.py` | 新增 `_redis_set()` 统一写入：`STM_SESSION_TTL_SECONDS`（默认 86400 秒=24 小时，0=永不过期）；每次追加消息刷新 TTL，活跃会话不过期、闲置会话自动回收 |
| 重置会话清理 | `web/chat_tab.py` `_reset_session()` | 更换 session_id 前调用 `clear_session(旧id)`，memory/redis 两后端统一释放旧会话数据（失败仅告警不影响重置） |
| 配置校验与摘要 | `core/config.py` | `MEMORY_BACKEND` 取值校验（memory/redis）、`STM_SESSION_TTL_SECONDS` 非负校验、`REDIS_PORT/DB` 数值校验；摘要新增"短期记忆后端 + TTL"行 |
| 配置模板 | `.env.example` | Redis 段补充 `MEMORY_BACKEND` / `STM_SESSION_TTL_SECONDS` 说明 |
| 单元测试 | `tests/test_short_term_memory.py`（13 项） | FakeRedis 桩验证：后端选择（默认/env/显式传参优先级）、读写与 OpenAI 格式、TTL 传递与 0=永不过期、跨实例持久化、clear、会话列举、不可达回退 memory |

### 11.3 实施与验证结果（本机真实 Redis，db 15 隔离）

| 验证点 | 实测结果 |
|--------|---------|
| MEMORY_BACKEND=redis 无参构造 | 主链路同款构造方式，backend=redis 且真实连接建立，写入落到 `stm:*` 键 |
| TTL 生效 | 写入后 `TTL=3600`（测试配 1 小时）；闲置 2 秒 TTL 递减，追加消息后重置回满（滑动过期确认） |
| 重置会话清理 | `_reset_session()` 后旧 session 的 stm 键从 Redis 消失，session_id 更换 |
| 跨进程持久化 | 新进程读回全部历史（含 TTL 继续倒计时） |
| 回退 | 连接不可达端口（127.0.0.1:1）时 backend 回退 memory，读写正常不中断 |
| 数据安全 | 测试全程 db 15 隔离，结束后键清空，db 0 与真实业务数据零污染 |
| 全量回归 | `pytest tests/` 98 passed（含新增 13 项记忆测试） |

### 11.4 实施中发现并修复的缺陷

**环境变量后端初始化失效（已修复）**：首轮实施将构造参数默认值改为 None 后，`__init__` 内仍以构造参数（而非解析后的 `self.backend`）判断是否执行 `_init_redis`——`MEMORY_BACKEND=redis` 时 backend 字符串为 redis 但 Redis 客户端为 None，数据静默落入进程内存且无任何报错（单测因手动注入客户端未拦截，实机验证拦截）。已修复判断依据，并补回归测试 `test_env_backend_redis_actually_inits_connection`（用不可达端口断言环境变量路径真正执行了连接初始化与失败回退）。
