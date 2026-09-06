# 🏥 AI智能医疗诊断系统

多智能体协作的医疗助手，集咨询、诊断、研究为一体。基于 Swarm 多智能体框架 + RAG 知识库检索 + 专家审核闭环 + 飞轮指标优化。

---

## ✨ 核心特性

- **多智能体协作**：咨询 / 诊断 / 研究三类 Agent，由 LeadAgent 自动分解任务并汇总
- **两阶段 RAG 检索**：向量召回（bge-small-zh-v1.5）→ CrossEncoder 精排（bge-reranker-base）
- **专家审核闭环**：实时审核 / 异步抽样审核 / 用户标记纠错，支持批准、修正、拒绝
- **飞轮自我优化**：记录查询、审核、修正指标，驱动知识库增强与 Prompt 优化
- **长期记忆**：Mem0 + Redis 短期记忆，支持上下文连续对话
- **质量评估体系**：RAG / Agent / Swarm / 性能 四维评估，支持 LLM-as-Judge

---

## 🏗️ 系统架构

```mermaid
graph TB
    subgraph Web["Web 前端 (Gradio)"]
        CHAT[对话标签页]
        REVIEW[专家审核标签页]
        METRICS[飞轮指标标签页]
    end

    subgraph Swarm["Swarm 协作层"]
        COORD[SwarmCoordinator<br/>协调器]
        LEAD[LeadAgent<br/>任务分解]
        CTX[SharedContext<br/>共享上下文]
    end

    subgraph Agents["智能体层"]
        CA[咨询 Agent<br/>consultation]
        DA[诊断 Agent<br/>diagnostic]
        RA[研究 Agent<br/>research]
    end

    subgraph Core["核心层"]
        LOOP[AgentLoop<br/>工具调用循环]
        REG[SkillRegistry<br/>技能注册表]
        SM[StateManager<br/>状态管理]
        CB[CircuitBreaker<br/>熔断器]
        ROUTER[ModelRouter<br/>模型路由]
    end

    subgraph Knowledge["知识层"]
        KB[Milvus Lite<br/>向量数据库]
        RERANK[CrossEncoder<br/>重排序器]
    end

    subgraph Review["专家审核"]
        QUEUE[ReviewQueue<br/>审核队列]
        TRIGGER[ReviewTrigger<br/>触发策略]
    end

    subgraph Flywheel["飞轮优化"]
        TRK[MetricsTracker<br/>指标记录]
        ENH[KBEnhancer<br/>知识库增强]
        OPT[PromptOptimizer<br/>Prompt 优化]
    end

    CHAT --> COORD
    COORD --> LEAD
    LEAD --> CA & DA & RA
    CA & DA & RA --> LOOP
    LOOP --> REG
    LOOP --> KB
    KB --> RERANK
    COORD --> TRIGGER
    TRIGGER --> QUEUE
    QUEUE --> REVIEW
    COORD --> TRK
    TRK --> ENH
    TRK --> OPT
```

---

## 🔄 问答处理流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant W as Web对话页
    participant C as SwarmCoordinator
    participant L as LeadAgent
    participant A as 专业Agent
    participant K as 知识库RAG
    participant R as 审核系统

    U->>W: 输入医疗问题
    W->>C: process(question, session)

    alt 简单问题
        C->>A: 单Agent直接回答
    else 复杂问题
        C->>L: assess_and_decompose()
        L->>L: 任务分解 + Agent选择
        L-->>C: 子任务列表
        C->>A: 多Agent并行执行
    end

    A->>K: 两阶段检索(召回+精排)
    K-->>A: Top-K 相关知识
    A->>A: AgentLoop工具调用循环
    A-->>C: 各Agent结果
    C->>L: synthesize_results()
    L-->>C: 汇总回答

    C->>R: ReviewTrigger判定
    alt 实时审核开启且专家在线
        R->>R: 等待专家审核
        R-->>C: 审核结果(批准/修正/拒绝)
    else 异步抽样 or 用户标记
        R->>R: 入队待审核
    end

    C->>C: 记录飞轮指标
    C-->>W: 最终回答
    W-->>U: 流式展示回答
```

---

## 🔍 RAG 两阶段检索

```mermaid
flowchart LR
    Q[用户查询] --> EMB[bge-small-zh-v1.5<br/>嵌入向量]
    EMB --> RECALL[向量召回<br/>RERANK_CANDIDATE_K = 15]
    RECALL --> CAND[候选文档池 15条]
    CAND --> CE[CrossEncoder<br/>bge-reranker-base]
    CE --> RANK[按精排分降序]
    RANK --> TOP[返回 Top-K = 5]
    TOP --> ANS[LLM 基于上下文生成回答]

    style RECALL fill:#e1f5fe
    style CE fill:#fff3e0
    style TOP fill:#e8f5e9
```

检索开启两阶段时，先扩大向量召回候选池保召回率，再经 CrossEncoder 交叉注意力精排提升精度。

---

## 🤖 Swarm 多智能体协作

```mermaid
flowchart TD
    START([用户问题]) --> JUDGE{复杂度判断}
    JUDGE -->|简单| SINGLE[单 Agent 直出]
    JUDGE -->|复杂| DECOMP[LeadAgent 任务分解]

    DECOMP --> SUB1[子任务1: 健康咨询]
    DECOMP --> SUB2[子任务2: 症状诊断]
    DECOMP --> SUB3[子任务3: 指南研究]

    SUB1 --> CA[咨询 Agent]
    SUB2 --> DA[诊断 Agent]
    SUB3 --> RA[研究 Agent]

    CA & DA & RA --> SYNC[结果汇总]
    SINGLE & SYNC --> SYNTH[LeadAgent 综合合成]
    SYNTH --> ANS([最终回答])

    CA -.调用.-> S1[search_knowledge<br/>recommend_lifestyle]
    DA -.调用.-> S2[assess_risk<br/>analyze_symptoms]
    RA -.调用.-> S3[clinical_guideline<br/>deep_research]
```

---

## 🚀 快速开始

### 方式一：本地运行

```bash
# 1. 安装依赖（清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 3. 初始化知识库（首次运行，自动下载嵌入模型）
python -m knowledge.scripts.import_hardcoded_data

# 4. 启动 Web 服务
python main.py
```

访问 http://localhost:7860

### 方式二：Docker 部署（推荐，全量中国镜像）

```bash
# 构建并启动（知识库会自动初始化）
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

Docker 构建全程使用国内镜像：DaoCloud 基础镜像、清华 apt、清华 pip、hf-mirror 模型下载。

---

## 📂 目录结构

```
medicalAgent/
├── main.py                 # 入口（Web 默认 / --cli 命令行）
├── requirements.txt        # 依赖
├── Dockerfile              # 容器构建
├── docker-compose.yml      # 编排启动
├── docker-entrypoint.sh    # 启动脚本（自动初始化知识库）
├── .env                    # 环境配置（含密钥，不提交）
├── agents/                 # 专业 Agent（咨询/诊断/研究）
├── core/                   # 核心：AgentLoop、LLM、技能、状态、熔断
├── swarm/                  # 多智能体协作：协调器、LeadAgent、共享上下文
├── knowledge/              # 知识库：Milvus Lite + 重排序器
│   ├── data/documents/     # 医疗文档源
│   └── scripts/            # 文档导入脚本
├── review/                 # 专家审核：队列、触发、接口
├── flywheel/               # 飞轮：指标、知识库增强、Prompt 优化
├── memory/                 # 记忆：短期、长期、会话摘要
├── research/               # 深度研究工作流
├── constraints/            # Agent 约束校验（YAML）
├── validation/             # 输出自动修复
├── evaluation/             # 评估：RAG/Agent/Swarm/性能
└── web/                    # Gradio Web 界面
```

---

## ⚙️ 关键配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | - | LLM API Key（必填） |
| `LLM_MODEL_NAME` | qwen-plus | 主模型 |
| `KB_TOP_K` | 5 | 检索返回条数 |
| `RERANK_ENABLED` | false | 是否开启两阶段重排 |
| `RERANK_CANDIDATE_K` | 15 | 重排候选池大小 |
| `REVIEW_ENABLED` | true | 是否开启专家审核 |
| `REVIEW_REALTIME_ENABLED` | false | 实时审核（否则异步抽样） |
| `REVIEW_ASYNC_SAMPLE_RATE` | 0.05 | 异步抽样审核比例 |

完整配置见 `.env.example`。

---

## 📊 质量评估

```bash
# 全部维度
python -m evaluation.run_eval

# 单维度
python -m evaluation.run_eval -d rag        # RAG 检索质量
python -m evaluation.run_eval -d agent      # Agent 能力
python -m evaluation.run_eval -d swarm      # 协作质量
python -m evaluation.run_eval -d performance # 性能延迟
```

评估报告自动按时间戳保存到 `evaluation/data/reports/eval_report_YYYYMMDD_HHMMSS.json`。

> 注意：评估需访问 Milvus 知识库，运行评估前请先停止 Web 服务（Milvus Lite 为单进程文件锁）。

---

## 🔒 免责声明

本系统提供的信息仅供参考，不能替代专业医生的诊断。如有健康问题，请咨询专业医生。
