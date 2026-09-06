# RAG 检索重排序 - 独立审查

- [x] CP-R1: 关闭态零影响
  - **Type**: `rule`
  - **Covers**: AC-1 / TR-2.2 / TR-2.4
  - **检查点**：RERANK_ENABLED 默认 false；关闭时 search() 召回 limit=top_k、返回无 rerank_score 键、knowledge.reranker 模块不被导入；既有代码路径完整保留
  - **Evidence**: PASS。代码：milvus_kb.py L41 默认 false、L347 关闭时 limit=top_k、L368 精排段在 `if RERANK_ENABLED and hits:` 内懒导入；hits 构造为既有逻辑。运行（临时副本库）：RERANK_ENABLED=False、results count=5、reranker module loaded=False、无 rerank_score 键（返回键 doc_type/page/score/source/text）、SMOKE_CLOSED_OK

- [x] CP-R2: 开启态两阶段检索正确
  - **Type**: `rule`
  - **Covers**: AC-2 / TR-2.3
  - **检查点**：开启时 Milvus 召回 max(top_k, RERANK_CANDIDATE_K) 条；CrossEncoder 精排后返回 top_k；每条含 rerank_score 且降序；doc_type 过滤在 Milvus 层生效；search_by_text 透传精排
  - **Evidence**: PASS。假模块注入验证：Milvus 层实召回 10、reranker 见候选 10、返回 5 条全部含 rerank_score 且降序（scores [500,500,500,500,499]）；doc_type=clinical_guideline 过滤在 Milvus filter 层完成、返回 1 条类型正确；search_by_text(top_k=3) 返回 3 条含分。真实模型附加验证：本机 HF 缓存已有 bge-reranker-base，离线加载（17.9s）后对"糖尿病饮食"查询相关文档打分 0.9974/0.9899、无关文档被挤出 top2，语义排序正确；SMOKE_ENABLED_OK

- [x] CP-R3: Reranker 单例与缓存逻辑
  - **Type**: `rule`
  - **Covers**: AC-3 / TR-1.1 / TR-1.2
  - **检查点**：reranker.py 单例（__new__ + Lock 双重检查）；get_default_reranker 单例入口；缓存检测复用 HF 路径逻辑；已缓存 local_files_only+离线标志，未缓存显式允许下载；py_compile 通过
  - **Evidence**: PASS。py_compile PY_COMPILE_OK；单例两次获取同一实例、真实模型第二次调用 0.0s（只加载一次）；rerank 排序/截断/空输入正确（['丙丙丙','乙乙'] scores [3.0,2.0]；空输入 []）；缓存分支：已缓存 HF_HUB_OFFLINE=1 离线加载成功，伪造未缓存模型名时 HF_HUB_OFFLINE=0（UNCACHED_BRANCH_OK）

- [x] CP-R4: 调用方零改动与配置完备
  - **Type**: `rule`
  - **Covers**: AC-4 / TR-3.1 / TR-3.2 / TR-3.3
  - **检查点**：三个 Skill 脚本、research/deep_research_workflow.py、evaluation/rag_eval.py 无改动；.env 与 .env.example 含三配置项且默认值正确；chat_tab 预热段有开关守卫；requirements.txt 无新增依赖
  - **Evidence**: PASS。全仓搜 rerank/RERANK 仅命中 reranker.py、milvus_kb.py、chat_tab.py；调用方文件时间戳均早于改动窗口；调用方经 kb.search/search_by_text 统一入口自动受益、新增字段为纯附加；.env.example L34-41 与 .env L33-40 配置齐备（false / bge-reranker-base / 10）；chat_tab L106-109 env 守卫且预热时 .env 已加载；requirements.txt 未改，sentence-transformers==5.3.0 已含 CrossEncoder

- [x] CP-U1: 代码质量与风格一致性
  - **Type**: `rubric`
  - **Covers**: AC-5 / TR-4.2
  - **Scale**: 1-5
  - **Anchors**: 1 = 破坏既有逻辑或风格冲突明显；3 = 功能正确但风格/注释与既有代码不一致；5 = 严格保留既有分支、单例/懒加载/日志风格与 milvus_kb 一致、中文注释清晰无 emoji 无乱码
  - **Pass Threshold**: >= 4
  - **Evidence**: 评分 **5/5**，PASS。单例/类锁/懒加载/loguru 中文日志/docstring/模块级 env 常量与 milvus_kb 风格严格一致；三文件 UTF-8 无乱码；新增代码段无 emoji；改动均为纯新增（常量+末尾分支+预热段），无既有分支删除

## Review History

### Review R1
- **Result**: `pass`
- **Evidence**:
  - 独立审查者（新鲜上下文）完成代码阅读 + 独立运行验证（未采信实施者自述）
  - 5 个检查点全部 PASS；假模块注入与真实 bge-reranker-base 模型双重验证两阶段检索
  - 审查中运行的命令：py_compile（PY_COMPILE_OK）、关闭态冒烟（SMOKE_CLOSED_OK）、开启态假模块验证（SMOKE_ENABLED_OK）、单例/缓存分支验证（UNCACHED_BRANCH_OK）、真实模型语义排序验证
  - 临时数据库副本审查后已清理（Test-Path = False），审查未修改任何文件
- **Findings**（均为 advisory，不阻塞）：
  1. [低] get_default_reranker() 函数体自身无锁，但 __init__ 类锁 + _initialized 双重检查保证模型只加载一次、返回同一实例，与 MedicalKnowledgeBase 并发语义等效
  2. [信息] RERANK_CANDIDATE_K 解析失败回退 10 属配置容错；search() 内对 reranker 异常无捕获、正常上抛由顶层兜底处理，符合"不新增 fallback"约束
  3. [信息] chat_tab 预热线程复用既有 try/except，reranker 预热失败仅 warning；检索主路径无此兜底
  4. [信息] reranker.py 复用 milvus_kb 私有函数 _detect_model_cached 为 spec 明确要求，且仅在开启时懒导入
