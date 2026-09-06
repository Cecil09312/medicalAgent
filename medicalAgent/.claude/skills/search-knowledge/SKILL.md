---
name: search-knowledge
description: Search medical knowledge base for disease, symptom, and treatment information using Milvus vector database
---

# Search Knowledge Skill

搜索医学知识库，获取疾病、症状、治疗等相关信息。

## 功能

- 使用 Milvus 向量数据库进行语义搜索
- 支持多种医学知识类型
- 返回结构化的搜索结果

## 使用方式

```python
result = await search_knowledge(query="高血压的治疗方法", max_results=5)
```
