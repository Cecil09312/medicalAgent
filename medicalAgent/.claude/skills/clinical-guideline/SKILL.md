---
name: clinical-guideline
description: Query clinical guidelines from medical database with organization and year info
---

# Clinical Guideline Skill

查询临床指南，获取权威医学指导。

## 功能

- 查询临床指南
- 获取发布组织和年份信息
- 使用 Milvus 向量数据库检索

## 使用方式

```python
result = await clinical_guideline(query="高血压治疗指南", max_results=1)
```
