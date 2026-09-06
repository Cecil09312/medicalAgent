---
name: disease-code
description: Query disease ICD-10 codes from medical classification database
---

# Disease Code Skill

查询疾病的 ICD-10 编码和分类信息。

## 功能

- 查询 ICD-10 疾病编码
- 获取疾病分类信息
- 使用 Milvus 向量数据库检索

## 使用方式

```python
result = await disease_code(disease_name="2型糖尿病")
```
