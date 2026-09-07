"""
医疗知识库模块

MedicalKnowledgeBase 采用懒加载（PEP 562）：
导入 milvus_kb 会连带加载 pymilvus 与嵌入模型依赖（较重），
轻量使用方（如 importer 的清单解析、单元测试）无需在 import 期承担该开销。
"""


def __getattr__(name):
    if name == "MedicalKnowledgeBase":
        from .milvus_kb import MedicalKnowledgeBase
        return MedicalKnowledgeBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'MedicalKnowledgeBase',
]
