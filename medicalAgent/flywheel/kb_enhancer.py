"""
知识库增强
将专家修正作为高质量知识注入 Milvus 知识库
"""
from typing import Optional
from loguru import logger


class KnowledgeBaseEnhancer:
    """知识库增强器 - 将专家修正注入知识库"""

    def __init__(self, knowledge_base=None):
        """
        初始化

        Args:
            knowledge_base: MedicalKnowledgeBase 实例（可选，延迟初始化）
        """
        self.kb = knowledge_base
        self._total_ingested = 0

    def _get_kb(self):
        """获取知识库实例"""
        if self.kb is None:
            from knowledge.milvus_kb import MedicalKnowledgeBase
            self.kb = MedicalKnowledgeBase()
        return self.kb

    def ingest_correction(
        self,
        question: str,
        correction: str,
        original_answer: str = "",
        source: str = "flywheel",
    ) -> int:
        """
        将专家修正注入知识库

        Args:
            question: 用户问题
            correction: 专家修正后的回答
            original_answer: 原始（有问题的）回答
            source: 来源标记

        Returns:
            成功注入的文档块数
        """
        kb = self._get_kb()

        # 构造增强文本：包含问题和专家修正，提高检索相关性
        enhanced_text = f"问题：{question}\n\n专家回答：{correction}"

        if original_answer:
            enhanced_text += f"\n\n（注：此回答已经过专家审核修正，优先于系统自动生成的回答）"

        try:
            count = kb.add_documents(
                texts=[enhanced_text],
                doc_type="expert_reviewed",
                source=source,
            )
            self._total_ingested += count
            logger.info(f"专家修正已注入知识库: {count} 个文档块 (累计: {self._total_ingested})")
            return count
        except Exception as e:
            logger.error(f"知识库注入失败: {e}")
            return 0

    def batch_ingest(self, corrections: list) -> int:
        """
        批量注入专家修正

        Args:
            corrections: 修正列表，每项为 {"question": ..., "correction": ..., "original_answer": ...}

        Returns:
            成功注入的总数
        """
        total = 0
        for item in corrections:
            count = self.ingest_correction(
                question=item["question"],
                correction=item["correction"],
                original_answer=item.get("original_answer", ""),
            )
            total += count
        logger.info(f"批量注入完成: {total} 个文档块")
        return total

    @property
    def total_ingested(self) -> int:
        """累计注入数"""
        return self._total_ingested
