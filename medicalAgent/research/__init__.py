"""
研究模块
提供网络搜索、证据合成、知识库和深度研究工作流
"""
from .web_search import WebSearchTool, SearchResult
from .evidence_synthesizer import EvidenceSynthesizer, ResearchReport
from .knowledge_base import KnowledgeBase, Document
from .deep_research_workflow import DeepResearchWorkflow

__all__ = [
    'WebSearchTool', 'SearchResult',
    'EvidenceSynthesizer', 'ResearchReport',
    'KnowledgeBase', 'Document',
    'DeepResearchWorkflow',
]
