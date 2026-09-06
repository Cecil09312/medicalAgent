"""
Deep Research - 深度研究技能
使用 DeepResearchWorkflow 进行深度医学研究
"""
from typing import Dict, Any, List
from loguru import logger

# 全局单例
_kb_instance = None


def get_knowledge_base():
    """获取知识库实例（懒加载）"""
    global _kb_instance
    if _kb_instance is None:
        try:
            from medical_knowledge_base import MedicalKnowledgeBase
            _kb_instance = MedicalKnowledgeBase()
        except ImportError:
            _kb_instance = None
    return _kb_instance


async def deep_research(query: str, max_iterations: int = 2) -> Dict[str, Any]:
    """
    深度研究医学问题

    Args:
        query: 研究问题
        max_iterations: 最大迭代次数

    Returns:
        研究结果
    """
    logger.info(f"Starting deep research: query='{query}', max_iterations={max_iterations}")

    try:
        # 尝试导入 DeepResearchWorkflow
        from deep_research_workflow import DeepResearchWorkflow

        workflow = DeepResearchWorkflow(max_iterations=max_iterations)
        research_result = await workflow.run(query)

        # 格式化报告
        answer = format_research_report(
            query=query,
            findings=research_result.get('findings', []),
            confidence=research_result.get('confidence', 0),
            sources=research_result.get('sources', [])
        )

        return {
            "answer": answer,
            "findings": research_result.get('findings', []),
            "confidence": research_result.get('confidence', 0),
            "sources": research_result.get('sources', []),
            "evidence_level": research_result.get('evidence_level', 'unknown'),
            "status": "completed"
        }

    except ImportError:
        logger.warning("DeepResearchWorkflow not available, using fallback")

        # 降级处理：使用知识库搜索
        kb = get_knowledge_base()
        if kb:
            try:
                results = await kb.search(query, max_results=5)
                findings = [{"summary": r.get('content', ''), "source": r.get('title', '')} for r in results]

                answer = format_research_report(
                    query=query,
                    findings=findings,
                    confidence=0.6,
                    sources=[r.get('title', '') for r in results]
                )

                return {
                    "answer": answer,
                    "findings": findings,
                    "confidence": 0.6,
                    "sources": [r.get('title', '') for r in results],
                    "evidence_level": "moderate",
                    "status": "completed_with_fallback"
                }
            except Exception as e:
                logger.error(f"Fallback research failed: {e}")

        return {
            "answer": f"深度研究功能暂不可用。查询: {query}",
            "findings": [],
            "confidence": 0,
            "sources": [],
            "evidence_level": "unknown",
            "status": "failed"
        }

    except Exception as e:
        logger.error(f"Deep research failed: {e}")
        return {
            "answer": f"研究失败: {str(e)}",
            "findings": [],
            "confidence": 0,
            "sources": [],
            "evidence_level": "unknown",
            "status": "error"
        }


def format_research_report(
    query: str,
    findings: List[Dict],
    confidence: float,
    sources: List[str]
) -> str:
    """
    格式化研究报告

    Args:
        query: 研究问题
        findings: 研究发现列表
        confidence: 置信度
        sources: 来源列表

    Returns:
        格式化后的报告
    """
    parts = [f"## 深度研究报告: {query}\n"]

    # 研究发现
    if findings:
        parts.append("### 主要发现\n")
        for i, finding in enumerate(findings, 1):
            summary = finding.get('summary', '无摘要')
            parts.append(f"{i}. {summary}")
    else:
        parts.append("### 主要发现\n\n未找到相关研究结果。")

    # 置信度
    confidence_pct = int(confidence * 100)
    parts.append(f"\n### 证据置信度: {confidence_pct}%")

    # 来源
    if sources:
        parts.append("\n### 参考来源\n")
        for i, source in enumerate(sources, 1):
            parts.append(f"{i}. {source}")

    return "\n".join(parts)
