"""
统一评估入口
支持按维度运行或全部运行，输出 console / json 格式报告
"""
import asyncio
import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from .config import EVAL_CONFIG, DATA_DIR
from .datasets import (
    load_rag_test_cases,
    load_agent_test_cases,
    load_swarm_test_cases,
)
from .rag_eval import RAGEvaluator, RAGEvalResult
from .agent_eval import AgentEvaluator, AgentEvalResult
from .swarm_eval import SwarmEvaluator, SwarmEvalResult
from .performance_eval import PerformanceEvaluator, PerfEvalResult


VALID_DIMENSIONS = ["rag", "agent", "swarm", "performance", "all"]


async def run_evaluation(
    dimensions: Optional[List[str]] = None,
    report_format: str = "console",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行评估

    Args:
        dimensions: 评估维度列表，可选: rag/agent/swarm/performance/all
                    默认 ["all"]
        report_format: 输出格式 "console" 或 "json"
        output_path: JSON 报告输出路径（可选，不指定时自动按时间戳保存到
                     evaluation/data/reports/eval_report_YYYYMMDD_HHMMSS.json）

    Returns:
        评估结果字典
    """
    if dimensions is None:
        dimensions = ["all"]

    # 验证维度
    for dim in dimensions:
        if dim not in VALID_DIMENSIONS:
            raise ValueError(f"未知评估维度: {dim}，可选: {VALID_DIMENSIONS}")

    if "all" in dimensions:
        dimensions = ["rag", "agent", "swarm", "performance"]

    logger.info(f"开始评估，维度: {dimensions}")
    overall_start = time.time()
    results = {}

    # RAG 评估
    if "rag" in dimensions:
        logger.info("=" * 60)
        logger.info("运行 RAG 知识库评估")
        try:
            rag_evaluator = RAGEvaluator()
            rag_cases = load_rag_test_cases()
            rag_result = await rag_evaluator.evaluate(rag_cases)
            results["rag"] = rag_result
            logger.info(rag_result.summary())
        except Exception as e:
            logger.error(f"RAG 评估失败: {e}")
            results["rag"] = RAGEvalResult(passed=False)

    # Agent 评估
    if "agent" in dimensions:
        logger.info("=" * 60)
        logger.info("运行 Agent 能力评估")
        try:
            agent_evaluator = AgentEvaluator()
            agent_cases = load_agent_test_cases()
            agent_result = await agent_evaluator.evaluate(agent_cases)
            results["agent"] = agent_result
            logger.info(agent_result.summary())
        except Exception as e:
            logger.error(f"Agent 评估失败: {e}")
            results["agent"] = AgentEvalResult(passed=False)

    # Swarm 评估
    if "swarm" in dimensions:
        logger.info("=" * 60)
        logger.info("运行 Swarm 协作评估")
        try:
            swarm_evaluator = SwarmEvaluator()
            swarm_cases = load_swarm_test_cases()
            swarm_result = await swarm_evaluator.evaluate(swarm_cases)
            results["swarm"] = swarm_result
            logger.info(swarm_result.summary())
        except Exception as e:
            logger.error(f"Swarm 评估失败: {e}")
            results["swarm"] = SwarmEvalResult(passed=False)

    # 性能评估
    if "performance" in dimensions:
        logger.info("=" * 60)
        logger.info("运行端到端性能评估")
        try:
            perf_evaluator = PerformanceEvaluator()
            # 从各维度测试用例中提取问题列表
            perf_cases = []
            try:
                for tc in load_rag_test_cases():
                    perf_cases.append({"question": tc.question})
                for tc in load_agent_test_cases():
                    perf_cases.append({"question": tc.question})
            except Exception:
                perf_cases = [{"question": "高血压患者应该注意什么？"}]

            perf_result = await perf_evaluator.evaluate(perf_cases, iterations=2)
            results["performance"] = perf_result
            logger.info(perf_result.summary())
        except Exception as e:
            logger.error(f"性能评估失败: {e}")
            results["performance"] = PerfEvalResult(passed=False)

    overall_elapsed = time.time() - overall_start

    # 生成报告
    report = _generate_json_report(results, overall_elapsed)

    # 报告落盘：显式指定 output_path 时使用指定路径；
    # 否则自动按时间戳保存到 evaluation/data/reports/ 目录，避免报告随控制台关闭丢失
    if output_path:
        save_path = output_path
    else:
        save_path = str(_auto_report_path())

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 报告已保存: {save_path}")

    if report_format == "console":
        _print_console_report(results, overall_elapsed)

    return report


def _auto_report_path() -> Path:
    """
    生成自动保存的报告路径（按当前时间戳命名）

    Returns:
        报告文件路径: evaluation/data/reports/eval_report_YYYYMMDD_HHMMSS.json
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return DATA_DIR / "reports" / f"eval_report_{timestamp}.json"


def _print_console_report(results: Dict[str, Any], elapsed: float):
    """打印终端报告"""
    print("\n" + "=" * 70)
    print("  MediX 医疗Agent 评估报告")
    print("=" * 70)

    for dim_name, result in results.items():
        print(f"\n--- {dim_name.upper()} ---")
        if hasattr(result, "summary"):
            print(result.summary())
        else:
            print(f"  状态: {result}")

    # 总体结论
    all_passed = all(getattr(r, "passed", False) for r in results.values())
    print("\n" + "=" * 70)
    print(f"  总体结果: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    print(f"  总耗时: {elapsed:.1f}s")
    print("=" * 70)

    # 不达标项警告
    warnings = []
    if "rag" in results and not results["rag"].passed:
        warnings.append("RAG 评估未达标")
    if "agent" in results and not results["agent"].passed:
        warnings.append("Agent 评估未达标")
    if "swarm" in results and not results["swarm"].passed:
        warnings.append("Swarm 评估未达标")
    if "performance" in results and not results["performance"].passed:
        warnings.append("性能评估未达标")

    if warnings:
        print("\n  警告:")
        for w in warnings:
            print(f"    - {w}")


def _generate_json_report(results: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    """生成 JSON 报告"""
    report = {
        "total_elapsed": elapsed,
        "dimensions": {},
        "overall_passed": all(getattr(r, "passed", False) for r in results.values()),
    }

    for dim_name, result in results.items():
        dim_report = {"passed": getattr(result, "passed", False)}

        if isinstance(result, RAGEvalResult):
            dim_report.update({
                "total_cases": result.total_cases,
                "avg_faithfulness": result.avg_faithfulness,
                "avg_relevancy": result.avg_relevancy,
                "avg_context_precision": result.avg_context_precision,
                "avg_context_recall": result.avg_context_recall,
                "avg_no_hallucination": result.avg_no_hallucination,
            })
        elif isinstance(result, AgentEvalResult):
            dim_report.update({
                "agent_id": result.agent_id,
                "total_cases": result.total_cases,
                "avg_tool_accuracy": result.avg_tool_accuracy,
                "avg_answer_correctness": result.avg_answer_correctness,
                "avg_task_completion": result.avg_task_completion,
                "avg_response_quality": result.avg_response_quality,
            })
        elif isinstance(result, SwarmEvalResult):
            dim_report.update({
                "total_cases": result.total_cases,
                "avg_decomposition_quality": result.avg_decomposition_quality,
                "avg_agent_selection_accuracy": result.avg_agent_selection_accuracy,
                "avg_coordination_efficiency": result.avg_coordination_efficiency,
                "avg_synthesis_quality": result.avg_synthesis_quality,
                "avg_coordination_time": result.avg_coordination_time,
            })
        elif isinstance(result, PerfEvalResult):
            dim_report.update({
                "total_queries": result.total_queries,
                "latency_p50": result.latency_p50,
                "latency_p95": result.latency_p95,
                "latency_p99": result.latency_p99,
                "avg_tokens_total": result.avg_tokens_total,
                "throughput_qpm": result.throughput_qpm,
            })

        report["dimensions"][dim_name] = dim_report

    return report


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="MediX 医疗Agent 评估工具")
    parser.add_argument(
        "-d", "--dimensions",
        nargs="+",
        default=["all"],
        choices=VALID_DIMENSIONS,
        help="评估维度 (默认: all)",
    )
    parser.add_argument(
        "-f", "--format",
        default="console",
        choices=["console", "json"],
        help="输出格式 (默认: console)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="JSON 报告输出路径（不指定时自动按时间戳保存到 evaluation/data/reports/）",
    )

    args = parser.parse_args()

    asyncio.run(run_evaluation(
        dimensions=args.dimensions,
        report_format=args.format,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
