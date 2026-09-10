#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/serve/evaluation_api.py
==========================
评估功能后端实现：数据质量评估 + BLEU 模型评估。

复用 src/data_quality_report.py 与 src/bleu_evaluation.py 的算法，
封装为可供 FastAPI 路由调用的同步函数（路由层用线程池执行）。
报告 JSON 落盘到 static/reports/，前端可直接按文件名回看历史报告。
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path

from ..data_quality_report import generate_quality_report
from ..bleu_evaluation import batch_evaluate, calculate_bleu_score

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS_DIR = STATIC_DIR / "reports"
DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

DATA_EXTENSIONS = (".csv", ".json", ".jsonl", ".txt")

# 与 src/build_dataset.py 一致的字段归一化候选
ASK_KEYS = ("ask", "input", "question", "问题", "ask_text", "query")
ANSWER_KEYS = ("answer", "output", "回答", "response")
DEPT_KEYS = ("department", "科室", "category", "type")
INSTRUCTION_PROMPT = "请回答医疗问题"


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_report(report: dict, prefix: str) -> str:
    """报告写入 static/reports/<prefix>_<timestamp>.json，返回文件名。"""
    _ensure_reports_dir()
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(REPORTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return filename


# ========================================
# 数据文件发现与加载
# ========================================

def list_data_files() -> list[dict]:
    """扫描 datasets/ 下可评估的数据文件（相对路径 + 大小 + 条数估算）。"""
    files = []
    if not DATASETS_DIR.is_dir():
        return files
    for path in sorted(DATASETS_DIR.rglob("*")):
        if path.suffix.lower() not in DATA_EXTENSIONS or not path.is_file():
            continue
        files.append({
            "path": path.relative_to(DATASETS_DIR).as_posix(),
            "size_kb": round(path.stat().st_size / 1024, 1),
        })
    return files


def _read_csv_rows(path: Path) -> list[dict]:
    """读 CSV，自动探测 utf-8 / utf-8-sig / gbk 编码。"""
    last_err: Exception | None = None
    # gb18030 是 gbk 的超集，几乎覆盖全部中文编码；最后用 utf-8 容错兜底
    for encoding, errors in (("utf-8-sig", "strict"), ("gb18030", "strict"), ("utf-8", "replace")):
        try:
            with open(path, "r", encoding=encoding, errors=errors, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(f"无法识别文件编码: {path.name} ({last_err})")


def _read_json_rows(path: Path) -> list[dict]:
    """读 JSON / JSONL 文件为 dict 列表。"""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # 尝试 JSONL（每行一个 JSON）
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def _pick(row: dict, keys: tuple) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def load_alpaca_data(file_path: str, sample_size: int | None = None) -> list[dict]:
    """
    加载数据文件并归一化为 Alpaca 风格:
        {instruction, input, output, department}
    """
    path = DATASETS_DIR / file_path
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw_rows = _read_csv_rows(path)
    elif suffix in (".json", ".jsonl"):
        raw_rows = _read_json_rows(path)
    else:  # .txt 等按 JSONL 尝试
        raw_rows = _read_json_rows(path)

    data = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        ask = _pick(row, ASK_KEYS)
        answer = _pick(row, ANSWER_KEYS)
        if not ask or not answer:
            continue
        item = {
            "instruction": _pick(row, ("instruction",)) or INSTRUCTION_PROMPT,
            "input": ask,
            "output": answer,
        }
        dept = _pick(row, DEPT_KEYS)
        if dept:
            item["department"] = dept
        data.append(item)

    if not data:
        raise RuntimeError(f"文件中没有可用的问答数据（需要 ask/question 与 answer 字段）: {file_path}")

    if sample_size and 0 < sample_size < len(data):
        data = random.sample(data, sample_size)
    return data


# ========================================
# 数据质量评估
# ========================================

def run_data_quality(file_path: str, sample_size: int | None = None) -> dict:
    """执行数据质量评估并保存报告。"""
    data = load_alpaca_data(file_path, sample_size)
    report = generate_quality_report(data, data_source=file_path)
    report["meta"]["sampled"] = len(data)
    if sample_size:
        report["meta"]["sample_size"] = sample_size
    report["report_file"] = _save_report(report, "data_quality")
    return report


# ========================================
# BLEU 模型评估
# ========================================

def _generate_answer(question: str, model_key: str) -> str:
    """调用推理模块流式生成，拼接为完整回答。"""
    from .inference import stream_chat  # 局部导入，避免评估接口触发模型相关模块加载

    parts = []
    for event in stream_chat(
        [{"role": "user", "content": question}], model_key=model_key
    ):
        if event[0] == "token":
            parts.append(event[1])
        elif event[0] == "error":
            raise RuntimeError(f"模型生成失败: {event[1]}")
    return "".join(parts).strip()


def run_bleu_evaluation(
    pairs: list[dict],
    use_inference: bool = False,
    model: str = "merged",
) -> dict:
    """
    BLEU 评估。

    pairs: [{question, reference, candidate?}]
        - use_inference=False: 使用传入的 candidate 作为模型回答
        - use_inference=True:  忽略 candidate，调用推理服务生成回答
    """
    if not pairs:
        raise ValueError("评估样本不能为空")

    references, candidates, questions = [], [], []
    for p in pairs:
        ref = (p.get("reference") or "").strip()
        if not ref:
            raise ValueError("存在参考答案为空的样本")
        references.append(ref)
        questions.append((p.get("question") or "").strip())
        if use_inference:
            candidates.append(_generate_answer(questions[-1], model))
        else:
            cand = (p.get("candidate") or "").strip()
            if not cand:
                raise ValueError("手动模式下每条样本都需要填写模型回答")
            candidates.append(cand)

    result = batch_evaluate(references, candidates)

    # 补充完整文本与问题，便于前端展示明细
    for i, detail in enumerate(result["details"]):
        detail["question"] = questions[i]
        detail["reference_full"] = references[i]
        detail["candidate_full"] = candidates[i]

    report = {
        "meta": {
            "mode": "inference" if use_inference else "manual",
            "model": model if use_inference else None,
            "total_pairs": len(pairs),
            "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": result["summary"],
        "details": result["details"],
    }
    report["report_file"] = _save_report(report, "bleu")
    return report


# ========================================
# 历史报告
# ========================================

def list_reports() -> list[dict]:
    """列出 static/reports/ 下的历史报告摘要。"""
    _ensure_reports_dir()
    items = []
    for path in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        kind = "data_quality" if path.name.startswith("data_quality") else "bleu"
        summary = {"file": path.name, "kind": kind, "time": path.stat().st_mtime}
        if kind == "data_quality":
            score = report.get("quality_score", {})
            summary["title"] = report.get("meta", {}).get("data_source", path.name)
            summary["score"] = score.get("total_score")
            summary["grade"] = score.get("grade", "")
        else:
            s = report.get("summary", {})
            summary["title"] = f"BLEU 评估 · {s.get('total_pairs', 0)} 条样本"
            summary["score"] = s.get("average_bleu")
            summary["grade"] = report.get("meta", {}).get("mode") == "inference" and "模型生成" or "手动对比"
        items.append(summary)
    return items


def load_report(filename: str) -> dict:
    """按文件名加载历史报告（仅允许 reports 目录内的文件）。"""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("非法文件名")
    path = REPORTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"报告不存在: {filename}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
