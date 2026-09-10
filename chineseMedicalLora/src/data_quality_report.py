# -*- coding: utf-8 -*-
"""
数据质量评估报告生成工具
课程：高质量微调数据工程与评估
功能：对微调数据集进行全面的质量评估，输出JSON格式的质量报告
"""

import os
import re
import json
import math
from collections import Counter, defaultdict
from datetime import datetime


# ========================================
# 统计维度评估
# ========================================

def analyze_text_length(texts, field_name="text"):
    """
    分析文本长度分布
    """
    lengths = [len(t) for t in texts if t]
    
    if not lengths:
        return {"error": "无有效数据"}
    
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    
    return {
        "field": field_name,
        "count": n,
        "min_length": lengths_sorted[0],
        "max_length": lengths_sorted[-1],
        "mean_length": round(sum(lengths) / n, 2),
        "median_length": lengths_sorted[n // 2],
        "p10": lengths_sorted[int(n * 0.1)],
        "p90": lengths_sorted[int(n * 0.9)],
        "very_short_count": sum(1 for l in lengths if l < 10),
        "very_long_count": sum(1 for l in lengths if l > 1000),
    }


def analyze_language_consistency(texts):
    """
    分析语言一致性（中文占比检测）
    """
    total = len(texts)
    if total == 0:
        return {"error": "无数据"}
    
    chinese_count = 0
    english_count = 0
    mixed_count = 0
    
    for text in texts:
        if not text:
            continue
        
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = cn_chars + en_chars
        
        if total_chars == 0:
            continue
        
        cn_ratio = cn_chars / total_chars
        if cn_ratio > 0.8:
            chinese_count += 1
        elif cn_ratio < 0.2:
            english_count += 1
        else:
            mixed_count += 1
    
    return {
        "total_samples": total,
        "chinese_dominant": chinese_count,
        "english_dominant": english_count,
        "mixed_language": mixed_count,
        "chinese_ratio": round(chinese_count / total, 4) if total > 0 else 0,
    }


def analyze_format_compliance(data, expected_format="alpaca"):
    """
    检查数据格式合规率
    """
    total = len(data)
    compliant = 0
    issues = Counter()
    
    for item in data:
        is_ok = True
        
        if expected_format == "alpaca":
            if "instruction" not in item:
                issues["缺少instruction字段"] += 1
                is_ok = False
            if "input" not in item and "question" not in item:
                issues["缺少input字段"] += 1
                is_ok = False
            if "output" not in item and "answer" not in item:
                issues["缺少output字段"] += 1
                is_ok = False
            
            output_val = item.get("output", item.get("answer", ""))
            if not output_val or not output_val.strip():
                issues["output为空"] += 1
                is_ok = False
                
        elif expected_format == "chat":
            if "messages" not in item:
                issues["缺少messages字段"] += 1
                is_ok = False
            else:
                roles = [m.get("role") for m in item["messages"]]
                if "user" not in roles:
                    issues["缺少user消息"] += 1
                    is_ok = False
                if "assistant" not in roles:
                    issues["缺少assistant消息"] += 1
                    is_ok = False
        
        if is_ok:
            compliant += 1
    
    return {
        "total": total,
        "compliant": compliant,
        "compliance_rate": round(compliant / total, 4) if total > 0 else 0,
        "issues": dict(issues.most_common(10)),
    }


def analyze_field_completeness(data):
    """
    分析字段完整性（空值比例）
    """
    if not data:
        return {"error": "无数据"}
    
    all_fields = set()
    for item in data:
        all_fields.update(item.keys())
    
    field_stats = {}
    total = len(data)
    
    for field in all_fields:
        filled = sum(1 for item in data if field in item and item[field] and str(item[field]).strip())
        field_stats[field] = {
            "filled": filled,
            "empty": total - filled,
            "fill_rate": round(filled / total, 4),
        }
    
    return field_stats


# ========================================
# 多样性分析
# ========================================

def analyze_category_distribution(data, category_field="department"):
    """
    分析类别分布（如科室分布）
    """
    categories = [item.get(category_field, "未知") for item in data if category_field in item]
    counter = Counter(categories)
    total = sum(counter.values())
    
    distribution = {}
    for cat, count in counter.most_common():
        distribution[cat] = {
            "count": count,
            "ratio": round(count / total, 4),
        }
    
    return {
        "total_categories": len(counter),
        "total_samples": total,
        "distribution": distribution,
    }


def analyze_duplicate_rate(texts):
    """
    分析文本重复率
    """
    total = len(texts)
    unique = len(set(texts))
    duplicates = total - unique
    
    return {
        "total": total,
        "unique": unique,
        "duplicates": duplicates,
        "duplicate_rate": round(duplicates / total, 4) if total > 0 else 0,
    }


def analyze_token_length_distribution(texts, estimate_ratio=1.5):
    """
    估算token长度分布
    对于中文，粗略估计 1个汉字 约 1.5 tokens
    """
    token_counts = []
    for text in texts:
        if not text:
            continue
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_words = len(re.findall(r'[a-zA-Z]+', text))
        estimated_tokens = int(cn_chars * estimate_ratio + other_words)
        token_counts.append(estimated_tokens)
    
    if not token_counts:
        return {"error": "无有效数据"}
    
    token_counts_sorted = sorted(token_counts)
    n = len(token_counts_sorted)
    
    return {
        "count": n,
        "min_tokens": token_counts_sorted[0],
        "max_tokens": token_counts_sorted[-1],
        "mean_tokens": round(sum(token_counts) / n, 2),
        "median_tokens": token_counts_sorted[n // 2],
        "over_512": sum(1 for t in token_counts if t > 512),
        "over_1024": sum(1 for t in token_counts if t > 1024),
        "over_2048": sum(1 for t in token_counts if t > 2048),
    }


# ========================================
# 综合质量评分
# ========================================

def calculate_quality_score(report):
    """
    根据各维度评估结果计算综合质量评分（0-100分）
    """
    scores = {}
    
    # 1. 格式合规性（满分20分）
    compliance_rate = report.get("format_compliance", {}).get("compliance_rate", 0)
    scores["format_score"] = round(compliance_rate * 20, 1)
    
    # 2. 字段完整性（满分20分）
    field_stats = report.get("field_completeness", {})
    if field_stats:
        avg_fill_rate = sum(v.get("fill_rate", 0) for v in field_stats.values()) / max(len(field_stats), 1)
        scores["completeness_score"] = round(avg_fill_rate * 20, 1)
    else:
        scores["completeness_score"] = 0
    
    # 3. 语言一致性（满分15分）
    cn_ratio = report.get("language_consistency", {}).get("chinese_ratio", 0)
    scores["language_score"] = round(cn_ratio * 15, 1)
    
    # 4. 重复率（满分15分，重复率越低分数越高）
    dup_rate = report.get("question_duplicates", {}).get("duplicate_rate", 1)
    scores["uniqueness_score"] = round((1 - dup_rate) * 15, 1)
    
    # 5. 长度合理性（满分15分）
    q_stats = report.get("question_length", {})
    a_stats = report.get("answer_length", {})
    short_q = q_stats.get("very_short_count", 0)
    total_q = q_stats.get("count", 1)
    short_a = a_stats.get("very_short_count", 0)
    total_a = a_stats.get("count", 1)
    short_ratio = (short_q + short_a) / (total_q + total_a) if (total_q + total_a) > 0 else 1
    scores["length_score"] = round((1 - short_ratio) * 15, 1)
    
    # 6. 多样性（满分15分）
    cat_info = report.get("category_distribution", {})
    num_cats = cat_info.get("total_categories", 1)
    max_cats = 6
    diversity = min(num_cats / max_cats, 1.0)
    scores["diversity_score"] = round(diversity * 15, 1)
    
    # 总分
    total_score = sum(scores.values())
    scores["total_score"] = round(total_score, 1)
    
    # 评级
    if total_score >= 90:
        scores["grade"] = "A (优秀)"
    elif total_score >= 75:
        scores["grade"] = "B (良好)"
    elif total_score >= 60:
        scores["grade"] = "C (合格)"
    else:
        scores["grade"] = "D (需改进)"
    
    return scores


# ========================================
# 生成完整报告
# ========================================

def generate_quality_report(data, data_source="unknown"):
    """
    生成完整的数据质量评估报告
    
    参数:
        data: list[dict], 训练数据列表
        data_source: str, 数据来源描述
    
    返回:
        dict: 质量报告
    """
    print(f"\n========== 生成数据质量报告 ==========")
    print(f"数据来源: {data_source}")
    print(f"数据量: {len(data)} 条")
    
    report = {
        "meta": {
            "data_source": data_source,
            "total_samples": len(data),
            "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    
    # 提取问题和回答文本
    questions = []
    answers = []
    for item in data:
        q = item.get("input", item.get("question", ""))
        a = item.get("output", item.get("answer", ""))
        if q:
            questions.append(q)
        if a:
            answers.append(a)
    
    # 1. 问题长度分析
    print("  分析问题长度分布...")
    report["question_length"] = analyze_text_length(questions, "question")
    
    # 2. 回答长度分析
    print("  分析回答长度分布...")
    report["answer_length"] = analyze_text_length(answers, "answer")
    
    # 3. 语言一致性
    print("  分析语言一致性...")
    report["language_consistency"] = analyze_language_consistency(questions + answers)
    
    # 4. 格式合规性
    print("  检查格式合规性...")
    report["format_compliance"] = analyze_format_compliance(data, "alpaca")
    
    # 5. 字段完整性
    print("  分析字段完整性...")
    report["field_completeness"] = analyze_field_completeness(data)
    
    # 6. 问题重复率
    print("  分析问题重复率...")
    report["question_duplicates"] = analyze_duplicate_rate(questions)
    
    # 7. 回答重复率
    print("  分析回答重复率...")
    report["answer_duplicates"] = analyze_duplicate_rate(answers)
    
    # 8. 类别分布
    print("  分析类别分布...")
    report["category_distribution"] = analyze_category_distribution(data)
    
    # 9. Token长度估算
    print("  估算Token长度分布...")
    combined_texts = [f"{q} {a}" for q, a in zip(questions, answers)]
    report["token_estimation"] = analyze_token_length_distribution(combined_texts)
    
    # 10. 综合质量评分
    print("  计算综合质量评分...")
    report["quality_score"] = calculate_quality_score(report)
    
    return report


def print_report_summary(report):
    """打印报告摘要"""
    print("\n" + "=" * 60)
    print("数据质量评估报告摘要")
    print("=" * 60)
    
    meta = report.get("meta", {})
    print(f"数据来源: {meta.get('data_source', '未知')}")
    print(f"数据量: {meta.get('total_samples', 0)} 条")
    print(f"报告时间: {meta.get('report_time', '')}")
    
    score = report.get("quality_score", {})
    print(f"\n--- 质量评分 ---")
    print(f"  格式合规: {score.get('format_score', 0)}/20")
    print(f"  字段完整: {score.get('completeness_score', 0)}/20")
    print(f"  语言一致: {score.get('language_score', 0)}/15")
    print(f"  数据唯一: {score.get('uniqueness_score', 0)}/15")
    print(f"  长度合理: {score.get('length_score', 0)}/15")
    print(f"  多样性:   {score.get('diversity_score', 0)}/15")
    print(f"  -----------")
    print(f"  总分: {score.get('total_score', 0)}/100")
    print(f"  评级: {score.get('grade', '未知')}")
    
    q_len = report.get("question_length", {})
    a_len = report.get("answer_length", {})
    print(f"\n--- 长度统计 ---")
    print(f"  问题: 平均{q_len.get('mean_length', 0)}字, 中位数{q_len.get('median_length', 0)}字")
    print(f"  回答: 平均{a_len.get('mean_length', 0)}字, 中位数{a_len.get('median_length', 0)}字")
    
    lang = report.get("language_consistency", {})
    print(f"\n--- 语言分布 ---")
    print(f"  中文为主: {lang.get('chinese_dominant', 0)} 条 ({lang.get('chinese_ratio', 0)*100:.1f}%)")
    
    dup_q = report.get("question_duplicates", {})
    print(f"\n--- 重复率 ---")
    print(f"  问题重复率: {dup_q.get('duplicate_rate', 0)*100:.1f}%")


# ========================================
# 主流程
# ========================================

def main():
    # 演示：使用示例数据生成报告
    sample_data = [
        {"instruction": "请回答医疗问题", "input": "我最近总是感觉头晕，应该怎么办？", "output": "头晕的原因很多，可能与低血糖、贫血、颈椎病等有关。建议先注意休息，保证充足睡眠，适当补充营养。如果持续不好转，建议到医院做详细检查。", "department": "内科"},
        {"instruction": "请回答医疗问题", "input": "感冒发烧应该吃什么药？", "output": "感冒发烧可以服用对乙酰氨基酚或布洛芬退烧，同时多喝温水，注意休息。如果体温持续超过38.5度超过3天，建议就医检查是否有细菌感染。", "department": "内科"},
        {"instruction": "请回答医疗问题", "input": "高血压患者需要注意什么？", "output": "高血压患者需要注意：1. 低盐饮食，每天盐摄入量控制在6克以下；2. 规律服药，不要随意停药；3. 适当运动，如散步、太极拳；4. 定期监测血压；5. 保持心情舒畅，避免情绪激动。", "department": "内科"},
        {"instruction": "请回答医疗问题", "input": "小孩发烧39度怎么处理？", "output": "孩子发烧39度属于高热，应立即给予退烧处理：1. 使用退烧药（布洛芬或对乙酰氨基酚，按体重计算剂量）；2. 物理降温（温水擦浴）；3. 多喝水防止脱水；4. 注意观察精神状态。如高热不退或出现抽搐，应立即就医。", "department": "儿科"},
        {"instruction": "请回答医疗问题", "input": "孕妇可以吃感冒药吗？", "output": "孕妇感冒用药需要特别谨慎，很多常见感冒药对胎儿有潜在影响。建议：1. 轻度感冒优先用物理方法（多休息、多喝水）；2. 如需用药必须在医生指导下使用；3. 孕早期用药风险最高；4. 禁用含麻黄碱、阿司匹林等成分的药物。", "department": "妇产科"},
    ]
    
    # 生成报告
    report = generate_quality_report(sample_data, data_source="医疗数据-演示样本")
    
    # 打印摘要
    print_report_summary(report)
    
    # 保存完整报告
    output_path = "data_quality_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {output_path}")


if __name__ == "__main__":
    main()
