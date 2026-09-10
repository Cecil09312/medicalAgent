# -*- coding: utf-8 -*-
"""
BLEU自动评估工具
课程：高质量微调数据工程与评估
功能：使用BLEU分数评估模型生成文本的质量，支持中英文
依赖：pip install nltk jieba
"""

import re
import jieba
from collections import Counter


# ========================================
# BLEU评估核心实现
# ========================================

def tokenize_chinese(text):
    """
    中文分词
    使用jieba进行中文分词，英文按空格分词
    """
    tokens = list(jieba.cut(text))
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens


def tokenize_english(text):
    """英文分词"""
    return text.lower().split()


def auto_tokenize(text):
    """
    自动检测语言并分词
    包含中文字符 -> 使用中文分词
    否则 -> 英文分词
    """
    if re.search(r'[\u4e00-\u9fff]', text):
        return tokenize_chinese(text)
    else:
        return tokenize_english(text)


def get_ngrams(tokens, n):
    """提取n-gram"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def calculate_bleu_score(reference, candidate, max_n=4, weights=None):
    """
    计算BLEU分数
    
    参数:
        reference: str, 参考文本（标准答案）
        candidate: str, 候选文本（模型生成）
        max_n: int, 最大n-gram阶数
        weights: list[float], 各n-gram的权重，默认均匀分布
    
    返回:
        dict: 包含各阶BLEU分数和加权总分
    """
    ref_tokens = auto_tokenize(reference)
    cand_tokens = auto_tokenize(candidate)
    
    if not cand_tokens:
        return {
            "bleu_score": 0.0,
            "precisions": [0.0] * max_n,
            "brevity_penalty": 0.0,
            "ref_length": len(ref_tokens),
            "cand_length": 0,
        }
    
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # 计算各阶n-gram精度
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        cand_ngrams = Counter(get_ngrams(cand_tokens, n))
        
        # 裁剪计数
        clipped_count = 0
        total_count = 0
        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count
        
        # 平滑处理（避免零精度），使用+1平滑
        if total_count == 0:
            precision = 0.0
        else:
            precision = (clipped_count + 1) / (total_count + 1)
        
        precisions.append(precision)
    
    # 计算简短惩罚（Brevity Penalty）
    import math
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # 计算加权几何平均
    log_avg = 0.0
    for w, p in zip(weights, precisions):
        if p > 0:
            log_avg += w * math.log(p)
        else:
            log_avg += w * math.log(1e-10)
    
    bleu = bp * math.exp(log_avg)
    
    return {
        "bleu_score": round(bleu, 4),
        "precisions": [round(p, 4) for p in precisions],
        "brevity_penalty": round(bp, 4),
        "ref_length": ref_len,
        "cand_length": cand_len,
    }


def calculate_bleu_simple(reference, candidate):
    """
    简化版BLEU计算
    只使用1-gram和2-gram，适合短文本评估
    """
    return calculate_bleu_score(
        reference, candidate,
        max_n=2,
        weights=[0.5, 0.5]
    )


# ========================================
# 批量评估
# ========================================

def batch_evaluate(reference_answers, model_answers):
    """
    批量评估多组问答对
    
    参数:
        reference_answers: list[str], 参考答案列表
        model_answers: list[str], 模型回答列表
    
    返回:
        dict: 包含每条评估结果和总体统计
    """
    assert len(reference_answers) == len(model_answers), "参考答案和模型回答数量不一致"
    
    results = []
    scores = []
    
    for i, (ref, cand) in enumerate(zip(reference_answers, model_answers)):
        result = calculate_bleu_score(ref, cand)
        result["index"] = i
        result["reference"] = ref[:100] + "..." if len(ref) > 100 else ref
        result["candidate"] = cand[:100] + "..." if len(cand) > 100 else cand
        results.append(result)
        scores.append(result["bleu_score"])
    
    # 总体统计
    n = len(scores)
    avg_score = sum(scores) / n if n > 0 else 0
    sorted_scores = sorted(scores)
    
    summary = {
        "total_pairs": n,
        "average_bleu": round(avg_score, 4),
        "median_bleu": round(sorted_scores[n // 2], 4) if n > 0 else 0,
        "min_bleu": round(min(scores), 4) if scores else 0,
        "max_bleu": round(max(scores), 4) if scores else 0,
        "good_count": sum(1 for s in scores if s >= 0.5),
        "poor_count": sum(1 for s in scores if s < 0.2),
    }
    
    return {
        "summary": summary,
        "details": results,
    }


# ========================================
# 微调前后对比评估
# ========================================

def compare_model_outputs(questions, reference_answers, 
                          before_answers, after_answers):
    """
    对比微调前后的模型输出质量
    
    参数:
        questions: 问题列表
        reference_answers: 参考答案列表
        before_answers: 微调前的模型回答
        after_answers: 微调后的模型回答
    """
    print("=" * 60)
    print("微调前后模型输出对比评估")
    print("=" * 60)
    
    before_scores = []
    after_scores = []
    
    for i, (q, ref, before, after) in enumerate(
        zip(questions, reference_answers, before_answers, after_answers)):
        
        before_result = calculate_bleu_simple(ref, before)
        after_result = calculate_bleu_simple(ref, after)
        
        before_scores.append(before_result["bleu_score"])
        after_scores.append(after_result["bleu_score"])
        
        print(f"\n--- 问题 {i+1} ---")
        print(f"  问题: {q}")
        print(f"  参考答案: {ref[:80]}...")
        print(f"  微调前BLEU: {before_result['bleu_score']:.4f}")
        print(f"  微调后BLEU: {after_result['bleu_score']:.4f}")
        
        improvement = after_result["bleu_score"] - before_result["bleu_score"]
        if improvement > 0:
            print(f"  提升: +{improvement:.4f}")
        else:
            print(f"  变化: {improvement:.4f}")
    
    # 总结
    avg_before = sum(before_scores) / len(before_scores) if before_scores else 0
    avg_after = sum(after_scores) / len(after_scores) if after_scores else 0
    
    print(f"\n{'='*60}")
    print(f"总体评估:")
    print(f"  微调前平均BLEU: {avg_before:.4f}")
    print(f"  微调后平均BLEU: {avg_after:.4f}")
    print(f"  平均提升: {avg_after - avg_before:+.4f}")
    print(f"  提升比例: {(avg_after - avg_before) / max(avg_before, 0.001) * 100:+.1f}%")
    
    return {
        "before_avg": round(avg_before, 4),
        "after_avg": round(avg_after, 4),
        "improvement": round(avg_after - avg_before, 4),
    }


# ========================================
# 主流程 - 演示
# ========================================

def main():
    print("=" * 60)
    print("BLEU评估工具 - 演示")
    print("=" * 60)
    
    # 演示1：单条评估
    print("\n--- 演示1：单条中文评估 ---")
    ref = "头晕的原因很多，可能与低血糖、贫血、颈椎病等有关。建议先注意休息，保证充足睡眠。"
    cand = "头晕可能是因为低血糖或贫血引起的，建议多休息，保证睡眠充足。"
    
    result = calculate_bleu_score(ref, cand)
    print(f"参考: {ref}")
    print(f"候选: {cand}")
    print(f"BLEU分数: {result['bleu_score']}")
    print(f"各阶精度: {result['precisions']}")
    print(f"简短惩罚: {result['brevity_penalty']}")
    
    # 演示2：简化版评估
    print("\n--- 演示2：简化版评估 ---")
    result2 = calculate_bleu_simple(ref, cand)
    print(f"简化BLEU: {result2['bleu_score']}")
    
    # 演示3：微调前后对比
    print("\n--- 演示3：微调前后对比 ---")
    questions = [
        "感冒发烧怎么办？",
        "高血压需要注意什么？",
    ]
    references = [
        "感冒发烧可以服用退烧药，如对乙酰氨基酚或布洛芬，同时多喝水多休息。",
        "高血压患者需要低盐饮食，规律服药，适当运动，定期监测血压。",
    ]
    before_finetune = [
        "I'm sorry, I can only provide general advice. Please consult a doctor.",
        "High blood pressure requires medical attention. Please see your doctor.",
    ]
    after_finetune = [
        "感冒发烧建议多喝水、多休息，如果体温超过38.5度可以服用退烧药物如布洛芬。",
        "高血压患者要注意：低盐饮食，按时吃药，适量运动，定期量血压。",
    ]
    
    compare_model_outputs(questions, references, before_finetune, after_finetune)
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()
