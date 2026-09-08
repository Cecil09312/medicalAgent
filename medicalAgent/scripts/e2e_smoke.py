"""端到端功能冒烟:真实 LLM 下验证核心链路(可分批运行)"""
import sys
import time

sys.path.insert(0, ".")


def report(name, ok, detail, elapsed):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} ({elapsed:.1f}s) {detail}", flush=True)
    return ok


def _build_stm():
    from memory.short_term import ShortTermMemory
    stm = ShortTermMemory()
    print(f"(短期记忆 backend={stm.backend})", flush=True)
    return stm


async def main(cases):
    from orchestrator import build_orchestrator

    # 多轮记忆需要显式注入短期记忆(Web 主链路同款组装)
    orch = build_orchestrator(short_term_memory=_build_stm())
    results = []

    if "simple" in cases:
        # 1. 简单问题:单 Agent + 小模型路由
        t = time.time()
        r = await orch.process("感冒了怎么办？", session_id="e2e-simple")
        ok = bool(r.get("answer")) and "agent_id" in r
        results.append(report("单Agent简单问题", ok,
                              f"agent={r.get('agent_id')} skills={r.get('skills_used')} "
                              f"len={len(str(r.get('answer') or ''))}", time.time() - t))

    if "complex" in cases:
        # 2. 复杂问题:多子任务并行 + 综合
        t = time.time()
        r = await orch.process(
            "我最近经常头晕头痛，请帮我分析可能的原因，并告诉我高血压的最新治疗指南和生活方式建议",
            session_id="e2e-complex",
        )
        summary = r.get("swarm_summary") or {}
        multi = summary.get("total_subtasks", 0) >= 2 or r.get("timeout_occurred")
        ok = bool(r.get("answer")) and multi
        results.append(report("多Agent并行协作", ok,
                              f"subtasks={summary.get('total_subtasks')} "
                              f"completed={summary.get('completed_subtasks')} "
                              f"timeout={r.get('timeout_occurred')} "
                              f"len={len(str(r.get('answer') or ''))}", time.time() - t))

    if "emergency" in cases:
        # 3. 高危问题:回答须含就医引导(模型自带或硬约束改写,二者皆合规)
        t = time.time()
        r = await orch.process("我父亲突然胸痛伴呼吸困难怎么办", session_id="e2e-emg")
        answer = str(r.get("answer") or "")
        ok = (r.get("warning") == "emergency_hard_constraint" and "120" in answer) or \
             ("120" in answer or "急诊" in answer)
        results.append(report("高危问题就医引导", ok,
                              f"warning={r.get('warning')} keyword={r.get('emergency_keyword')} "
                              f"含120/急诊={'120' in answer or '急诊' in answer}",
                              time.time() - t))

    if "memory" in cases:
        # 4. 多轮对话:短期记忆注入(注入 STM 后,第二轮应感知第一轮上下文)
        t = time.time()
        await orch.process("我对青霉素过敏", session_id="e2e-mem")
        r2 = await orch.process("我刚才说过我对什么药物过敏？请直接回答药物名", session_id="e2e-mem")
        answer = str(r2.get("answer") or "")
        ok = "青霉素" in answer
        results.append(report("多轮对话短期记忆", ok,
                              f"answer含青霉素={ok} len={len(answer)}", time.time() - t))

    if "shortcut" in cases:
        # 5. 便捷函数
        from agents.consultation_agent import consult
        t = time.time()
        r = await consult("熬夜后心悸怎么办")
        ok = bool(r.get("answer")) and r.get("agent_id") == "consultation_agent"
        results.append(report("consult() 便捷函数", ok,
                              f"agent={r.get('agent_id')} len={len(str(r.get('answer') or ''))}",
                              time.time() - t))

    if "rag" in cases:
        # 6. RAG:Agent 经 search_knowledge 检索知识库后作答
        t = time.time()
        r = await orch.process(
            "高血压的ICD-10编码是多少？请查询医学知识库后回答",
            session_id="e2e-rag",
        )
        answer = str(r.get("answer") or "")
        skills = r.get("skills_used") or []
        kb_skills = {"search_knowledge", "disease_code", "clinical_guideline"}
        ok = bool(set(skills) & kb_skills) and ("I10" in answer or "I10-I15" in answer)
        results.append(report("RAG 知识库检索", ok,
                              f"skills={skills} 答案含I10={'I10' in answer}", time.time() - t))

    print(f"\n=== {sum(results)}/{len(results)} passed ===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    import asyncio
    cases = sys.argv[1].split(",") if len(sys.argv) > 1 else ["simple", "complex"]
    sys.exit(asyncio.run(main(cases)))
