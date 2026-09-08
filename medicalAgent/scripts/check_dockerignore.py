"""静态模拟 .dockerignore 规则,验证关键运行时文件的打包/排除去留"""
import fnmatch

PATTERNS = [
    line.strip()
    for line in open(".dockerignore", encoding="utf-8")
    if line.strip() and not line.startswith("#")
]


def excluded(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    for raw in PATTERNS:
        is_dir_rule = raw.endswith("/")
        p = raw.rstrip("/")
        if not p:
            continue
        # 目录型规则(review/data/):按路径前缀匹配,排除目录下全部内容
        if is_dir_rule and (rel == p or rel.startswith(p + "/")):
            return True
        # 文件/通配规则:匹配任意路径段或完整相对路径
        if any(fnmatch.fnmatch(part, p) for part in parts):
            return True
        if fnmatch.fnmatch(rel, p):
            return True
    return False


CHECKS = [
    (".claude/skills/search-knowledge/script/search.py", False, "技能脚本必须打包"),
    (".claude/skills/deep-research/script/research.py", False, "技能脚本必须打包"),
    ("knowledge/scripts/import_hardcoded_data.py", False, "知识库导入脚本"),
    ("knowledge/data/documents/01_lifestyle_hypertension.txt", False, "知识库源文档"),
    ("knowledge/data/sources.yaml", False, "知识库多源配置"),
    ("orchestrator/coordinator.py", False, "编排器代码"),
    ("agents/base_agent.py", False, "Agent代码"),
    ("web/app.py", False, "Web代码"),
    ("main.py", False, "入口"),
    ("requirements.txt", False, "依赖清单"),
    (".env", True, "密钥必须排除"),
    ("knowledge/data/milvus_lite.db", True, "向量库文件排除(经volume挂载)"),
    ("logs/medical_agent.log", True, "日志排除"),
    ("review/data/pending_reviews.json", True, "审核数据排除(经volume挂载)"),
]

if __name__ == "__main__":
    ok = True
    for path, should_excl, why in CHECKS:
        got = excluded(path)
        status = "OK" if got == should_excl else "!!MISMATCH!!"
        if got != should_excl:
            ok = False
        print(f"[{status}] {'排除' if got else '打包'} {path}  ({why})")
    print()
    print("全部正确" if ok else "存在不匹配")
    raise SystemExit(0 if ok else 1)
