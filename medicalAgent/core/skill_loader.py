"""
Skill Loader 辅助函数
用于动态加载 .claude/skills 目录下的 Skill 函数（自动发现）
"""
from pathlib import Path
import importlib.util
from typing import Callable, Dict, List, Optional
import yaml
import os
from loguru import logger


def load_skill_function(skill_name: str, script_name: str, function_name: str, project_root: Path = None) -> Callable:
    """
    动态加载 Skill 函数

    Args:
        skill_name: Skill 目录名（如 "search-knowledge"）
        script_name: Python 脚本名（如 "search"）
        function_name: 函数名（如 "search_knowledge"）
        project_root: 项目根目录

    Returns:
        Skill 函数
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    skills_dir = project_root / ".claude" / "skills"
    module_path = skills_dir / skill_name / "script" / f"{script_name}.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Skill module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"skill_{skill_name.replace('-', '_')}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {module_path}")

    return getattr(module, function_name)


def parse_skill_md(file_path: Path) -> Optional[Dict]:
    """
    解析 SKILL.md 文件的 YAML frontmatter

    Args:
        file_path: SKILL.md 文件路径

    Returns:
        解析后的 frontmatter 字典，或 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if content.startswith('---'):
            end_idx = content.find('---', 3)
            if end_idx != -1:
                yaml_content = content[3:end_idx]
                try:
                    data = yaml.safe_load(yaml_content)
                    return data
                except yaml.YAMLError as e:
                    logger.warning(f"Error parsing YAML in {file_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return None


def discover_skills(project_root: Path = None) -> List[Dict]:
    """
    自动扫描 .claude/skills 目录，发现所有 skills

    Args:
        project_root: 项目根目录

    Returns:
        发现的 skills 列表
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    skills_dir = project_root / ".claude" / "skills"

    if not skills_dir.exists():
        logger.warning(f"Skills directory not found: {skills_dir}")
        return []

    discovered_skills = []

    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue

        skill_name = skill_dir.name

        # 查找 SKILL.md
        skill_md_path = None
        for md_name in ["SKILL.md", "skill.md"]:
            test_path = skill_dir / md_name
            if test_path.exists():
                skill_md_path = test_path
                break

        if not skill_md_path:
            logger.debug(f"Skipping {skill_name}: no SKILL.md found")
            continue

        metadata = parse_skill_md(skill_md_path)
        if not metadata:
            logger.warning(f"Skipping {skill_name}: failed to parse SKILL.md")
            continue

        # 查找 script 目录
        script_dir = skill_dir / "script"
        if not script_dir.exists():
            logger.warning(f"Skipping {skill_name}: no script/ directory")
            continue

        script_files = [f for f in script_dir.iterdir() if f.suffix == '.py' and f.name != '__init__.py']

        if not script_files:
            logger.warning(f"Skipping {skill_name}: no Python script found in script/")
            continue

        script_file = script_files[0]
        script_name = script_file.stem
        function_name = skill_name.replace('-', '_')

        try:
            func = load_skill_function(skill_name, script_name, function_name, project_root)
            discovered_skills.append({
                "name": skill_name,
                "function_name": function_name,
                "script_name": script_name,
                "metadata": metadata,
                "function": func
            })
            logger.info(f"Discovered skill: {skill_name} (function={function_name})")
        except (FileNotFoundError, AttributeError) as e:
            logger.warning(f"Skipping {skill_name}: {e}")
            continue

    logger.info(f"Discovered {len(discovered_skills)} skills in total")
    return discovered_skills


def load_all_skills(project_root: Path = None) -> dict:
    """
    自动扫描并加载所有 Skills

    Returns:
        {function_name: function, ...}
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    discovered = discover_skills(project_root)

    skills = {}
    for skill_info in discovered:
        function_name = skill_info["function_name"]
        skills[function_name] = skill_info["function"]

    return skills
