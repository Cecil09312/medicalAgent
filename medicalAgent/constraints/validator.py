"""
约束验证器
验证 Agent 的工具调用和输出是否符合约束规则
"""
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

# 模块级 YAML 配置缓存：ConstraintValidator 可能被频繁实例化（如 AgentLoop 每次运行新建），
# 首次加载后缓存配置对象，避免每次实例化都重复读盘。
# 配置内容在整个运行期只读（全项目仅通过 .get() 读取），因此可安全共享同一缓存对象。
_constraints_cache: Optional[Dict[str, Any]] = None

# 模块级预编译正则常量：避免 validate_output 每次调用都重新构造模式字符串并重复编译

# 免责声明检查模式（对应 must_include_disclaimer），未加 flags，大小写敏感性与原逻辑一致
_RE_DISCLAIMER_PATTERNS = (
    re.compile(r"仅供参考"),
    re.compile(r"不构成医疗建议"),
    re.compile(r"请咨询.*医生"),
    re.compile(r"建议就医"),
)

# 高风险场景的就医建议检查模式（对应 must_recommend_doctor_visit_if_high_risk）
_RE_DOCTOR_VISIT_PATTERNS = (
    re.compile(r"建议.*就医"),
    re.compile(r"尽快.*医院"),
    re.compile(r"立即.*急诊"),
    re.compile(r"寻求.*医疗帮助"),
)

# 禁止行为检查模式（对应 forbidden_actions），按动作名分组，结构与原 forbidden_patterns 一致
_RE_FORBIDDEN_PATTERNS = {
    "diagnose_disease": (
        re.compile(r"你(患|得)了.{2,10}(病|症)"),
        re.compile(r"诊断.*为"),
        re.compile(r"确诊.*为"),
    ),
    "prescribe_medication": (
        re.compile(r"建议.*服用"),
        re.compile(r"推荐.*药物"),
        re.compile(r"处方.*药"),
        re.compile(r"用药.*方案"),
    ),
}


class ConstraintValidator:
    """约束验证器"""

    def __init__(self):
        """初始化验证器，加载约束配置"""
        self._load_constraints()

    def _load_constraints(self):
        """加载约束配置文件（首次加载后写入模块级缓存，后续实例直接复用）"""
        global _constraints_cache
        try:
            # 缓存已存在时直接复用，不再重复读盘
            if _constraints_cache is not None:
                self.agent_constraints = _constraints_cache["agent"]
                self.swarm_constraints = _constraints_cache["swarm"]
                return

            constraints_dir = Path(__file__).parent
            
            # 加载 Agent 约束
            agent_constraints_path = constraints_dir / "agent_constraints.yaml"
            with open(agent_constraints_path, 'r', encoding='utf-8') as f:
                self.agent_constraints = yaml.safe_load(f)
            
            # 加载 Swarm 约束
            swarm_constraints_path = constraints_dir / "swarm_constraints.yaml"
            with open(swarm_constraints_path, 'r', encoding='utf-8') as f:
                self.swarm_constraints = yaml.safe_load(f)
            
            # 仅在加载成功后写入缓存；加载失败时不缓存，行为与原实现一致（下次实例化仍会重试加载）
            _constraints_cache = {
                "agent": self.agent_constraints,
                "swarm": self.swarm_constraints,
            }
            logger.debug("Constraint configurations loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load constraint configs: {e}")
            raise

    def validate_tool_call(self, agent_id: str, tool_name: str) -> Dict[str, Any]:
        """
        验证工具调用是否允许
        
        Args:
            agent_id: Agent ID
            tool_name: 工具名称
        
        Returns:
            验证结果字典
        """
        agent_config = self.agent_constraints.get(agent_id)
        if not agent_config:
            logger.warning(f"No constraints found for agent: {agent_id}")
            return {"valid": True, "reason": "No constraints defined"}
        
        allowed_tools = agent_config.get("allowed_tools", [])
        
        if tool_name in allowed_tools:
            return {"valid": True, "reason": "Tool is allowed"}
        else:
            reason = f"Tool '{tool_name}' is not in allowed list for {agent_id}"
            logger.warning(reason)
            return {"valid": False, "reason": reason}

    def validate_output(self, agent_id: str, output: str) -> Dict[str, Any]:
        """
        验证输出是否符合约束
        
        Args:
            agent_id: Agent ID
            output: 输出文本
        
        Returns:
            验证结果字典，包含违规列表和可自动修复项
        """
        agent_config = self.agent_constraints.get(agent_id)
        if not agent_config:
            return {"valid": True, "violations": [], "auto_fixable": []}
        
        output_constraints = agent_config.get("output_constraints", {})
        violations = []
        auto_fixable = []
        
        # 检查免责声明（使用模块级预编译正则，匹配逻辑与原实现完全一致）
        if output_constraints.get("must_include_disclaimer"):
            has_disclaimer = any(p.search(output) for p in _RE_DISCLAIMER_PATTERNS)
            if not has_disclaimer:
                violations.append("missing_disclaimer")
                auto_fixable.append("missing_disclaimer")
        
        # 检查长度限制
        max_length = output_constraints.get("max_response_length")
        if max_length and len(output) > max_length:
            violations.append(f"exceeds_max_length_{max_length}")
            auto_fixable.append("excessive_length")
        
        # 检查高风险警告
        if output_constraints.get("must_recommend_doctor_visit_if_high_risk"):
            high_risk_keywords = ["严重", "紧急", "危险", "立即就医", "急诊"]
            has_high_risk = any(kw in output for kw in high_risk_keywords)
            if has_high_risk:
                has_recommendation = any(p.search(output) for p in _RE_DOCTOR_VISIT_PATTERNS)
                if not has_recommendation:
                    violations.append("missing_high_risk_warning")
                    auto_fixable.append("high_risk_warning")
        
        # 检查禁止行为（诊断、开药等，使用模块级预编译正则，匹配逻辑与原实现完全一致）
        forbidden_actions = agent_config.get("forbidden_actions", [])
        for action in forbidden_actions:
            if action in _RE_FORBIDDEN_PATTERNS:
                for pattern in _RE_FORBIDDEN_PATTERNS[action]:
                    if pattern.search(output):
                        violations.append(f"forbidden_action_{action}")
                        auto_fixable.append(f"remove_{action}")
                        break
        
        valid = len(violations) == 0
        return {
            "valid": valid,
            "violations": violations,
            "auto_fixable": auto_fixable
        }

    def validate_task_decomposition(self, question: str, subtasks: List[Dict]) -> Dict[str, Any]:
        """
        验证任务分解是否符合 Swarm 规则
        
        Args:
            question: 原始问题
            subtasks: 分解后的子任务列表
        
        Returns:
            验证结果字典
        """
        violations = []
        
        max_agents = self.swarm_constraints.get("max_agents_per_task", 3)
        if len(subtasks) > max_agents:
            violations.append(f"exceeds_max_agents_{max_agents}")
        
        # 检查是否选择了合适的 Agent
        agent_rules = self.swarm_constraints.get("agent_selection_rules", {})
        
        # 高风险症状检查
        high_risk_config = agent_rules.get("high_risk_symptoms", {})
        high_risk_keywords = high_risk_config.get("keywords", [])
        has_high_risk = any(kw in question for kw in high_risk_keywords)
        
        if has_high_risk:
            assigned_agents = [t.get("assigned_agent") for t in subtasks]
            if high_risk_config.get("agent") not in assigned_agents:
                violations.append("missing_diagnostic_agent_for_high_risk")
        
        valid = len(violations) == 0
        return {
            "valid": valid,
            "violations": violations
        }

    def get_required_agents(self, question: str) -> List[str]:
        """
        根据问题推荐需要的 Agent
        
        Args:
            question: 用户问题
        
        Returns:
            推荐的 Agent ID 列表
        """
        required_agents = []
        agent_rules = self.swarm_constraints.get("agent_selection_rules", {})
        
        # 检查高风险症状
        high_risk_config = agent_rules.get("high_risk_symptoms", {})
        high_risk_keywords = high_risk_config.get("keywords", [])
        if any(kw in question for kw in high_risk_keywords):
            if high_risk_config.get("agent"):
                required_agents.append(high_risk_config["agent"])
        
        # 检查指南关键词
        guideline_config = agent_rules.get("guideline_keywords", {})
        guideline_keywords = guideline_config.get("keywords", [])
        if any(kw in question for kw in guideline_keywords):
            research_agent = guideline_config.get("agent")
            if research_agent and research_agent not in required_agents:
                required_agents.append(research_agent)
        
        # 检查生活方式关键词
        lifestyle_config = agent_rules.get("lifestyle_keywords", {})
        lifestyle_keywords = lifestyle_config.get("keywords", [])
        if any(kw in question for kw in lifestyle_keywords):
            consultation_agent = lifestyle_config.get("agent")
            if consultation_agent and consultation_agent not in required_agents:
                required_agents.append(consultation_agent)
        
        # 默认至少一个 Agent
        if not required_agents:
            required_agents.append("consultation_agent")
        
        return required_agents
