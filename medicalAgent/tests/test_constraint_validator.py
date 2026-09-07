"""约束验证器与自动修复器单元测试：软约束 + 医疗安全硬约束"""
from constraints.validator import ConstraintValidator
from validation.auto_fixer import AutoFixer


class TestSoftConstraints:
    def test_missing_disclaimer_violation(self):
        validator = ConstraintValidator()
        result = validator.validate_output("consultation_agent", "多喝水，注意休息。")
        assert not result["valid"]
        assert "missing_disclaimer" in result["violations"]

    def test_disclaimer_present_ok(self):
        validator = ConstraintValidator()
        result = validator.validate_output(
            "consultation_agent", "多喝水。以上内容仅供参考，请咨询专业医生。"
        )
        assert "missing_disclaimer" not in result["violations"]

    def test_forbidden_diagnosis(self):
        validator = ConstraintValidator()
        result = validator.validate_output(
            "consultation_agent", "诊断该症状为感冒，仅供参考。"
        )
        assert any(v.startswith("forbidden_action_diagnose") for v in result["violations"])

    def test_excessive_length(self):
        validator = ConstraintValidator()
        result = validator.validate_output("consultation_agent", "感冒" + "多喝水。" * 500)
        assert any(v.startswith("exceeds_max_length") for v in result["violations"])


class TestEmergencyHardConstraint:
    def test_emergency_without_guidance_violates(self):
        validator = ConstraintValidator()
        result = validator.validate_output(
            "consultation_agent", "您这是呼吸困难的表现，可以先在家观察。仅供参考。"
        )
        assert "missing_emergency_guidance" in result["violations"]
        assert "emergency_guidance" in result["auto_fixable"]

    def test_emergency_with_guidance_passes(self):
        validator = ConstraintValidator()
        result = validator.validate_output(
            "consultation_agent", "出现呼吸困难请立即就医。以上内容仅供参考。"
        )
        assert "missing_emergency_guidance" not in result["violations"]

    def test_applies_to_unknown_agent(self):
        """硬约束对未配置约束的 Agent 同样生效（阻断级）"""
        validator = ConstraintValidator()
        result = validator.validate_output(
            "some_new_agent", "咯血可以先观察看看。"
        )
        assert "missing_emergency_guidance" in result["violations"]


class TestAutoFixer:
    def test_fix_emergency_guidance_prepends(self):
        fixer = AutoFixer()
        fixed = fixer.fix_output(
            "您这是呼吸困难的表现，可以先在家观察。", ["emergency_guidance"]
        )
        assert fixed.startswith("⚠️")
        assert "120" in fixed
        assert "可以先在家观察" in fixed  # 原文保留

    def test_fix_emergency_guidance_no_duplication(self):
        fixer = AutoFixer()
        text = "出现呼吸困难请立即就医或拨打 120。"
        assert fixer.fix_output(text, ["emergency_guidance"]) == text

    def test_fix_missing_disclaimer_appends(self):
        fixer = AutoFixer()
        fixed = fixer.fix_output("多喝水。", ["missing_disclaimer"])
        assert "仅供参考" in fixed
