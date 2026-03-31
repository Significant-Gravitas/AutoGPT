"""Tests for prompting module — verifies supplement assembly."""

from backend.copilot.prompting import get_baseline_supplement, get_sdk_supplement


class TestSupplementContainsClarifyNote:
    """_AGENT_CLARIFY_NOTE must be present in all supplement outputs."""

    def test_sdk_supplement_local_includes_clarify_note(self):
        result = get_sdk_supplement(use_e2b=False, cwd="/tmp")
        assert "Clarifying ambiguous agent goals" in result
        assert "find_block" in result

    def test_sdk_supplement_e2b_includes_clarify_note(self):
        result = get_sdk_supplement(use_e2b=True)
        assert "Clarifying ambiguous agent goals" in result
        assert "find_block" in result

    def test_baseline_supplement_includes_clarify_note(self):
        result = get_baseline_supplement()
        assert "Clarifying ambiguous agent goals" in result
        assert "find_block" in result
