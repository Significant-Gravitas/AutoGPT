"""Tests for prompting module — verifies supplement assembly."""

from backend.copilot.prompting import get_baseline_supplement, get_sdk_supplement


class TestSupplementContainsClarifyNote:
    """_AGENT_CLARIFY_NOTE must be present in all supplement outputs."""

    def test_sdk_supplement_local_includes_clarify_note(self):
        result = get_sdk_supplement(use_e2b=False, cwd="/tmp")
        assert "clarify_agent_request" in result

    def test_sdk_supplement_e2b_includes_clarify_note(self):
        result = get_sdk_supplement(use_e2b=True)
        assert "clarify_agent_request" in result

    def test_baseline_supplement_includes_clarify_note(self):
        result = get_baseline_supplement()
        assert "clarify_agent_request" in result
