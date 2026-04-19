"""Tests for agent generation guide — verifies clarification section."""

import importlib
from pathlib import Path

from backend.copilot import prompting


class TestGetSdkSupplementStaticPlaceholder:
    """get_sdk_supplement must return a static string so the system prompt is
    identical for all users and sessions, enabling cross-user prompt-cache hits.
    """

    def setup_method(self):
        # Reset the module-level singleton before each test so tests are isolated.
        importlib.reload(prompting)

    def test_local_mode_uses_placeholder_not_uuid(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "/tmp/copilot-<session-id>" in result

    def test_local_mode_is_idempotent(self):
        first = prompting.get_sdk_supplement(use_e2b=False)
        second = prompting.get_sdk_supplement(use_e2b=False)
        assert first == second, "Supplement must be identical across calls"

    def test_e2b_mode_uses_home_user(self):
        result = prompting.get_sdk_supplement(use_e2b=True)
        assert "/home/user" in result

    def test_e2b_mode_has_no_session_placeholder(self):
        result = prompting.get_sdk_supplement(use_e2b=True)
        assert "<session-id>" not in result


class TestAgentGenerationGuideContainsClarifySection:
    """The agent generation guide must include the clarification section."""

    def test_guide_includes_clarify_section(self):
        guide_path = Path(__file__).parent / "sdk" / "agent_generation_guide.md"
        content = guide_path.read_text(encoding="utf-8")
        assert "Before or During Building" in content

    def test_guide_mentions_find_block_for_clarification(self):
        guide_path = Path(__file__).parent / "sdk" / "agent_generation_guide.md"
        content = guide_path.read_text(encoding="utf-8")
        clarify_section = content.split("Before or During Building")[1].split(
            "### Workflow"
        )[0]
        assert "find_block" in clarify_section

    def test_guide_mentions_ask_question_tool(self):
        guide_path = Path(__file__).parent / "sdk" / "agent_generation_guide.md"
        content = guide_path.read_text(encoding="utf-8")
        clarify_section = content.split("Before or During Building")[1].split(
            "### Workflow"
        )[0]
        assert "ask_question" in clarify_section
