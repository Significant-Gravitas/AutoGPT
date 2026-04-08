"""Tests for agent generation guide — verifies clarification section."""

from pathlib import Path


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
