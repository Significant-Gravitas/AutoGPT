"""Tests for agent generation guide — verifies clarification section."""

from pathlib import Path


class TestAgentGenerationGuideContainsClarifySection:
    """The agent generation guide must include the clarification section."""

    def test_guide_includes_clarify_before_building(self):
        guide_path = Path(__file__).parent / "sdk" / "agent_generation_guide.md"
        content = guide_path.read_text(encoding="utf-8")
        assert "Clarifying Before Building" in content

    def test_guide_mentions_find_block_for_clarification(self):
        guide_path = Path(__file__).parent / "sdk" / "agent_generation_guide.md"
        content = guide_path.read_text(encoding="utf-8")
        # find_block must appear in the clarification section (before the workflow)
        clarify_section = content.split("Clarifying Before Building")[1].split(
            "### Workflow"
        )[0]
        assert "find_block" in clarify_section
