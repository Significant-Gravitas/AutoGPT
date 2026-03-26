"""Tests for the dry-run verification loop prompt changes.

Validates that the create -> dry-run -> fix iterative workflow is properly
communicated through system prompts, tool descriptions, and the agent
building guide.
"""

from pathlib import Path
from typing import Any, cast

import pytest

from backend.copilot.prompting import _SHARED_TOOL_NOTES, get_sdk_supplement
from backend.copilot.service import DEFAULT_SYSTEM_PROMPT
from backend.copilot.tools import TOOL_REGISTRY


class TestSystemPromptDryRunLoop:
    """Verify the system prompt includes dry-run loop instructions."""

    def test_system_prompt_mentions_dry_run(self):
        assert (
            "dry-run" in DEFAULT_SYSTEM_PROMPT.lower()
            or "dry_run" in DEFAULT_SYSTEM_PROMPT
        )

    def test_system_prompt_mentions_create_edit_loop(self):
        prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "create" in prompt_lower
        assert "edit_agent" in DEFAULT_SYSTEM_PROMPT or "edit" in prompt_lower
        assert "loop" in prompt_lower or "repeat" in prompt_lower

    def test_system_prompt_mentions_max_iterations(self):
        assert "3" in DEFAULT_SYSTEM_PROMPT
        assert "iteration" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_inspect_output(self):
        prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
        assert "inspect" in prompt_lower or "check" in prompt_lower

    def test_system_prompt_mentions_never_skip(self):
        assert "NEVER skip" in DEFAULT_SYSTEM_PROMPT


class TestToolDescriptionsDryRunLoop:
    """Verify tool descriptions guide the LLM through the dry-run loop."""

    def test_create_agent_mentions_dry_run(self):
        tool = TOOL_REGISTRY["create_agent"]
        desc = tool.description
        assert "dry_run" in desc or "dry-run" in desc.lower()
        assert "run_agent" in desc

    def test_edit_agent_mentions_dry_run_verification(self):
        tool = TOOL_REGISTRY["edit_agent"]
        desc = tool.description
        assert "dry_run" in desc or "dry-run" in desc.lower()
        assert "fix" in desc.lower() or "issues" in desc.lower()

    def test_run_agent_mentions_dry_run_for_testing(self):
        tool = TOOL_REGISTRY["run_agent"]
        desc = tool.description
        assert "dry_run" in desc or "dry-run" in desc.lower()
        assert "test" in desc.lower() or "verify" in desc.lower()

    def test_run_agent_dry_run_param_mentions_workflow(self):
        tool = TOOL_REGISTRY["run_agent"]
        schema = tool.as_openai_tool()
        fn_def = schema["function"]
        params = cast(dict[str, Any], fn_def.get("parameters", {}))
        dry_run_desc = params["properties"]["dry_run"]["description"]
        assert "create_agent" in dry_run_desc or "edit_agent" in dry_run_desc
        assert "wait_for_result" in dry_run_desc
        assert "3" in dry_run_desc  # max iterations

    def test_get_agent_building_guide_mentions_workflow(self):
        tool = TOOL_REGISTRY["get_agent_building_guide"]
        desc = tool.description
        assert "dry-run" in desc.lower() or "dry_run" in desc

    def test_run_agent_dry_run_param_exists(self):
        tool = TOOL_REGISTRY["run_agent"]
        schema = tool.as_openai_tool()
        fn_def = schema["function"]
        params = cast(dict[str, Any], fn_def.get("parameters", {}))
        assert "dry_run" in params["properties"]
        assert params["properties"]["dry_run"]["type"] == "boolean"


class TestPromptingSupplementDryRunLoop:
    """Verify the prompting supplement includes the iterative workflow."""

    def test_shared_tool_notes_include_dry_run_section(self):
        assert (
            "dry-run" in _SHARED_TOOL_NOTES.lower() or "dry_run" in _SHARED_TOOL_NOTES
        )

    def test_shared_tool_notes_include_loop_workflow(self):
        notes_lower = _SHARED_TOOL_NOTES.lower()
        assert "create" in notes_lower
        assert "fix" in notes_lower
        assert "iteration" in notes_lower or "repeat" in notes_lower

    def test_shared_tool_notes_include_error_patterns(self):
        notes_lower = _SHARED_TOOL_NOTES.lower()
        assert "error" in notes_lower
        assert "null" in notes_lower or "empty" in notes_lower

    def test_sdk_supplement_includes_dry_run_section(self):
        supplement = get_sdk_supplement(use_e2b=False, cwd="/tmp/test")
        supplement_lower = supplement.lower()
        assert "dry-run" in supplement_lower or "dry_run" in supplement_lower


class TestAgentBuildingGuideDryRunLoop:
    """Verify the agent building guide includes the dry-run loop."""

    @pytest.fixture
    def guide_content(self):
        guide_path = Path(__file__).parent / "sdk" / "agent_generation_guide.md"
        return guide_path.read_text(encoding="utf-8")

    def test_guide_has_dry_run_verification_section(self, guide_content):
        assert "Dry-Run Verification Loop" in guide_content

    def test_guide_workflow_includes_dry_run_step(self, guide_content):
        # Check the workflow section mentions dry-run as a step
        assert "dry_run=True" in guide_content or "dry_run" in guide_content

    def test_guide_mentions_good_vs_bad_output(self, guide_content):
        assert "Good output" in guide_content or "good" in guide_content.lower()
        assert "Bad output" in guide_content or "bad" in guide_content.lower()

    def test_guide_mentions_max_iterations(self, guide_content):
        assert "3 times" in guide_content or "3 iterations" in guide_content

    def test_guide_mentions_wait_for_result(self, guide_content):
        assert "wait_for_result" in guide_content

    def test_guide_workflow_has_steps_8_and_9(self, guide_content):
        assert "8. **Dry-run**" in guide_content
        assert "9. **Inspect & fix**" in guide_content
