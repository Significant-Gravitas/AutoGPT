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
    """Verify the system prompt includes a brief dry-run loop reference.

    The detailed workflow lives in the supplement (_SHARED_TOOL_NOTES);
    the system prompt only carries a short pointer to keep it minimal.
    """

    def test_system_prompt_mentions_dry_run(self):
        assert "dry-run" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_never_skip(self):
        assert "NEVER skip" in DEFAULT_SYSTEM_PROMPT

    def test_system_prompt_references_tool_notes(self):
        assert "tool notes" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_iterations(self):
        assert "3 iteration" in DEFAULT_SYSTEM_PROMPT.lower()


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
        assert "dry-run testing" in desc.lower() or "wiring errors" in desc.lower()

    def test_run_agent_mentions_dry_run_for_testing(self):
        tool = TOOL_REGISTRY["run_agent"]
        desc = tool.description
        assert "dry_run=True" in desc
        assert (
            "test agent wiring" in desc.lower() or "simulates execution" in desc.lower()
        )

    def test_run_agent_dry_run_param_mentions_workflow(self):
        tool = TOOL_REGISTRY["run_agent"]
        schema = tool.as_openai_tool()
        fn_def = schema["function"]
        params = cast(dict[str, Any], fn_def.get("parameters", {}))
        dry_run_desc = params["properties"]["dry_run"]["description"]
        assert "create_agent" in dry_run_desc or "edit_agent" in dry_run_desc
        assert "wait_for_result" in dry_run_desc
        assert "3 iterations" in dry_run_desc

    def test_get_agent_building_guide_mentions_workflow(self):
        tool = TOOL_REGISTRY["get_agent_building_guide"]
        desc = tool.description
        assert "dry-run" in desc.lower()

    def test_run_agent_dry_run_param_exists(self):
        tool = TOOL_REGISTRY["run_agent"]
        schema = tool.as_openai_tool()
        fn_def = schema["function"]
        params = cast(dict[str, Any], fn_def.get("parameters", {}))
        assert "dry_run" in params["properties"]
        assert params["properties"]["dry_run"]["type"] == "boolean"


class TestPromptingSupplementDryRunLoop:
    """Verify the prompting supplement includes the iterative workflow."""

    def test_shared_tool_notes_include_dry_run_section_header(self):
        assert "Iterative agent development" in _SHARED_TOOL_NOTES

    def test_shared_tool_notes_include_create_dry_run_fix_workflow(self):
        assert "create -> dry-run -> fix" in _SHARED_TOOL_NOTES.lower()

    def test_shared_tool_notes_include_error_patterns(self):
        notes_lower = _SHARED_TOOL_NOTES.lower()
        assert "errors / failed nodes" in notes_lower
        assert "null / empty outputs" in notes_lower
        assert "nodes that never executed" in notes_lower

    def test_shared_tool_notes_include_max_iterations(self):
        assert "3 iterations" in _SHARED_TOOL_NOTES

    def test_sdk_supplement_includes_dry_run_section(self):
        supplement = get_sdk_supplement(use_e2b=False, cwd="/tmp/test")
        assert "Iterative agent development" in supplement

    def test_shared_tool_notes_include_tool_discovery_priority(self):
        assert "Tool Discovery Priority" in _SHARED_TOOL_NOTES


class TestAgentBuildingGuideDryRunLoop:
    """Verify the agent building guide includes the dry-run loop."""

    @pytest.fixture
    def guide_content(self):
        guide_path = Path(__file__).parent / "sdk" / "agent_generation_guide.md"
        return guide_path.read_text(encoding="utf-8")

    def test_guide_has_dry_run_verification_section(self, guide_content):
        assert "REQUIRED: Dry-Run Verification Loop" in guide_content

    def test_guide_workflow_includes_dry_run_step(self, guide_content):
        assert "dry_run=True" in guide_content

    def test_guide_mentions_good_vs_bad_output(self, guide_content):
        assert "**Good output**" in guide_content
        assert "**Bad output**" in guide_content

    def test_guide_mentions_max_iterations(self, guide_content):
        assert "**3 iterations**" in guide_content

    def test_guide_mentions_wait_for_result(self, guide_content):
        assert "wait_for_result=120" in guide_content

    def test_guide_mentions_view_agent_output(self, guide_content):
        assert "view_agent_output" in guide_content

    def test_guide_workflow_has_steps_8_and_9(self, guide_content):
        assert "8. **Dry-run**" in guide_content
        assert "9. **Inspect & fix**" in guide_content
