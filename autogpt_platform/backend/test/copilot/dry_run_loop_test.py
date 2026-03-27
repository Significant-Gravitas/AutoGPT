"""Prompt regression tests for the dry-run verification loop.

These tests verify that the create -> dry-run -> fix iterative workflow is
properly communicated through system prompts, tool descriptions, the
prompting supplement, and the agent building guide.

**Intentionally brittle**: the assertions check for specific substrings so
that accidental removal or rewording of key instructions is caught. If you
deliberately reword a prompt, update the corresponding assertion here.
"""

from pathlib import Path
from typing import Any, cast

import pytest

from backend.copilot.prompting import get_sdk_supplement
from backend.copilot.service import DEFAULT_SYSTEM_PROMPT
from backend.copilot.tools import TOOL_REGISTRY

# Resolved once for the whole module so individual tests stay fast.
_SDK_SUPPLEMENT = get_sdk_supplement(use_e2b=False, cwd="/tmp/test")


class TestSystemPromptDryRunLoop:
    """Verify the system prompt includes a brief dry-run loop reference.

    The detailed workflow lives in the supplement / guide; the system prompt
    only carries a short pointer to keep it minimal.
    """

    def test_mentions_dry_run(self):
        assert (
            "dry-run" in DEFAULT_SYSTEM_PROMPT.lower()
            or "dry_run" in DEFAULT_SYSTEM_PROMPT
        )

    def test_mentions_never_skip(self):
        assert "NEVER skip" in DEFAULT_SYSTEM_PROMPT

    def test_references_tool_notes(self):
        assert "tool notes" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_mentions_repeat_until(self):
        assert "repeat until" in DEFAULT_SYSTEM_PROMPT.lower()


class TestToolDescriptionsDryRunLoop:
    """Verify tool descriptions guide the LLM through the dry-run loop."""

    def test_create_agent_mentions_dry_run(self):
        desc = TOOL_REGISTRY["create_agent"].description
        assert "dry_run" in desc or "dry-run" in desc.lower()
        assert "run_agent" in desc

    def test_edit_agent_mentions_dry_run_verification(self):
        desc = TOOL_REGISTRY["edit_agent"].description
        assert "dry_run" in desc or "dry-run" in desc.lower()
        assert "dry-run testing" in desc.lower() or "wiring errors" in desc.lower()

    def test_run_agent_mentions_dry_run_for_testing(self):
        desc = TOOL_REGISTRY["run_agent"].description
        assert "dry_run=True" in desc
        desc_lower = desc.lower()
        assert "test agent wiring" in desc_lower or "simulates execution" in desc_lower

    def test_run_agent_dry_run_param_mentions_workflow(self):
        schema = TOOL_REGISTRY["run_agent"].as_openai_tool()
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        dry_run_desc = params["properties"]["dry_run"]["description"]
        assert "create_agent" in dry_run_desc or "edit_agent" in dry_run_desc
        assert "wait_for_result" in dry_run_desc
        assert "repeat" in dry_run_desc.lower()

    def test_get_agent_building_guide_mentions_workflow(self):
        desc = TOOL_REGISTRY["get_agent_building_guide"].description
        assert "dry-run" in desc.lower()

    def test_run_agent_dry_run_param_exists_and_is_boolean(self):
        schema = TOOL_REGISTRY["run_agent"].as_openai_tool()
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        assert "dry_run" in params["properties"]
        assert params["properties"]["dry_run"]["type"] == "boolean"


class TestPromptingSupplementDryRunLoop:
    """Verify the prompting supplement (via get_sdk_supplement) includes the
    iterative workflow.  Tests use the public ``get_sdk_supplement`` API
    rather than importing the private ``_SHARED_TOOL_NOTES`` directly.
    """

    def test_includes_dry_run_section_header(self):
        assert "Iterative agent development" in _SDK_SUPPLEMENT

    def test_includes_create_dry_run_fix_workflow(self):
        assert "create -> dry-run -> fix" in _SDK_SUPPLEMENT.lower()

    def test_includes_error_inspection_guidance(self):
        """The supplement should reference error patterns (detailed list
        lives in the guide; the supplement carries at least a summary)."""
        lower = _SDK_SUPPLEMENT.lower()
        assert "errors" in lower or "failed" in lower
        assert "null" in lower or "empty" in lower

    def test_includes_repeat_until(self):
        assert "repeat until" in _SDK_SUPPLEMENT.lower()

    def test_includes_tool_discovery_priority(self):
        assert "Tool Discovery Priority" in _SDK_SUPPLEMENT


class TestAgentBuildingGuideDryRunLoop:
    """Verify the agent building guide includes the dry-run loop."""

    @pytest.fixture
    def guide_content(self):
        guide_path = (
            Path(__file__).resolve().parent.parent.parent
            / "backend"
            / "copilot"
            / "sdk"
            / "agent_generation_guide.md"
        )
        return guide_path.read_text(encoding="utf-8")

    def test_has_dry_run_verification_section(self, guide_content):
        assert "REQUIRED: Dry-Run Verification Loop" in guide_content

    def test_workflow_includes_dry_run_step(self, guide_content):
        assert "dry_run=True" in guide_content

    def test_mentions_good_vs_bad_output(self, guide_content):
        assert "**Good output**" in guide_content
        assert "**Bad output**" in guide_content

    def test_mentions_repeat_until_pass(self, guide_content):
        lower = guide_content.lower()
        assert "repeat" in lower
        assert "clearly unfixable" in lower

    def test_mentions_wait_for_result(self, guide_content):
        assert "wait_for_result=120" in guide_content

    def test_mentions_view_agent_output(self, guide_content):
        assert "view_agent_output" in guide_content

    def test_workflow_has_dry_run_and_inspect_steps(self, guide_content):
        assert "**Dry-run**" in guide_content
        assert "**Inspect & fix**" in guide_content
