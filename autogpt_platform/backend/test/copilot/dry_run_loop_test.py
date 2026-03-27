"""Prompt regression tests for the dry-run verification loop.

These tests verify that the create -> dry-run -> fix iterative workflow is
properly communicated through tool descriptions, the prompting supplement,
and the agent building guide.

After deduplication, the full dry-run workflow lives in the
agent_generation_guide.md only. The system prompt and individual tool
descriptions no longer repeat it — they keep a minimal footprint.

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


class TestSystemPromptBasics:
    """Verify the system prompt includes essential baseline content.

    After deduplication, the dry-run workflow lives only in the guide.
    The system prompt carries tone and personality only.
    """

    def test_mentions_automations(self):
        assert "automations" in DEFAULT_SYSTEM_PROMPT.lower()

    def test_mentions_action_oriented(self):
        assert "action-oriented" in DEFAULT_SYSTEM_PROMPT.lower()


class TestToolDescriptionsDryRunLoop:
    """Verify tool descriptions and parameters related to the dry-run loop."""

    def test_get_agent_building_guide_mentions_workflow(self):
        desc = TOOL_REGISTRY["get_agent_building_guide"].description
        assert "dry-run" in desc.lower()

    def test_run_agent_dry_run_param_exists_and_is_boolean(self):
        schema = TOOL_REGISTRY["run_agent"].as_openai_tool()
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        assert "dry_run" in params["properties"]
        assert params["properties"]["dry_run"]["type"] == "boolean"

    def test_run_agent_dry_run_param_references_guide(self):
        """After deduplication the dry_run param description points to the guide."""
        schema = TOOL_REGISTRY["run_agent"].as_openai_tool()
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        dry_run_desc = params["properties"]["dry_run"]["description"]
        assert "agent_generation_guide" in dry_run_desc

    def test_run_agent_dry_run_param_mentions_simulates(self):
        schema = TOOL_REGISTRY["run_agent"].as_openai_tool()
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        dry_run_desc = params["properties"]["dry_run"]["description"]
        assert "simulates" in dry_run_desc.lower()


class TestPromptingSupplementContent:
    """Verify the prompting supplement (via get_sdk_supplement) includes
    essential shared tool notes.  After deduplication, the dry-run workflow
    lives only in the guide; the supplement carries storage, file-handling,
    and tool-discovery notes.
    """

    def test_includes_tool_discovery_priority(self):
        assert "Tool Discovery Priority" in _SDK_SUPPLEMENT

    def test_includes_find_block_first(self):
        assert "find_block first" in _SDK_SUPPLEMENT or "find_block" in _SDK_SUPPLEMENT

    def test_includes_send_authenticated_web_request(self):
        assert "SendAuthenticatedWebRequestBlock" in _SDK_SUPPLEMENT


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
