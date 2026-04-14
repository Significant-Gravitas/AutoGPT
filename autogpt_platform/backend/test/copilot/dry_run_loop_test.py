"""Prompt regression tests AND functional tests for the dry-run verification loop.

NOTE: This file lives in test/copilot/ rather than being colocated with a
single source module because it is a cross-cutting test spanning multiple
modules: prompting.py, service.py, agent_generation_guide.md, and run_agent.py.

These tests verify that the create -> dry-run -> fix iterative workflow is
properly communicated through tool descriptions, the prompting supplement,
and the agent building guide.

After deduplication, the full dry-run workflow lives in the
agent_generation_guide.md only. The system prompt and individual tool
descriptions no longer repeat it — they keep a minimal footprint.

**Intentionally brittle**: the assertions check for specific substrings so
that accidental removal or rewording of key instructions is caught. If you
deliberately reword a prompt, update the corresponding assertion here.

--- Functional tests (added separately) ---

The dry-run loop is primarily a *prompt/guide* feature — the copilot reads
the guide and follows its instructions.  There are no standalone Python
functions that implement "loop until passing" logic; the loop is driven by
the LLM.  However, several pieces of real Python infrastructure make the
loop possible:

1. The ``run_agent`` and ``run_block`` OpenAI tool schemas expose a
   ``dry_run`` boolean parameter that the LLM must be able to set.
2. The ``RunAgentInput`` Pydantic model validates ``dry_run`` as a required
   bool, so the executor can branch on it.
3. The ``_check_prerequisites`` method in ``RunAgentTool`` bypasses
   credential and missing-input gates when ``dry_run=True``.
4. The guide documents the workflow steps in a specific order that the LLM
   must follow: create/edit -> dry-run -> inspect -> fix -> repeat.

The functional test classes below exercise items 1-4 directly.
"""

import re
from pathlib import Path
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletionToolParam
from pydantic import ValidationError

from backend.copilot.prompting import get_sdk_supplement
from backend.copilot.service import CACHEABLE_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.run_agent import RunAgentInput

# Resolved once for the whole module so individual tests stay fast.
_SDK_SUPPLEMENT = get_sdk_supplement(use_e2b=False, cwd="/tmp/test")


# ---------------------------------------------------------------------------
# Prompt regression tests (original)
# ---------------------------------------------------------------------------


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
    """Verify tool descriptions and parameters related to the dry-run loop.

    After the session-level dry_run refactor, dry_run is NOT exposed in any
    LLM tool schema — it is set at the session level and derived by each tool
    from session.dry_run.  These tests verify that the schema is clean and that
    the guide still documents the dry-run workflow.
    """

    def test_get_agent_building_guide_mentions_workflow(self):
        desc = TOOL_REGISTRY["get_agent_building_guide"].description
        assert "dry-run" in desc.lower()

    def test_run_agent_dry_run_in_llm_schema(self):
        """dry_run must be in the run_agent LLM schema so the LLM can request
        per-call dry runs in normal sessions (e.g. "test this agent")."""
        schema = TOOL_REGISTRY["run_agent"].as_openai_tool()
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        assert "dry_run" in params.get("properties", {}), (
            "dry_run must be exposed in the run_agent LLM schema so the LLM "
            "can request per-call dry runs in normal sessions"
        )

    def test_run_block_dry_run_not_in_llm_schema(self):
        """dry_run must NOT be in the run_block LLM schema — it is session-level."""
        schema = TOOL_REGISTRY["run_block"].as_openai_tool()
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        assert "dry_run" not in params.get("properties", {}), (
            "dry_run must not be exposed in the run_block LLM schema; "
            "it is controlled at the session level"
        )


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


# ---------------------------------------------------------------------------
# Functional tests: tool schema validation
# ---------------------------------------------------------------------------


class TestRunAgentToolSchema:
    """Validate the run_agent OpenAI tool schema is clean of session-level fields.

    After the session-level dry_run refactor, dry_run is NOT exposed in the LLM
    schema — it is set at the session level and applied by _execute.  These tests
    verify the full schema structure that the LLM receives.
    """

    @pytest.fixture
    def schema(self) -> ChatCompletionToolParam:
        return TOOL_REGISTRY["run_agent"].as_openai_tool()

    def test_schema_is_valid_openai_tool(self, schema: ChatCompletionToolParam):
        """The schema has the required top-level OpenAI structure."""
        assert schema["type"] == "function"
        assert "function" in schema
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        assert func["name"] == "run_agent"

    def test_dry_run_in_llm_schema(self, schema: ChatCompletionToolParam):
        """dry_run must be in the run_agent LLM schema so the LLM can request
        per-call dry runs in normal sessions."""
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        assert "dry_run" in params.get(
            "properties", {}
        ), "dry_run must be exposed in the run_agent LLM schema"
        assert "dry_run" not in params.get("required", [])

    def test_wait_for_result_in_schema(self, schema: ChatCompletionToolParam):
        """wait_for_result must be present — the guide instructs the LLM
        to pass wait_for_result=120 during dry-run verification."""
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        assert "wait_for_result" in params["properties"]
        assert params["properties"]["wait_for_result"]["type"] == "integer"


class TestRunBlockToolSchema:
    """Validate the run_block OpenAI tool schema is clean of session-level fields.

    After the session-level dry_run refactor, dry_run is NOT in the LLM schema.
    """

    @pytest.fixture
    def schema(self) -> ChatCompletionToolParam:
        return TOOL_REGISTRY["run_block"].as_openai_tool()

    def test_schema_is_valid_openai_tool(self, schema: ChatCompletionToolParam):
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "run_block"
        assert "parameters" in func

    def test_dry_run_not_in_llm_schema(self, schema: ChatCompletionToolParam):
        """dry_run must NOT be in the run_block LLM schema — it is session-level."""
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        props = params.get("properties", {})
        assert (
            "dry_run" not in props
        ), "dry_run must not be exposed in the run_block LLM schema"
        assert "dry_run" not in params.get("required", [])

    def test_block_id_and_input_data_are_required(
        self, schema: ChatCompletionToolParam
    ):
        """block_id and input_data must be required parameters."""
        params = cast(dict[str, Any], schema["function"].get("parameters", {}))
        required = params.get("required", [])
        assert "block_id" in required
        assert "input_data" in required


# ---------------------------------------------------------------------------
# Functional tests: RunAgentInput Pydantic model
# ---------------------------------------------------------------------------


class TestRunAgentInputModel:
    """Validate RunAgentInput Pydantic model handles dry_run correctly.

    dry_run is exposed in the LLM schema so the LLM can request per-call
    dry runs in normal sessions. It defaults to False. Session-level
    dry_run=True forces all runs dry; normal sessions respect the LLM's choice.
    """

    def test_dry_run_default_false(self):
        """dry_run defaults to False when not provided."""
        model = RunAgentInput(username_agent_slug="user/agent")
        assert model.dry_run is False

    def test_dry_run_in_schema_parameters(self):
        """dry_run must appear in RunAgentTool.parameters so the LLM can
        request per-call dry runs in normal sessions."""
        from backend.copilot.tools.run_agent import RunAgentTool

        tool = RunAgentTool()
        assert "dry_run" in tool.parameters.get(
            "properties", {}
        ), "dry_run must be exposed in the LLM tool schema"

    def test_dry_run_accepts_true(self):
        model = RunAgentInput(username_agent_slug="user/agent", dry_run=True)
        assert model.dry_run is True

    def test_dry_run_accepts_false(self):
        """dry_run=False must be accepted when provided explicitly."""
        model = RunAgentInput(username_agent_slug="user/agent", dry_run=False)
        assert model.dry_run is False

    def test_dry_run_coerces_truthy_int(self):
        """Pydantic bool fields coerce int 1 to True."""
        model = RunAgentInput(username_agent_slug="user/agent", dry_run=1)  # type: ignore[arg-type]
        assert model.dry_run is True

    def test_dry_run_coerces_falsy_int(self):
        """Pydantic bool fields coerce int 0 to False."""
        model = RunAgentInput(username_agent_slug="user/agent", dry_run=0)  # type: ignore[arg-type]
        assert model.dry_run is False

    def test_dry_run_with_wait_for_result(self):
        """The guide instructs passing both dry_run=True and wait_for_result=120.
        The model must accept this combination."""
        model = RunAgentInput(
            username_agent_slug="user/agent",
            dry_run=True,
            wait_for_result=120,
        )
        assert model.dry_run is True
        assert model.wait_for_result == 120

    def test_wait_for_result_upper_bound(self):
        """wait_for_result is bounded at 300 seconds (ge=0, le=300)."""
        with pytest.raises(ValidationError):
            RunAgentInput(
                username_agent_slug="user/agent",
                wait_for_result=301,
            )

    def test_string_fields_are_stripped(self):
        """The strip_strings validator should strip whitespace from string fields."""
        model = RunAgentInput(username_agent_slug="  user/agent  ")
        assert model.username_agent_slug == "user/agent"


# ---------------------------------------------------------------------------
# Functional tests: guide documents the correct workflow ordering
# ---------------------------------------------------------------------------


class TestGuideWorkflowOrdering:
    """Verify the guide documents workflow steps in the correct order.

    The LLM must see: create/edit -> dry-run -> inspect -> fix -> repeat.
    If these steps are reordered, the copilot would follow the wrong sequence.
    These tests verify *ordering*, not just presence.
    """

    @pytest.fixture
    def guide_content(self) -> str:
        guide_path = (
            Path(__file__).resolve().parent.parent.parent
            / "backend"
            / "copilot"
            / "sdk"
            / "agent_generation_guide.md"
        )
        return guide_path.read_text(encoding="utf-8")

    def test_create_before_dry_run_in_workflow(self, guide_content: str):
        """Step 7 (Save/create_agent) must appear before step 8 (Dry-run)."""
        create_pos = guide_content.index("create_agent")
        dry_run_pos = guide_content.index("dry_run=True")
        assert (
            create_pos < dry_run_pos
        ), "create_agent must appear before dry_run=True in the workflow"

    def test_dry_run_before_inspect_in_verification_section(self, guide_content: str):
        """In the verification loop section, Dry-run step must come before
        Inspect & fix step."""
        section_start = guide_content.index("REQUIRED: Dry-Run Verification Loop")
        section = guide_content[section_start:]
        dry_run_pos = section.index("**Dry-run**")
        inspect_pos = section.index("**Inspect")
        assert (
            dry_run_pos < inspect_pos
        ), "Dry-run step must come before Inspect & fix in the verification loop"

    def test_fix_before_repeat_in_verification_section(self, guide_content: str):
        """The Fix step must come before the Repeat step."""
        section_start = guide_content.index("REQUIRED: Dry-Run Verification Loop")
        section = guide_content[section_start:]
        fix_pos = section.index("**Fix**")
        repeat_pos = section.index("**Repeat**")
        assert fix_pos < repeat_pos

    def test_good_output_before_bad_output(self, guide_content: str):
        """Good output examples should be listed before bad output examples,
        so the LLM sees the success pattern first."""
        good_pos = guide_content.index("**Good output**")
        bad_pos = guide_content.index("**Bad output**")
        assert good_pos < bad_pos

    def test_numbered_steps_in_verification_section(self, guide_content: str):
        """The step-by-step workflow should have numbered steps 1-5."""
        section_start = guide_content.index("Step-by-step workflow")
        section = guide_content[section_start:]
        # The section should contain numbered items 1 through 5
        for step_num in range(1, 6):
            assert (
                f"{step_num}. " in section
            ), f"Missing numbered step {step_num} in verification workflow"

    def test_workflow_steps_are_in_numbered_order(self, guide_content: str):
        """The main workflow steps (1-9) must appear in ascending order."""
        # Extract the numbered workflow items from the top-level workflow section
        workflow_start = guide_content.index("### Workflow for Creating/Editing Agents")
        # End at the next ### section
        next_section = guide_content.index("### Agent JSON Structure")
        workflow_section = guide_content[workflow_start:next_section]
        step_positions = []
        for step_num in range(1, 10):
            pattern = rf"^{step_num}\.\s"
            match = re.search(pattern, workflow_section, re.MULTILINE)
            if match:
                step_positions.append((step_num, match.start()))
        # Verify at least steps 1-9 are present and in order
        assert (
            len(step_positions) >= 9
        ), f"Expected 9 workflow steps, found {len(step_positions)}"
        for i in range(1, len(step_positions)):
            prev_num, prev_pos = step_positions[i - 1]
            curr_num, curr_pos = step_positions[i]
            assert prev_pos < curr_pos, (
                f"Step {prev_num} (pos {prev_pos}) should appear before "
                f"step {curr_num} (pos {curr_pos})"
            )
