"""Schema regression tests for all registered CoPilot tools.

Validates that every tool in TOOL_REGISTRY produces a well-formed schema:
- description is non-empty
- all `required` fields exist in `properties`
- every property has a `type` and `description`
- total schema character budget does not regress past threshold
"""

import json
from typing import Any, cast

import pytest

from backend.copilot.tools import TOOL_REGISTRY

# Character budget (~4 chars/token heuristic, targeting ~8000 tokens).
# Bumped 32000 -> 32500 on PR #12699 to fit two pieces of load-bearing
# guidance: the wait_for_result dispatch-mode docs on run_agent
# (tells the LLM when to block vs fire-and-forget, and what each
# response shape carries) and the dry_run description. Keeps the
# regression gate effective while accepting a deliberate ~120-token
# spend on LLM-decision-critical copy.
# Bumped 32500 -> 32800 on PR #12871 for the new web_search tool
# (server-side Anthropic beta). Description already trimmed to the
# minimum viable copy; the bump absorbs the schema skeleton cost
# (~300 chars / ~75 tokens) for a new LLM-facing primitive.
# Bumped 32800 -> 33200 on PR #12873 for the web_search Perplexity
# Sonar refactor — adds a load-bearing `deep` boolean with explicit
# "~100x more expensive" cost warning the model must see to avoid
# accidentally triggering sonar-reasoning on ordinary lookups, plus
# synthesised-answer wording in the top-level description so the LLM
# reads the answer before reaching for `web_fetch`. Both are
# LLM-decision-critical copy, not bloat.
# Bumped 33200 -> 34000 when baseline gained the MCP `TodoWrite` tool
# for parity with the Claude Code SDK's built-in (PR #12879). The new
# schema adds ~600 chars; description already trimmed to the minimum
# viable copy.
# Bumped 34000 -> 35000 on PR #12740 for the schedule management tools
# (list_schedules, delete_schedule) needed by the trigger-agent flow.
# Bumped 35000 -> 35500 on PR #12740 for the list_agent_triggers tool
# (returns trigger agents + webhook presets for a parent agent so
# AutoPilot can inspect/manage them).
# Bumped 35500 -> 36500 for the schedule_followup tool. Adds ~950 chars
# of LLM-decision-critical copy: delay_seconds vs cron disambiguation,
# explicit "ends your turn" caveat, and an example wake-up message.
# Bumped 36500 -> 37000 for the schedule_followup `session_id` override
# parameter — lets the model target a different conversation owned by
# the same user (parent autopilot → sub-session followups). The parameter
# description spends ~170 chars on the ownership-rejection semantics so
# the model doesn't try to wake up other users' sessions.
# Bumped 37000 -> 38500 for the skill registry (store_skill, read_skill,
# delete_skill, list_skills) — the four tools that back the new
# self-learning loop.  Descriptions are already trimmed to the minimum
# viable copy; the bump absorbs the four schema skeletons plus the
# canonical SKILL.md frontmatter callout the model needs to format
# distillations correctly.
# Bumped 38500 -> 39000 for the schedule_followup ``session_id=null``
# sentinel — its description spends ~170 chars explaining the "fire
# into a fresh chat" semantics so the model picks the right value
# (null vs omit vs target_session_id) for autopilot-style flows.
# Bumped 39000 -> 39500 for the create-time library-similarity gate:
# find_library_agent's new ``for_creation`` and ``goal_summary``
# parameters and create_agent's ``library_check_ack`` bypass — the
# extra ~270 chars on CI (env-flagged tool registrations push CI
# higher than local) carry the LLM-decision-critical copy for
# "search the library before building new" + "user-confirmed bypass".
# Bumped 39500 -> 40500 on PR #12731 for the decompose_goal tool.
# Adds ~1k chars: step-level schema (id/description/action/block_name),
# the require_approval gate, and the "STOP before building" caveat the
# model needs to halt for user approval instead of rushing into
# create_agent.
# Bumped 40500 -> 41000 when find_library_agent absorbed direct by-id lookup:
# a new ``agent_id`` parameter (library_agent_id / graph_id) that resolves the
# exact agent with no fuzzy name-search fallback, so the library "Chat" flow is
# reliable without a separate tool. Net smaller than a dedicated tool would add.
# Bumped 41000 -> 42500 for the setup_agent_webhook_trigger tool (OPEN-3152). Adds
# ~1.3k chars: identifier + trigger_config + explicit-credentials schema
# and the "manual webhooks return an exact URL / provider webhooks need
# an explicitly chosen account" copy the model needs to drive webhook
# trigger setup without inventing URLs or auto-picking credentials.
# Bumped 42500 -> 45000 for the preset-management tools (list_presets /
# update_preset / delete_preset) that complete the /presets lifecycle for
# AutoPilot. Adds ~1.6k chars: three tool skeletons plus the "is_active
# pauses/resumes the trigger" + "inputs reconfigure & re-register the webhook"
# copy the model needs to manage triggers without re-running setup.
# Bumped 45000 -> 47000 on the dev merge: dev added the proactive chat-platform
# tools (post_to_chat_platform + list_chat_platform_channels, ~1.4k chars) on top
# of the trigger/preset tools above, so the merged registry needs both deltas.
_CHAR_BUDGET = 47_000


@pytest.fixture(scope="module")
def all_tool_schemas() -> list[tuple[str, Any]]:
    """Return (tool_name, openai_schema) pairs for every registered tool."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


def _get_parametrize_data() -> list[tuple[str, object]]:
    """Build parametrize data at collection time."""
    return [(name, tool.as_openai_tool()) for name, tool in TOOL_REGISTRY.items()]


@pytest.mark.parametrize(
    "tool_name,schema",
    _get_parametrize_data(),
    ids=[name for name, _ in _get_parametrize_data()],
)
class TestToolSchema:
    """Validate schema invariants for every registered tool."""

    def test_description_non_empty(self, tool_name: str, schema: dict) -> None:
        desc = schema["function"].get("description", "")
        assert desc, f"Tool '{tool_name}' has an empty description"

    def test_required_fields_exist_in_properties(
        self, tool_name: str, schema: dict
    ) -> None:
        params = schema["function"].get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        for field in required:
            assert field in properties, (
                f"Tool '{tool_name}': required field '{field}' "
                f"not found in properties {list(properties.keys())}"
            )

    def test_every_property_has_type_and_description(
        self, tool_name: str, schema: dict
    ) -> None:
        params = schema["function"].get("parameters", {})
        properties = params.get("properties", {})
        for prop_name, prop_def in properties.items():
            # ``anyOf`` is the JSON-Schema-compliant way to model a
            # nullable / union-typed parameter (e.g. ``session_id`` may
            # be ``string`` or ``null`` for the fresh-chat sentinel).
            # Accept either a top-level ``type`` OR an ``anyOf`` whose
            # branches each carry their own ``type``.
            has_type = "type" in prop_def or (
                isinstance(prop_def.get("anyOf"), list)
                and all(isinstance(b, dict) and "type" in b for b in prop_def["anyOf"])
            )
            assert (
                has_type
            ), f"Tool '{tool_name}', property '{prop_name}' is missing 'type' (or a typed 'anyOf')"
            assert (
                "description" in prop_def
            ), f"Tool '{tool_name}', property '{prop_name}' is missing 'description'"


def test_browser_act_action_enum_complete() -> None:
    """Assert browser_act action enum still contains all 14 supported actions.

    This prevents future PRs from accidentally dropping actions during description
    trimming. The enum is the authoritative list — this locks it at 14 values.
    """
    tool = TOOL_REGISTRY["browser_act"]
    schema = tool.as_openai_tool()
    fn_def = schema["function"]
    params = cast(dict[str, Any], fn_def.get("parameters", {}))
    actions = params["properties"]["action"]["enum"]
    expected = {
        "click",
        "dblclick",
        "fill",
        "type",
        "scroll",
        "hover",
        "press",
        "check",
        "uncheck",
        "select",
        "wait",
        "back",
        "forward",
        "reload",
    }
    assert set(actions) == expected, (
        f"browser_act action enum changed. Got {set(actions)}, expected {expected}. "
        "If you added/removed an action, update this test intentionally."
    )


def test_total_schema_char_budget() -> None:
    """Assert total tool schema size stays under the character budget.

    This locks in the 34% token reduction from #12398 and prevents future
    description bloat from eroding the gains. Uses character count with a
    ~4 chars/token heuristic; see ``_CHAR_BUDGET`` above for the current
    value and its change history.  Character count is tokenizer-agnostic
    — no dependency on GPT or Claude tokenizers — while still providing a
    stable regression gate.
    """
    schemas = [tool.as_openai_tool() for tool in TOOL_REGISTRY.values()]
    serialized = json.dumps(schemas)
    total_chars = len(serialized)
    assert total_chars < _CHAR_BUDGET, (
        f"Tool schemas use {total_chars} chars (~{total_chars // 4} tokens), "
        f"exceeding budget of {_CHAR_BUDGET} chars (~{_CHAR_BUDGET // 4} tokens). "
        f"Description bloat detected — trim descriptions or raise the budget intentionally."
    )


# ── Capability-group filtering (ToolGroup / disabled_groups) ───────────


def test_get_available_tools_hides_graphiti_when_disabled() -> None:
    """When the ``graphiti`` group is disabled, the memory_* tools must
    not appear in the OpenAI schema list — they'd just confuse the model
    and produce opaque runtime errors."""
    from backend.copilot.tools import get_available_tools

    memory_tool_names = {
        "memory_store",
        "memory_search",
        "memory_forget_search",
        "memory_forget_confirm",
    }

    default = {t["function"]["name"] for t in get_available_tools()}
    assert memory_tool_names.issubset(
        default
    ), "sanity: memory_* tools should be present when no groups disabled"

    filtered = {
        t["function"]["name"] for t in get_available_tools(disabled_groups=["graphiti"])
    }
    assert not (
        memory_tool_names & filtered
    ), f"graphiti disabled but memory_* still present: {memory_tool_names & filtered}"
    # Non-graphiti tools stay visible.
    assert "find_block" in filtered
    assert "TodoWrite" in filtered


def test_get_copilot_tool_names_hides_graphiti_when_disabled() -> None:
    """Same invariant for the SDK tool-name list."""
    from backend.copilot.sdk.tool_adapter import MCP_TOOL_PREFIX, get_copilot_tool_names

    memory_mcp_names = {
        f"{MCP_TOOL_PREFIX}memory_store",
        f"{MCP_TOOL_PREFIX}memory_search",
        f"{MCP_TOOL_PREFIX}memory_forget_search",
        f"{MCP_TOOL_PREFIX}memory_forget_confirm",
    }

    default = set(get_copilot_tool_names())
    assert memory_mcp_names.issubset(default)

    filtered = set(get_copilot_tool_names(disabled_groups=["graphiti"]))
    assert not (
        memory_mcp_names & filtered
    ), f"graphiti disabled but memory MCP names still present: {memory_mcp_names & filtered}"
    # E2B path stays consistent.
    filtered_e2b = set(
        get_copilot_tool_names(use_e2b=True, disabled_groups=["graphiti"])
    )
    assert not (memory_mcp_names & filtered_e2b)
