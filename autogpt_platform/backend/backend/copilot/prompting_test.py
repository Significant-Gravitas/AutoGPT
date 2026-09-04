"""Tests for prompting helpers."""

import ast
import importlib
import inspect
from unittest.mock import AsyncMock

import pytest

from backend.copilot import prompting
from backend.copilot.baseline import service as baseline_service
from backend.copilot.sdk import service as sdk_service
from backend.util.feature_flag import Flag


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


class TestCredentialsSurfacingGuardrails:
    """The system prompt must instruct the model to (a) surface sign-in cards
    eagerly via tool calls and (b) never claim a card has appeared unless one
    was just emitted in the same turn. Both behaviours prevent the user from
    being stranded waiting for a card that was never produced.
    """

    def test_local_prompt_contains_eager_surfacing_rule(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "Surface the sign-in card EAGERLY" in result

    def test_e2b_prompt_contains_eager_surfacing_rule(self):
        result = prompting.get_sdk_supplement(use_e2b=True)
        assert "Surface the sign-in card EAGERLY" in result

    def test_prompt_contains_anti_hallucination_guardrail(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "NEVER claim a card has appeared" in result
        assert "call the tool first" in result


class TestToolDiscoveryPriorityAntiPattern:
    """The Tool Discovery Priority section must forbid claiming a capability
    gap without calling ``find_block`` first — this is the regression the
    LinkedIn-skip incident on dev (May 2026) exposed.
    """

    def test_supplement_contains_find_block_mandatory_language(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        # The header must signal that find_block is mandatory before any
        # "no integration" reply.
        assert "find_block` is MANDATORY" in result

    def test_supplement_lists_the_forbidden_phrases(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        # The anti-pattern section must explicitly enumerate the
        # phrases the model emitted in the regression so the model
        # can pattern-match on its own draft and reject it.
        assert "We don't have a native X integration yet." in result
        assert "There's no block for X." in result

    def test_supplement_includes_correct_flow_template(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        # The 3-step correct-flow block must be present so the model
        # has a concrete template to follow, not just a prohibition.
        assert "Correct flow" in result
        assert 'find_block(query="<service> <action>")' in result


class TestGraphitiMemoryScope:
    def test_supplement_describes_assistant_scoped_memory(self):
        result = prompting.get_graphiti_supplement()

        assert "scoped to the assistant running this session" in result
        assert "AutoPilot uses the user's personal memory" in result
        assert "each hired expert uses its own separate memory" in result
        assert "Memory is private and isolated to the current assistant" in result
        assert "cannot read each other's memories" in result
        assert "Memory is private to this user — no other user can see it" not in result


class TestAutopilotDelegationSupplement:
    """One gate both engines go through, so flag-off is provably a no-op."""

    @pytest.mark.asyncio
    async def test_flag_off_leaves_the_prompt_byte_identical(self, monkeypatch):
        _set_flag(monkeypatch, False)
        assert (
            await prompting.build_autopilot_delegation_supplement("u1", "interactive")
            == ""
        )

    @pytest.mark.asyncio
    async def test_anonymous_turn_fails_closed_without_consulting_the_flag(
        self, monkeypatch
    ):
        flag = _set_flag(monkeypatch, True)
        assert (
            await prompting.build_autopilot_delegation_supplement(None, "interactive")
            == ""
        )
        flag.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_on_returns_the_rules(self, monkeypatch):
        flag = _set_flag(monkeypatch, True)
        result = await prompting.build_autopilot_delegation_supplement(
            "u1", "interactive"
        )
        assert result == prompting._autopilot_delegation_rules()
        assert result != ""
        assert flag.await_args.args[0] is Flag.AUTOPILOT_DELEGATION
        assert flag.await_args.args[1] == "u1", "must evaluate for THIS user"
        assert flag.await_args.kwargs["default"] is False

    @pytest.mark.asyncio
    async def test_a_sub_session_is_never_told_to_delegate(self, monkeypatch):
        """A sub told to delegate would spawn its own, with nothing bounding
        the chain."""
        flag = _set_flag(monkeypatch, True)
        result = await prompting.build_autopilot_delegation_supplement(
            "u1", "automation"
        )
        assert result == ""
        flag.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_origin_counts_as_automation(self, monkeypatch):
        """Matches child_session_origin."""
        _set_flag(monkeypatch, True)
        assert await prompting.build_autopilot_delegation_supplement("u1", None) == ""

    def test_the_rules_name_the_tool_and_forbid_continuing_a_sub(self):
        rules = prompting._autopilot_delegation_rules()
        assert "run_sub_session" in rules
        assert "Do not try to continue a previous one" in rules
        # Polling a still-running sub must stay allowed.
        assert "get_sub_session_result" in rules

    def test_both_engines_append_the_gated_supplement(self):
        """Asserted against the AST, not the text: ``getsource`` returns
        comments, so a grep stays green on a commented-out concatenation."""
        for module in (baseline_service, sdk_service):
            concatenated = _system_prompt_operands(module)
            assert concatenated, f"no system_prompt assignment found in {module}"
            assert "autopilot_delegation_supplement" in concatenated, module


def _set_flag(monkeypatch, enabled: bool) -> AsyncMock:
    flag = AsyncMock(return_value=enabled)
    monkeypatch.setattr("backend.copilot.prompting.is_feature_enabled", flag)
    return flag


def _system_prompt_operands(module) -> set[str]:
    """Names concatenated into the module's ``system_prompt = a + b + ...``."""
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "system_prompt" for t in node.targets
        ):
            continue
        names |= {
            operand.id
            for operand in ast.walk(node.value)
            if isinstance(operand, ast.Name)
        }
    return names
