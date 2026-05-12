"""Tests for prompting helpers."""

import importlib

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
