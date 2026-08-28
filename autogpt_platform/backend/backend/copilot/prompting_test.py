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


class TestFlaggedExpertOperatingPolicies:
    def test_flag_off_adds_zero_bytes_to_the_assembled_prompt(self):
        unchanged = "base-system\nshared-tools"
        assembled = unchanged + prompting.get_expert_team_supplement(
            experts_enabled=False,
            expert_id=None,
        )

        assert assembled.encode() == unchanged.encode()

    def test_autopilot_receives_head_of_ai_policy(self):
        result = prompting.get_expert_team_supplement(
            experts_enabled=True,
            expert_id=None,
        )

        assert "Head of AI operating policy" in result
        assert "own the founder's overall outcome" in result
        assert "Delegate independent work concurrently" in result
        assert "founder must never have to coordinate, poll, or nudge" in result
        assert "Expert employee operating policy" not in result

    def test_expert_receives_employee_policy_without_staffing_permission(self):
        result = prompting.get_expert_team_supplement(
            experts_enabled=True,
            expert_id="expert-1",
        )

        assert "Expert employee operating policy" in result
        assert "report to AutoPilot" in result
        assert "Do not hire, raise, edit, or otherwise staff teammates" in result
        assert "Head of AI operating policy" not in result

    def test_manager_policy_limits_questions_and_keeps_other_work_moving(self):
        result = prompting.get_delegation_supplement()

        flattened = " ".join(result.split())
        assert (
            "Ask only for information or credentials only the founder holds"
            in flattened
        )
        assert "does not pause independent work" in result
        assert "reasonable degraded-result fallback" in result
        assert "Stop on a verified hard failure" in result

    def test_manager_uses_fresh_phase_for_new_work(self):
        result = prompting.get_delegation_supplement()

        assert "create a new task phase" in result
        assert "fresh criteria, owners, and estimates" in result

    def test_work_item_wake_replaces_long_polling(self):
        result = prompting.get_delegation_supplement()

        assert "do not spin in a polling turn" in result
        assert "wakes its manager once" in result

    def test_manager_uses_deterministic_verification_truth(self):
        result = " ".join(prompting.get_delegation_supplement().split())

        assert "A required node failure means the workflow test failed" in result
        assert "Never describe a failed test as verified" in result
        assert "missing required artifact means incomplete" in result

    def test_manager_prefers_workspace_delivery_without_unneeded_credentials(self):
        result = " ".join(prompting.get_delegation_supplement().split())

        assert "A workspace deliverable does not require an external SaaS" in result
        assert "Do not ask for a credential" in result

    def test_direct_expert_routes_work_without_making_founder_coordinate(self):
        result = " ".join(prompting.get_delegation_supplement("expert-1").split())

        assert "route it with `delegate_to_expert`" in result
        assert "Never tell the founder to forward" in result


class TestGraphitiMemoryScope:
    def test_supplement_describes_assistant_scoped_memory(self):
        result = prompting.get_graphiti_supplement()

        assert "scoped to the assistant running this session" in result
        assert "AutoPilot uses the user's personal memory" in result
        assert "each hired expert uses its own separate memory" in result
        assert "Memory is private and isolated to the current assistant" in result
        assert "cannot read each other's memories" in result
        assert "Memory is private to this user — no other user can see it" not in result
