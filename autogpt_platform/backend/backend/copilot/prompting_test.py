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


class TestGraphitiMemoryScope:
    def test_autopilot_supplement_describes_assistant_scoped_memory(self):
        result = prompting.get_graphiti_supplement("autopilot")

        assert "scoped to the assistant running this session" in result
        assert "You are AutoPilot and use the user's personal memory" in result
        assert "each hired expert keeps its own" in result
        assert "Memory is private and isolated to the current assistant" in result
        assert "cannot read each other's memories" in result
        assert "Memory is private to this user — no other user can see it" not in result

    def test_expert_supplement_frames_memory_as_own_experience(self):
        result = prompting.get_graphiti_supplement("expert")

        assert "scoped to the assistant running this session" in result
        assert "your accumulated professional experience" in result
        assert "AutoPilot and other experts cannot read it" in result
        assert "You are AutoPilot" not in result


class TestRoleAwareTeamSections:
    """Autopilot gets a head-of-team charter; an expert session gets employee
    operating rules. Both roles are told about ``escalate_task`` targets so a
    blocked chain routes up instead of stalling."""

    def test_autopilot_charter_is_head_of_staff(self):
        charter = prompting.get_role_charter("autopilot")
        assert "head of the user's team" in charter
        assert "hire_expert" in charter
        assert "Operating as a hired expert" not in charter

    def test_expert_charter_is_employee_rules(self):
        charter = prompting.get_role_charter("expert")
        assert "Operating as a hired expert" in charter
        assert 'escalate_task(target="manager")' in charter
        assert 'escalate_task(target="user")' in charter
        assert "hire_expert" not in charter

    def test_delegation_supplement_routes_escalation_by_role(self):
        expert = prompting.get_delegation_supplement("expert")
        autopilot = prompting.get_delegation_supplement("autopilot")
        assert 'escalate_task(target="manager")' in expert
        assert "Experts may escalate their blocked tasks to you" in autopilot

    def test_assemble_orders_static_before_identity(self):
        prompt = prompting.assemble_system_prompt(
            "BASE",
            engine_supplement="ENGINE",
            delegation_supplement=prompting.get_delegation_supplement("expert"),
            graphiti_supplement=prompting.get_graphiti_supplement("expert"),
            role_charter=prompting.get_role_charter("expert"),
            builder_session_suffix="<builder/>",
            expert_session_suffix="<expert_identity/>",
        )
        assert (
            prompt.index("ENGINE")
            < prompt.index("### Delegating to a teammate")
            < prompt.index("## Memory System (Graphiti)")
            < prompt.index("## Operating as a hired expert")
            < prompt.index("<builder/>")
            < prompt.index("<expert_identity/>")
        )
