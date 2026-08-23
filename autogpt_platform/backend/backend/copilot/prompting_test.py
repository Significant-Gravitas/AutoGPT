"""Tests for prompting helpers."""

import importlib

from backend.api.features.experts.models import EXTERNAL_ACTION_APPROVAL_RULE
from backend.copilot import prompting


def flat(text: str) -> str:
    """Collapse wrapped prose so assertions survive re-flowed paragraphs."""
    return " ".join(text.split())


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


class TestDriveToCompletionIsAlwaysOn:
    """The general "never hand back half-done work" rules must live in
    ``SHARED_TOOL_NOTES`` so they reach BOTH engines on every turn — the
    baseline path concatenates the constant directly and the SDK path
    embeds it via ``get_sdk_supplement``. Keeping them in the delegation
    supplement would gate them behind ``Flag.HIRE_EXPERTS``.
    """

    def setup_method(self):
        importlib.reload(prompting)

    def test_shared_notes_contain_finishing_the_job_section(self):
        assert (
            "Finishing the job — never hand back half-done work"
            in prompting.SHARED_TOOL_NOTES
        )

    def test_shared_notes_name_next_actor_for_pending_work(self):
        assert "what happens next and who acts next" in flat(
            prompting.SHARED_TOOL_NOTES
        )

    def test_shared_notes_forbid_check_back_later_endings(self):
        assert '"check back later"' in prompting.SHARED_TOOL_NOTES

    def test_local_sdk_supplement_includes_finishing_the_job(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "Finishing the job — never hand back half-done work" in result

    def test_e2b_sdk_supplement_includes_finishing_the_job(self):
        result = prompting.get_sdk_supplement(use_e2b=True)
        assert "Finishing the job — never hand back half-done work" in result

    def test_drive_to_completion_is_not_gated_behind_delegation(self):
        # The general rule must NOT be confined to the flagged supplement.
        assert (
            "Finishing the job — never hand back half-done work"
            not in prompting.get_delegation_supplement()
        )


class TestGrantedPermissionIsNotReAsked:
    """After the user approves a plan, reversible next steps proceed without a
    fresh confirmation gate; only irreversible/external actions re-gate. The
    wording tracks ``PROTECTED_SOUL_RULES.EXTERNAL_ACTION_APPROVAL_RULE``.
    """

    def setup_method(self):
        importlib.reload(prompting)

    def test_shared_notes_contain_no_re_ask_section(self):
        assert (
            "Permission already granted — do not re-ask" in prompting.SHARED_TOOL_NOTES
        )

    def test_shared_notes_list_approval_phrases(self):
        assert '"confirm all"' in prompting.SHARED_TOOL_NOTES
        assert '"go with option 1"' in prompting.SHARED_TOOL_NOTES

    def test_shared_notes_keep_gate_for_irreversible_actions(self):
        notes = prompting.SHARED_TOOL_NOTES
        assert "irreversible or external actions" in notes
        assert "sending, publishing, deploying, spending" in notes

    def test_wording_stays_consistent_with_protected_soul_rules(self):
        # "External actions require approval." — restated verbatim so the
        # copilot prompt and the expert soul rules cannot drift apart.
        assert EXTERNAL_ACTION_APPROVAL_RULE in prompting.SHARED_TOOL_NOTES

    def test_rule_reaches_the_sdk_supplement(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "Permission already granted — do not re-ask" in result


class TestDelegationWaitingEtiquette:
    """While a sub-session is still running the model owns the wait: one short
    status line, then either an automatic wake-up or a self-scheduled
    ``schedule_followup`` — never a question to the user about polling.
    """

    def setup_method(self):
        importlib.reload(prompting)

    def test_supplement_contains_waiting_etiquette_rule(self):
        assert "Waiting etiquette." in prompting.get_delegation_supplement()

    def test_waiting_rule_names_schedule_followup_as_the_resume_path(self):
        supplement = prompting.get_delegation_supplement()
        assert "`schedule_followup` with this session's `session_id`" in supplement
        assert "60s minimum" in flat(supplement)

    def test_waiting_rule_promises_no_resume_mechanism_that_does_not_exist(self):
        """Nothing enqueues a turn back onto the parent when a sub finishes —
        a delegation that outlives its turn resumes only because the model
        scheduled a followup. Promising an automatic wake-up here would let
        the model end the turn with no resume at all, which is worse than the
        babysitting these rules exist to remove. The PR that adds the wake
        mechanism updates this assertion along with the wording."""
        supplement = flat(prompting.get_delegation_supplement())

        assert "woken automatically" not in supplement
        assert "wakes this chat" not in supplement

    def test_waiting_rule_requires_status_line_with_eta(self):
        supplement = prompting.get_delegation_supplement()
        assert "ONE short status line" in supplement
        assert "include the ETA when you know it" in supplement

    def test_waiting_rule_forbids_asking_the_user_to_wait(self):
        supplement = prompting.get_delegation_supplement()
        assert "Never ask the user whether to keep polling" in flat(supplement)
        assert "never tell them to check back later" in flat(supplement)

    def test_supplement_caps_poll_narration(self):
        supplement = prompting.get_delegation_supplement()
        assert "One status line per wait cycle." in supplement
        assert 'skip repeated "still working…" filler' in supplement

    def test_poll_cap_points_at_the_sub_session_card(self):
        assert "Live progress already renders on the sub-session card above" in flat(
            prompting.get_delegation_supplement()
        )


class TestDurablePreferenceCapture:
    """A mid-task correction with lasting intent ("always use X") is a rule to
    persist immediately, not a one-off instruction to follow and forget.
    """

    def setup_method(self):
        importlib.reload(prompting)

    def test_store_triggers_include_mid_task_corrections(self):
        assert "Mid-task corrections" in prompting.get_graphiti_supplement()

    def test_trigger_lists_the_durable_intent_phrases(self):
        supplement = prompting.get_graphiti_supplement()
        assert '"always use X"' in supplement
        assert '"never do Y"' in supplement
        assert '"from now on…"' in supplement

    def test_trigger_specifies_the_real_memory_store_schema(self):
        supplement = prompting.get_graphiti_supplement()
        assert '`memory_kind="rule"`' in supplement
        assert "`instruction`" in supplement
        assert "`actor`/`trigger`/`negation`" in supplement

    def test_trigger_requires_a_one_line_acknowledgement(self):
        supplement = prompting.get_graphiti_supplement()
        assert "acknowledge in ONE line" in supplement
        assert "Noted — Codex for all code tasks from now on." in supplement

    def test_trigger_forbids_deferring_the_write(self):
        assert (
            "store immediately, do not batch to the end"
            in prompting.get_graphiti_supplement()
        )


class TestGraphitiMemoryScope:
    def test_supplement_describes_assistant_scoped_memory(self):
        result = prompting.get_graphiti_supplement()

        assert "scoped to the assistant running this session" in result
        assert "AutoPilot uses the user's personal memory" in result
        assert "each hired expert uses its own separate memory" in result
        assert "Memory is private and isolated to the current assistant" in result
        assert "cannot read each other's memories" in result
        assert "Memory is private to this user — no other user can see it" not in result
