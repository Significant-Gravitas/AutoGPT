"""Tests for prompting helpers."""

import importlib

import pytest

from backend.blocks.desktop._api import DISPLAY
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


class TestComputerNote:
    """A plain chat on E2B is told it has a screen; an expert session is told
    by its own ``<expert_computer>`` block instead, and a local session has no
    computer at all."""

    def test_plain_e2b_session_learns_about_the_screen(self):
        result = prompting.get_sdk_supplement(use_e2b=True)
        assert result.count("### Your computer") == 1
        assert "`start_desktop`" in result
        assert "lost when the session expires" in result
        assert "sign into personal accounts" in result

    def test_screen_shows_only_what_runs_in_the_sandbox(self):
        """``browser_*`` drives a browser outside the sandbox, so the model
        must not offer the user a takeover of something the screen never
        shows."""
        result = prompting.get_sdk_supplement(use_e2b=True)
        assert f"DISPLAY={DISPLAY}" in result
        assert "`browser_*` tools run elsewhere" in result

    def test_no_computer_note_without_e2b(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "### Your computer" not in result
        assert "start_desktop" not in result
        assert (
            prompting.get_sdk_supplement(use_e2b=False, expert_session=True) == result
        )

    def test_expert_session_differs_only_by_the_computer_note(self):
        plain = prompting.get_sdk_supplement(use_e2b=True)
        expert = prompting.get_sdk_supplement(use_e2b=True, expert_session=True)
        assert "### Your computer" not in expert
        assert plain.replace(prompting._COMPUTER_NOTE, "") == expert

    def test_note_sits_inside_the_tool_notes_before_the_follow_up_rules(self):
        result = prompting.get_sdk_supplement(use_e2b=True)
        assert (
            result.index("## Tool notes")
            < result.index("### Your computer")
            < result.index("# `<user_follow_up>` blocks")
        )


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

    def test_prompt_contains_rejection_rule(self):
        """This section collects rules from several PRs at once, so a merge
        that takes one side drops a rule silently."""
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "refused a credential the user already has" in result
        assert "Connecting is not running" in result
        assert "The card asks for credentials, not inputs" in result


class TestToolDiscoveryPriorityAntiPattern:
    """The Discovery section must forbid claiming a capability gap without
    calling ``find_capability`` first — this is the regression the
    LinkedIn-skip incident on dev (May 2026) exposed.
    """

    def test_supplement_contains_find_capability_mandatory_language(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        # The header must signal that find_capability is mandatory before
        # any "no integration" reply.
        assert "find_capability` is MANDATORY" in result

    def test_supplement_lists_the_forbidden_phrases(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        # The anti-pattern section must explicitly enumerate the
        # phrases the model emitted in the regression so the model
        # can pattern-match on its own draft and reject it.
        assert "we don't have an X integration" in result
        assert "there's no block for X" in result

    def test_supplement_includes_the_flow_and_no_legacy_names(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        # The numbered flow gives the model a concrete template to follow,
        # not just a prohibition; the retired tools must not be named.
        assert 'find_capability(query="<service>' in result
        assert "describe_capability(id)" in result
        assert "resume_capability(review_id)" in result
        for legacy in ("find_block", "run_block", "run_mcp_tool", "get_mcp_guide"):
            assert legacy not in result, legacy


class TestGraphitiMemoryScope:
    def test_supplement_describes_assistant_scoped_memory(self):
        result = prompting.get_graphiti_supplement()

        assert "scoped to the assistant running this session" in result
        assert "Otto uses the user's personal memory" in result
        assert "each hired expert uses its own separate memory" in result
        assert "Memory is private and isolated to the current assistant" in result
        assert "cannot read each other's memories" in result
        assert "Memory is private to this user — no other user can see it" not in result


class TestTeamBuildingSupplement:
    """``hire_expert`` / ``raise_expert`` are ``expert_admin`` tools, so only a
    plain Otto turn with the team flag on may be told to grow the roster.
    An expert session sees both sides of a delegation but cannot hire."""

    def test_an_autopilot_turn_with_the_flag_on_is_head_of_ai(self):
        result = prompting.get_team_building_supplement(
            experts_enabled=True, expert_id=None
        )

        assert "Building the team" in result
        assert "hire_expert" in result
        assert "raise_expert" in result
        assert "One proposal at a time" in result
        assert "Never hire silently" in result

    def test_an_expert_session_is_not_told_to_hire(self):
        result = prompting.get_team_building_supplement(
            experts_enabled=True, expert_id="expert-a"
        )

        assert result == ""

    def test_the_flag_off_tells_nobody(self):
        assert (
            prompting.get_team_building_supplement(
                experts_enabled=False, expert_id=None
            )
            == ""
        )

    def test_delegation_supplement_no_longer_carries_hiring_rules(self):
        result = prompting.get_delegation_supplement()

        assert "Delegating to a teammate" in result
        assert "Building the team" not in result
        assert "hire_expert" not in result


class TestChatPlatformSupplement:
    """The silence rule belongs to sessions a chat bot opened, and to no
    others: on the web a human is waiting, and silence there is a bug."""

    def test_a_web_session_gets_nothing(self):
        assert prompting.get_chat_platform_supplement(None) == ""
        assert prompting.get_chat_platform_supplement("") == ""

    def test_every_bot_platform_gets_the_same_rule(self):
        rules = {
            prompting.get_chat_platform_supplement(p)
            for p in ("discord", "slack", "telegram", "teams")
        }
        assert len(rules) == 1, "one string, so the prompt cache is shared"
        rule = rules.pop()
        assert f"exactly `{prompting.NO_REPLY}` as your entire message" in rule
        assert "Otherwise answer normally" in rule

    def test_the_word_is_exact_and_case_sensitive(self):
        assert prompting.NO_REPLY == "NO_REPLY"


class TestExpertOversightSupplement:
    """The chat-reading tools are in the ``expert_admin`` group, so only an
    Otto session with the team flag on can call them — a turn that
    cannot must not be told about them."""

    def test_an_autopilot_turn_with_the_flag_on_names_both_tools(self):
        result = prompting.get_expert_oversight_supplement(
            experts_enabled=True, expert_id=None
        )
        assert "list_expert_chats" in result
        assert "read_expert_chat" in result

    def test_an_expert_session_is_told_nothing(self):
        assert (
            prompting.get_expert_oversight_supplement(
                experts_enabled=True, expert_id="expert-a"
            )
            == ""
        )

    def test_the_flag_off_tells_nobody(self):
        assert (
            prompting.get_expert_oversight_supplement(
                experts_enabled=False, expert_id=None
            )
            == ""
        )


class TestSchedulingGuidance:
    """The CLI's cron built-ins are blocked (REQ-121), but blocking alone just
    moves the failure: the model must be told which primitive is durable, and
    told not to promise monitoring it never scheduled.
    """

    def test_supplement_names_the_building_gate_before_it_refuses(self):
        # The gate's refusal used to be the only text naming the tool, so the
        # model met it by being refused and then stalled retrying the entry.
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "call `enter_agent_building_mode` first" in result
        assert "tool:enter_agent_building_mode" not in result

    def test_supplement_names_schedule_followup_by_capability_id(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "### Scheduling future work — use `tool:schedule_followup`" in result
        assert "`tool:schedule_followup` schedules a future copilot turn" in result

    def test_supplement_sends_standing_work_to_a_routine_without_naming_it(self):
        # Routines ride the flag-gated ``expert_resources`` group, so this
        # ungated supplement points at the block that appears alongside them
        # rather than at a tool the session may not be able to call.
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "set up a routine for it rather than a" in result
        assert "schedule_routine" not in result

    def test_supplement_keeps_agent_schedules_on_run_agent(self):
        # "Run my report agent every morning" must stay a graph schedule, not
        # become a recurring copilot turn that re-decides what to run.
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "use `run_agent` with `schedule_name` +" in result
        assert "use `tool:setup_agent_webhook_trigger`" in result

    def test_supplement_rejects_the_confirmed_but_dead_alternative(self):
        result = prompting.get_sdk_supplement(use_e2b=False)
        # CronCreate reports success and claims it persisted to disk, so
        # "it said it worked" must not be treated as evidence it is scheduled.
        assert "even if it reports success and says it persisted to disk" in result
        assert "unless a scheduling" in result
        assert "call actually succeeded" in result

    def test_supplement_describes_list_schedules_scope_honestly(self):
        # list_schedules filters by expert_id, not session_id: it returns the
        # expert's (or plain copilot's) schedules from every chat.
        result = prompting.get_sdk_supplement(use_e2b=False)
        assert "across all chats, not only the ones created here" in result
        assert "current chat's scope" not in result

    def test_baseline_mode_gets_the_same_rule(self):
        # SHARED_TOOL_NOTES feeds both the SDK supplement and baseline's
        # system prompt; the rule is useless if it only reaches one mode.
        assert "### Scheduling future work" in prompting.SHARED_TOOL_NOTES


class TestMathGuidance:
    @pytest.mark.parametrize("use_e2b", [False, True])
    def test_sdk_supplement_tells_the_model_formulas_render(self, use_e2b):
        result = prompting.get_sdk_supplement(use_e2b=use_e2b)
        assert "`$…$` inline, `$$…$$` for display" in result
