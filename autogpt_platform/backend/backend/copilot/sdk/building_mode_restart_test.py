"""Tests for the building-mode restart boundary gate.

``enter_agent_building_mode`` on an SDK turn requests an in-turn restart
with the guide in the system prompt. The restart may only fire at a clean
message boundary — interrupting mid-tool-call would strand ``tool_use``
blocks without results and leave the CLI session file unresumable.
"""

from unittest.mock import MagicMock

import pytest

from backend.copilot.model import ChatSession

from . import service
from .service import _graphiti_ingest_allowed, _ready_for_building_mode_restart
from .system_prompt import SystemPromptParts, assemble_system_prompt


def _session(*, requested: bool = True, guide_loaded: bool = False) -> ChatSession:
    session = ChatSession.new(user_id="user-1", dry_run=False)
    session.building_mode_requested = requested
    session.guide_in_system_prompt = guide_loaded
    return session


def _parts(**overrides) -> SystemPromptParts:
    fields = {
        "base": "BASE",
        "use_e2b": False,
        "expert_session": False,
        "delegation": "<delegation>RULES</delegation>",
        "oversight": "<oversight>RULES</oversight>",
        "team_building": "",
        "chat_platform": "",
        "graphiti": "",
        "auto_mode": "\n\n<auto_mode>RULES</auto_mode>",
        "expert_identity": "",
    }
    return SystemPromptParts(**(fields | overrides))


def _prompt_text(prompt: str | dict) -> str:
    if isinstance(prompt, str):
        return prompt
    return prompt["prompt"] if prompt["type"] == "custom" else prompt["append"]


def _adapter(*, unresolved: bool) -> MagicMock:
    adapter = MagicMock()
    adapter.has_unresolved_tool_calls = unresolved
    return adapter


def test_fires_at_clean_boundary():
    assert (
        _ready_for_building_mode_restart(_session(), _adapter(unresolved=False)) is True
    )


def test_must_not_fire_mid_tool_call():
    assert (
        _ready_for_building_mode_restart(_session(), _adapter(unresolved=True)) is False
    )


def test_noop_without_request():
    assert (
        _ready_for_building_mode_restart(
            _session(requested=False), _adapter(unresolved=False)
        )
        is False
    )


def test_noop_once_guide_already_loaded():
    assert (
        _ready_for_building_mode_restart(
            _session(guide_loaded=True), _adapter(unresolved=False)
        )
        is False
    )


class TestApplyBuildingModeRestart:
    """Execution tests for the restart application: prompt rebuild, resume
    wiring, flag transitions, and adapter carry-over."""

    def _state(self, *, prior_emitted: bool, thinking_reprompted: bool):
        state = MagicMock()
        state.thinking_only_reprompted = thinking_reprompted
        state.adapter = MagicMock()
        state.adapter.emitted_real_content_to_wire = prior_emitted
        return state

    async def _run(
        self,
        mocker,
        *,
        suffix: str = "\n\n<building_guide>GUIDE</building_guide>",
        prior_emitted: bool = False,
        thinking_reprompted: bool = False,
        prompt_parts: SystemPromptParts | None = None,
        expert_id: str | None = None,
    ):
        from backend.copilot.sdk.service import (
            _BUILDING_MODE_CONTINUATION,
            _apply_building_mode_restart,
        )

        mocker.patch(
            "backend.copilot.sdk.service.build_builder_system_prompt_suffix",
            new=mocker.AsyncMock(return_value=suffix),
        )
        prompt_parts = prompt_parts or _parts()
        session = _session(requested=True, guide_loaded=False)
        session.expert_id = expert_id
        state = self._state(
            prior_emitted=prior_emitted, thinking_reprompted=thinking_reprompted
        )
        sdk_options = MagicMock()
        status = await _apply_building_mode_restart(
            session=session,
            state=state,
            sdk_options=sdk_options,
            prompt_parts=prompt_parts,
            session_id="sess-1",
            message_id="msg-1",
            log_prefix="[test]",
        )
        return session, state, status, _BUILDING_MODE_CONTINUATION

    @pytest.mark.asyncio
    async def test_flags_resume_and_prompt_wiring(self, mocker):
        session, state, status, continuation = await self._run(mocker)

        assert session.building_mode_requested is False
        assert session.guide_in_system_prompt is True
        assert state.options.resume == "sess-1"
        assert state.options.session_id is None
        assert state.use_resume is True
        assert state.query_message == continuation
        assert "building mode" in status.message.lower()

    @pytest.mark.asyncio
    async def test_restart_prompt_is_the_launch_prompt_plus_the_guide(self, mocker):
        """The restart resumes the turn's session with the guide added and
        nothing else changed: the delegation, oversight and auto-mode rules
        stay, because their tools and gates stay active for the rest of the
        turn."""
        guide = "\n\n<building_guide>GUIDE</building_guide>"
        parts = _parts()
        launch = assemble_system_prompt(parts, building_suffix="")

        _, state, _, _ = await self._run(mocker, suffix=guide, prompt_parts=parts)

        assert _prompt_text(state.options.system_prompt) == launch + guide

    @pytest.mark.asyncio
    async def test_expert_identity_is_the_turns_own(self, mocker):
        """The restart re-validates the expert but keeps the identity the
        turn launched with, so its prompt is still the launch prompt plus the
        guide."""
        identity = mocker.patch(
            "backend.copilot.sdk.service.build_expert_identity_suffix",
            new=mocker.AsyncMock(
                return_value="<expert_identity>RENAMED</expert_identity>"
            ),
        )
        parts = _parts(
            expert_session=True,
            expert_identity="<expert_identity>LAUNCHED</expert_identity>",
        )

        _, state, _, _ = await self._run(
            mocker, suffix="<guide/>", prompt_parts=parts, expert_id="expert-1"
        )

        identity.assert_awaited_once()
        assert _prompt_text(state.options.system_prompt) == assemble_system_prompt(
            parts, building_suffix="<guide/>"
        )

    @pytest.mark.asyncio
    async def test_restart_prompt_is_live(self, mocker):
        """The restart resumes the session the interrupted run was writing,
        so with the CLI's prompt snapshot on, the prompt that session recorded
        on its first request, without the guide, would be sent instead."""
        mocker.patch.object(service.config, "claude_agent_live_system_prompt", True)

        _, state, _, _ = await self._run(mocker)

        assert state.options.system_prompt["snapshot"] is False

    @pytest.mark.asyncio
    async def test_live_prompt_switch_off_restores_the_raw_prompt(self, mocker):
        mocker.patch.object(service.config, "claude_agent_live_system_prompt", False)

        _, state, _, _ = await self._run(mocker)

        assert isinstance(state.options.system_prompt, str)
        assert "<building_guide>GUIDE</building_guide>" in state.options.system_prompt

    @pytest.mark.asyncio
    async def test_empty_suffix_relaunches_without_the_confirmation(self, mocker):
        """An empty suffix means the guide is genuinely absent, so the model
        must not be told it is present — that sentence costs it the rest of
        the turn chasing a gate that cannot clear."""
        session, state, _, continuation = await self._run(mocker, suffix="")

        assert state.query_message != continuation
        assert "could not be loaded" in state.query_message
        assert session.guide_in_system_prompt is False
        # Cleared as on the success path, which is what stops the restart
        # re-firing at the next message boundary of this turn.
        assert session.building_mode_requested is False
        # The relaunch itself still proceeds — resume wiring is unconditional.
        assert state.use_resume is True

    @pytest.mark.asyncio
    async def test_guide_applied_although_history_lacks_the_enter_call(self, mocker):
        """Production shape: the restart runs microseconds after the enter
        tool ran, before its row is in ``messages``. Deriving "is this session
        building?" from history there answers False and strands the turn with
        no guide — this is the case dev logged 16 times in six hours.

        The real suffix builder runs here on purpose: patching it would prove
        only the wiring, never that the predicate underneath it answers.
        """
        from backend.copilot.sdk.service import (
            _BUILDING_MODE_CONTINUATION,
            _apply_building_mode_restart,
        )

        session = _session(requested=True, guide_loaded=False)
        assert session.messages == []
        assert session.has_tool_been_called("enter_agent_building_mode") is False
        state = self._state(prior_emitted=False, thinking_reprompted=False)
        mocker.patch(
            "backend.copilot.builder_context._load_guide",
            return_value="# Guide body",
        )

        await _apply_building_mode_restart(
            session=session,
            state=state,
            sdk_options=MagicMock(),
            prompt_parts=_parts(),
            session_id="sess-1",
            message_id="msg-1",
            log_prefix="[test]",
        )

        text = _prompt_text(state.options.system_prompt)
        assert "<building_guide>" in text
        assert "# Guide body" in text
        assert session.guide_in_system_prompt is True
        assert session.building_mode_requested is False
        assert state.query_message == _BUILDING_MODE_CONTINUATION

    @pytest.mark.asyncio
    async def test_adapter_carry_over(self, mocker):
        _, state, _, _ = await self._run(
            mocker, prior_emitted=True, thinking_reprompted=True
        )
        assert state.adapter.thinking_only_reprompted is True
        assert state.adapter.prior_attempt_emitted_visible_content is True

    @pytest.mark.asyncio
    async def test_no_carry_over_without_prior_content(self, mocker):
        _, state, _, _ = await self._run(mocker)
        assert state.adapter.thinking_only_reprompted is False
        assert state.adapter.prior_attempt_emitted_visible_content is False

    @pytest.mark.asyncio
    async def test_failed_expert_revalidation_keeps_ingest_gated(self, mocker):
        from backend.copilot.expert_context import ExpertSessionUnavailableError
        from backend.copilot.sdk.service import _apply_building_mode_restart

        session = _session(requested=True, guide_loaded=False)
        session.expert_id = "expert-1"
        state = self._state(prior_emitted=False, thinking_reprompted=False)
        mocker.patch(
            "backend.copilot.sdk.service.build_builder_system_prompt_suffix",
            new=mocker.AsyncMock(return_value="<building_guide>GUIDE</building_guide>"),
        )
        identity = mocker.patch(
            "backend.copilot.sdk.service.build_expert_identity_suffix",
            new=mocker.AsyncMock(
                side_effect=ExpertSessionUnavailableError("expert unavailable")
            ),
        )

        with pytest.raises(ExpertSessionUnavailableError):
            await _apply_building_mode_restart(
                session=session,
                state=state,
                sdk_options=MagicMock(),
                prompt_parts=_parts(),
                session_id="sess-1",
                message_id="msg-1",
                log_prefix="[test]",
            )

        identity.assert_awaited_once_with(
            "user-1", "expert-1", organization_id=None, team_id=None
        )
        assert not _graphiti_ingest_allowed(
            expert_identity_validated=False,
            graphiti_enabled=True,
            user_id="user-1",
            message="private prompt",
            is_user_message=True,
        )
