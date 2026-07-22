"""Tests for the building-mode restart boundary gate.

``enter_agent_building_mode`` on an SDK turn requests an in-turn restart
with the guide in the system prompt. The restart may only fire at a clean
message boundary — interrupting mid-tool-call would strand ``tool_use``
blocks without results and leave the CLI session file unresumable.
"""

from unittest.mock import MagicMock

import pytest

from backend.copilot.model import ChatSession

from .service import _ready_for_building_mode_restart


def _session(*, requested: bool = True, guide_loaded: bool = False) -> ChatSession:
    session = ChatSession.new(user_id="user-1", dry_run=False)
    session.building_mode_requested = requested
    session.guide_in_system_prompt = guide_loaded
    return session


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
    ):
        from backend.copilot.sdk.service import (
            _BUILDING_MODE_CONTINUATION,
            _apply_building_mode_restart,
        )

        mocker.patch(
            "backend.copilot.sdk.service.build_builder_system_prompt_suffix",
            new=mocker.AsyncMock(return_value=suffix),
        )
        session = _session(requested=True, guide_loaded=False)
        state = self._state(
            prior_emitted=prior_emitted, thinking_reprompted=thinking_reprompted
        )
        sdk_options = MagicMock()
        status = await _apply_building_mode_restart(
            session=session,
            state=state,
            sdk_options=sdk_options,
            base_system_prompt="BASE",
            graphiti_supplement="",
            use_e2b=False,
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
        assert state.resume_file == "sess-1"
        assert state.query_message == continuation
        assert "building mode" in status.message.lower()

    @pytest.mark.asyncio
    async def test_empty_suffix_degrades_without_prompt_upgrade(self, mocker):
        session, state, _, _ = await self._run(mocker, suffix="")

        assert session.building_mode_requested is False
        assert session.guide_in_system_prompt is False
        # The restart still proceeds — resume wiring is unconditional.
        assert state.use_resume is True

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
