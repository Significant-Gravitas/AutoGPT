"""Tests for the building-mode restart boundary gate.

``enter_agent_building_mode`` on an SDK turn requests an in-turn restart
with the guide in the system prompt. The restart may only fire at a clean
message boundary — interrupting mid-tool-call would strand ``tool_use``
blocks without results and leave the CLI session file unresumable.
"""

from unittest.mock import MagicMock

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
