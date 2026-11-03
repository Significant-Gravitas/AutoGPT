"""Tests for ListTeamTool — the model's live view of the roster."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ._test_data import make_session
from .list_team import ListTeamTool
from .models import ErrorResponse, TeamRosterResponse

_USER = "user-list-team"


def _expert(
    expert_id: str = "exp-1",
    name: str = "Vera",
    *,
    is_archived: bool = False,
    schedules_paused_at=None,
):
    return SimpleNamespace(
        id=expert_id,
        name=name,
        role="Market research",
        color="violet-300",
        avatar_url=None,
        is_archived=is_archived,
        schedules_paused_at=schedules_paused_at,
        workflows=[],
        credential_count=0,
    )


def _patch_db(monkeypatch, experts):
    db = MagicMock()
    db.list_experts = AsyncMock(return_value=experts)
    monkeypatch.setattr(
        "backend.copilot.tools.list_team.experts_db", lambda: db, raising=True
    )
    return db


@pytest.mark.asyncio
async def test_lists_active_experts_with_ids(monkeypatch):
    _patch_db(monkeypatch, [_expert(), _expert("exp-2", "Max")])
    result = await ListTeamTool()._execute(user_id=_USER, session=make_session(_USER))
    assert isinstance(result, TeamRosterResponse)
    assert [e.id for e in result.experts] == ["exp-1", "exp-2"]
    assert "Vera" in result.message
    assert "exp-2" in result.message


@pytest.mark.asyncio
async def test_archived_experts_are_excluded(monkeypatch):
    _patch_db(monkeypatch, [_expert(), _expert("exp-9", "Old", is_archived=True)])
    result = await ListTeamTool()._execute(user_id=_USER, session=make_session(_USER))
    assert isinstance(result, TeamRosterResponse)
    assert [e.id for e in result.experts] == ["exp-1"]
    assert "Old" not in result.message


@pytest.mark.asyncio
async def test_paused_experts_are_flagged(monkeypatch):
    _patch_db(
        monkeypatch,
        [_expert(schedules_paused_at="2026-01-01T00:00:00Z")],
    )
    result = await ListTeamTool()._execute(user_id=_USER, session=make_session(_USER))
    assert isinstance(result, TeamRosterResponse)
    assert result.experts[0].is_paused is True
    assert "[paused]" in result.message


@pytest.mark.asyncio
async def test_empty_team_says_so(monkeypatch):
    _patch_db(monkeypatch, [])
    result = await ListTeamTool()._execute(user_id=_USER, session=make_session(_USER))
    assert isinstance(result, TeamRosterResponse)
    assert result.experts == []
    assert "empty" in result.message


@pytest.mark.asyncio
async def test_lookup_failure_returns_error(monkeypatch):
    db = MagicMock()
    db.list_experts = AsyncMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(
        "backend.copilot.tools.list_team.experts_db", lambda: db, raising=True
    )
    result = await ListTeamTool()._execute(user_id=_USER, session=make_session(_USER))
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_requires_auth():
    result = await ListTeamTool()._execute(user_id=None, session=make_session(_USER))
    assert isinstance(result, ErrorResponse)


@pytest.mark.asyncio
async def test_autopilot_sees_each_teammates_grant_count(monkeypatch):
    _patch_db(monkeypatch, [_expert_with_credentials(3)])
    result = await ListTeamTool()._execute(user_id=_USER, session=make_session(_USER))
    assert isinstance(result, TeamRosterResponse)
    assert "3 credential(s)" in result.message
    assert result.experts[0].credential_count == 3


@pytest.mark.asyncio
async def test_an_expert_is_not_told_how_much_access_its_teammates_hold(monkeypatch):
    """An expert picks who to delegate to from the workflow count; a teammate's
    grant count is the owner's view of the team, not an expert's."""
    _patch_db(monkeypatch, [_expert_with_credentials(3)])
    result = await ListTeamTool()._execute(
        user_id=_USER, session=make_session(_USER, expert_id="exp-9")
    )
    assert isinstance(result, TeamRosterResponse)
    assert "credential(s)" not in result.message
    assert result.experts[0].credential_count is None
    # The rest of the row still reaches the expert.
    assert "0 workflow(s)" in result.message


def _expert_with_credentials(count: int):
    expert = _expert()
    expert.credential_count = count
    return expert
