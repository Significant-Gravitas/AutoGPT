"""Tests for UpdateExpertSoulTool (confirm-gated Soul edits)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .models import ErrorResponse, ExpertSoulUpdatedResponse
from .update_soul import UpdateExpertSoulTool

_TEST_USER_ID = "test-user-update-soul"
_MODULE = "backend.copilot.tools.update_soul"


def _current_expert():
    return SimpleNamespace(
        identity="Old identity.",
        voice_preferences="Old voice.",
        boundaries="Old boundaries.",
    )


def _mock_experts_db(*, expert=None, updated=None):
    mock_db = MagicMock()
    mock_db.get_expert = AsyncMock(return_value=expert)
    mock_db.update_soul_fields = AsyncMock(return_value=updated)
    return mock_db, patch(f"{_MODULE}.experts_db", MagicMock(return_value=mock_db))


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_applies_patch_and_returns_diff():
    mock_db, patcher = _mock_experts_db(
        expert=_current_expert(), updated=SimpleNamespace()
    )
    with patcher:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await UpdateExpertSoulTool()._execute(
            user_id=_TEST_USER_ID,
            session=session,
            voice_preferences="New voice.",
            confirm=True,
        )
    assert isinstance(resp, ExpertSoulUpdatedResponse)
    assert resp.applied is True
    assert [(c.field, c.before, c.after) for c in resp.changes] == [
        ("voice_preferences", "Old voice.", "New voice.")
    ]
    mock_db.update_soul_fields.assert_awaited_once_with(
        _TEST_USER_ID, "exp-1", voice_preferences="New voice."
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_confirm_previews_without_writing():
    mock_db, patcher = _mock_experts_db(expert=_current_expert())
    with patcher:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await UpdateExpertSoulTool()._execute(
            user_id=_TEST_USER_ID,
            session=session,
            identity="New identity.",
        )
    assert isinstance(resp, ExpertSoulUpdatedResponse)
    assert resp.applied is False
    assert [c.field for c in resp.changes] == ["identity"]
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_plain_session_refuses():
    mock_db, patcher = _mock_experts_db(expert=_current_expert())
    with patcher:
        session = make_session(_TEST_USER_ID)  # no expert_id
        resp = await UpdateExpertSoulTool()._execute(
            user_id=_TEST_USER_ID, session=session, identity="x", confirm=True
        )
    assert isinstance(resp, ErrorResponse)
    mock_db.get_expert.assert_not_called()
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_no_fields_rejected():
    mock_db, patcher = _mock_experts_db(expert=_current_expert())
    with patcher:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await UpdateExpertSoulTool()._execute(
            user_id=_TEST_USER_ID, session=session, confirm=True
        )
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_noop_when_values_match_current():
    mock_db, patcher = _mock_experts_db(expert=_current_expert())
    with patcher:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await UpdateExpertSoulTool()._execute(
            user_id=_TEST_USER_ID,
            session=session,
            voice_preferences="Old voice.",  # unchanged
            confirm=True,
        )
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_expert_errors():
    mock_db, patcher = _mock_experts_db(expert=None)
    with patcher:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await UpdateExpertSoulTool()._execute(
            user_id=_TEST_USER_ID, session=session, identity="x", confirm=True
        )
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_requires_authentication():
    session = make_session(_TEST_USER_ID, expert_id="exp-1")
    resp = await UpdateExpertSoulTool()._execute(
        user_id=None, session=session, identity="x", confirm=True
    )
    assert isinstance(resp, ErrorResponse)
