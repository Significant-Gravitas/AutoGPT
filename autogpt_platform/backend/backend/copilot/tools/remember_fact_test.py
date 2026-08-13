"""Tests for RememberFactTool (expert learned-note capture)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .models import ErrorResponse, FactRememberedResponse
from .remember_fact import RememberFactTool

_TEST_USER_ID = "test-user-remember-fact"
_MODULE = "backend.copilot.tools.remember_fact"


def _mock_experts_db(**methods):
    mock_db = MagicMock()
    for name, value in methods.items():
        setattr(mock_db, name, AsyncMock(return_value=value))
    return mock_db, patch(f"{_MODULE}.experts_db", MagicMock(return_value=mock_db))


@pytest.mark.asyncio(loop_scope="session")
async def test_remembers_fact_for_expert_session():
    expert = SimpleNamespace(learned_notes=[SimpleNamespace(id="note-1")])
    mock_db, patcher = _mock_experts_db(append_learned_note=expert)
    with patcher:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await RememberFactTool()._execute(
            user_id=_TEST_USER_ID,
            session=session,
            fact="  Prefers weekly reports on Mondays.  ",
        )
    assert isinstance(resp, FactRememberedResponse)
    assert resp.note_id == "note-1"
    assert resp.fact == "Prefers weekly reports on Mondays."
    assert resp.total_notes == 1
    mock_db.append_learned_note.assert_awaited_once_with(
        _TEST_USER_ID, "exp-1", "Prefers weekly reports on Mondays.", source="chat"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_plain_session_refuses_without_writing():
    mock_db, patcher = _mock_experts_db(append_learned_note=None)
    with patcher:
        session = make_session(_TEST_USER_ID)  # no expert_id
        resp = await RememberFactTool()._execute(
            user_id=_TEST_USER_ID, session=session, fact="anything"
        )
    assert isinstance(resp, ErrorResponse)
    assert "expert" in resp.message.lower()
    mock_db.append_learned_note.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_blank_fact_rejected():
    mock_db, patcher = _mock_experts_db(append_learned_note=None)
    with patcher:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await RememberFactTool()._execute(
            user_id=_TEST_USER_ID, session=session, fact="   "
        )
    assert isinstance(resp, ErrorResponse)
    mock_db.append_learned_note.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_requires_authentication():
    session = make_session(_TEST_USER_ID, expert_id="exp-1")
    resp = await RememberFactTool()._execute(user_id=None, session=session, fact="x")
    assert isinstance(resp, ErrorResponse)
