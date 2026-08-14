"""Tests for the two-step Soul edit flow (preview + confirm-by-id)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .models import ErrorResponse, ExpertSoulUpdatedResponse
from .update_soul import ConfirmExpertSoulUpdateTool, UpdateExpertSoulTool

_TEST_USER_ID = "test-user-update-soul"
_MODULE = "backend.copilot.tools.update_soul"


class _FakeRedis:
    """Minimal async Redis: just the setex/get/delete surface the tools use."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def setex(self, key, ttl, value):
        self.store[key] = value

    async def get(self, key):
        value = self.store.get(key)
        return value.encode() if value is not None else None

    async def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


def _current_expert():
    return SimpleNamespace(
        identity="Old identity.",
        voice_preferences="Old voice.",
        boundaries="Old boundaries.",
    )


def _mock_env(*, expert=None, updated=None, redis=None):
    mock_db = MagicMock()
    mock_db.get_expert = AsyncMock(return_value=expert)
    mock_db.update_soul_fields = AsyncMock(return_value=updated)
    patchers = [
        patch(f"{_MODULE}.experts_db", MagicMock(return_value=mock_db)),
        patch(
            f"{_MODULE}.get_redis_async",
            AsyncMock(return_value=redis or _FakeRedis()),
        ),
    ]
    return mock_db, patchers


async def _preview(session, **fields):
    return await UpdateExpertSoulTool()._execute(
        user_id=_TEST_USER_ID, session=session, **fields
    )


async def _confirm(session, **kwargs):
    return await ConfirmExpertSoulUpdateTool()._execute(
        user_id=_TEST_USER_ID, session=session, **kwargs
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_never_writes_even_with_confirm_true():
    """The old confirm=true shortcut must not exist: the preview tool has no
    write path, so a model-supplied confirm flag changes nothing."""
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session, voice_preferences="New voice.", confirm=True)
    assert isinstance(resp, ExpertSoulUpdatedResponse)
    assert resp.applied is False
    assert resp.confirmation_id
    assert [(c.field, c.before, c.after) for c in resp.changes] == [
        ("voice_preferences", "Old voice.", "New voice.")
    ]
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_applies_exact_previewed_proposal():
    mock_db, patchers = _mock_env(expert=_current_expert(), updated=SimpleNamespace())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        resp = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ExpertSoulUpdatedResponse)
    assert resp.applied is True
    assert [(c.field, c.before, c.after) for c in resp.changes] == [
        ("voice_preferences", "Old voice.", "New voice.")
    ]
    mock_db.update_soul_fields.assert_awaited_once_with(
        _TEST_USER_ID, "exp-1", voice_preferences="New voice."
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_field_values():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        resp = await _confirm(
            session,
            confirmation_id=preview.confirmation_id,
            identity="Sneaky replacement.",
        )
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirmation_id_is_single_use():
    mock_db, patchers = _mock_env(expert=_current_expert(), updated=SimpleNamespace())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        first = await _confirm(session, confirmation_id=preview.confirmation_id)
        second = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(first, ExpertSoulUpdatedResponse) and first.applied is True
    assert isinstance(second, ErrorResponse)
    mock_db.update_soul_fields.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_stale_before_state():
    """A Soul that changed between preview and confirm discards the proposal."""
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        mock_db.get_expert.return_value = SimpleNamespace(
            identity="Old identity.",
            voice_preferences="Changed elsewhere meanwhile.",
            boundaries="Old boundaries.",
        )
        resp = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ErrorResponse)
    assert "changed since" in resp.message
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_unknown_id():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _confirm(session, confirmation_id="never-issued")
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_proposal_from_other_session():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session_a = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session_a, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        session_b = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _confirm(session_b, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_plain_session_refuses():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID)  # no expert_id
        resp = await _preview(session, identity="x")
    assert isinstance(resp, ErrorResponse)
    mock_db.get_expert.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_plain_session_refuses():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID)  # no expert_id
        resp = await _confirm(session, confirmation_id="anything")
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_no_fields_rejected():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session)
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_noop_when_values_match_current():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session, voice_preferences="Old voice.")
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_missing_expert_errors():
    mock_db, patchers = _mock_env(expert=None)
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session, identity="x")
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_both_tools_require_authentication():
    session = make_session(_TEST_USER_ID, expert_id="exp-1")
    preview = await UpdateExpertSoulTool()._execute(
        user_id=None, session=session, identity="x"
    )
    confirm = await ConfirmExpertSoulUpdateTool()._execute(
        user_id=None, session=session, confirmation_id="anything"
    )
    assert isinstance(preview, ErrorResponse)
    assert isinstance(confirm, ErrorResponse)
