"""Tests for the two-step Soul edit flow (preview + confirm-by-id)."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._test_data import make_session
from .models import ErrorResponse, ExpertSoulUpdatedResponse
from .soul_proposal import proposal_key
from .update_soul import ConfirmExpertSoulUpdateTool, UpdateExpertSoulTool

_TEST_USER_ID = "test-user-update-soul"
_MODULE = "backend.copilot.tools.update_soul"
_PROPOSAL_MODULE = "backend.copilot.tools.soul_proposal"


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


class _ConcurrentReadRedis(_FakeRedis):
    """Hold both GETs until they have read the same proposal."""

    def __init__(self):
        super().__init__()
        self.readers = 0
        self.both_read = asyncio.Event()

    async def get(self, key):
        value = await super().get(key)
        self.readers += 1
        if self.readers == 2:
            self.both_read.set()
        await asyncio.wait_for(self.both_read.wait(), timeout=5)
        return value


def _current_expert():
    return SimpleNamespace(
        identity="Old identity.",
        voice_preferences="Old voice.",
        boundaries="Old boundaries.",
    )


@contextmanager
def _patch_databases(mock_db):
    with (
        patch(f"{_MODULE}.experts_db", MagicMock(return_value=mock_db)),
        patch(f"{_PROPOSAL_MODULE}.experts_db", MagicMock(return_value=mock_db)),
    ):
        yield


def _mock_env(*, expert=None, applied=True, redis=None):
    mock_db = MagicMock()
    mock_db.get_expert = AsyncMock(return_value=expert)
    mock_db.update_soul_fields_if_current = AsyncMock(return_value=applied)
    patchers = [
        _patch_databases(mock_db),
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
    """A model-supplied confirm flag must be inert: the preview tool has no
    write path, so nothing it is called with can cause a write."""
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
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_applies_exact_previewed_proposal():
    mock_db, patchers = _mock_env(expert=_current_expert())
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
    mock_db.update_soul_fields_if_current.assert_awaited_once_with(
        _TEST_USER_ID,
        "exp-1",
        voice_preferences="New voice.",
        expected_voice_preferences="Old voice.",
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
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirmation_id_is_single_use():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        first = await _confirm(session, confirmation_id=preview.confirmation_id)
        second = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(first, ExpertSoulUpdatedResponse) and first.applied is True
    assert isinstance(second, ErrorResponse)
    mock_db.update_soul_fields_if_current.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_confirms_apply_exactly_once():
    redis = _ConcurrentReadRedis()
    mock_db, patchers = _mock_env(expert=_current_expert(), redis=redis)
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        results = await asyncio.gather(
            _confirm(session, confirmation_id=preview.confirmation_id),
            _confirm(session, confirmation_id=preview.confirmation_id),
        )

    assert sum(isinstance(result, ExpertSoulUpdatedResponse) for result in results) == 1
    assert sum(isinstance(result, ErrorResponse) for result in results) == 1
    mock_db.update_soul_fields_if_current.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_stale_before_state():
    """A Soul that changed between preview and confirm discards the proposal."""
    mock_db, patchers = _mock_env(expert=_current_expert(), applied=False)
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        resp = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ErrorResponse)
    assert "changed since" in resp.message
    mock_db.update_soul_fields_if_current.assert_awaited_once_with(
        _TEST_USER_ID,
        "exp-1",
        voice_preferences="New voice.",
        expected_voice_preferences="Old voice.",
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_unknown_id():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _confirm(session, confirmation_id="never-issued")
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_proposal_from_other_session():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session_a = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session_a, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        session_b = make_session(_TEST_USER_ID, expert_id="exp-1")
        rejected = await _confirm(session_b, confirmation_id=preview.confirmation_id)
        accepted = await _confirm(session_a, confirmation_id=preview.confirmation_id)
    assert isinstance(rejected, ErrorResponse)
    assert isinstance(accepted, ExpertSoulUpdatedResponse)
    mock_db.update_soul_fields_if_current.assert_awaited_once()


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
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_no_fields_rejected():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session)
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_noop_when_values_match_current():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session, voice_preferences="Old voice.")
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_missing_expert_errors():
    mock_db, patchers = _mock_env(expert=None)
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session, identity="x")
    assert isinstance(resp, ErrorResponse)
    mock_db.update_soul_fields_if_current.assert_not_called()


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


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_rejects_blank_identity_before_storing():
    """Invalid input must die at preview — never become a stored proposal
    that explodes after the user has already approved the diff."""
    fake_redis = _FakeRedis()
    mock_db, patchers = _mock_env(expert=_current_expert(), redis=fake_redis)
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session, identity="   ")
    assert isinstance(resp, ErrorResponse)
    assert "identity" in resp.message
    assert fake_redis.store == {}
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_rejects_overlong_field():
    fake_redis = _FakeRedis()
    mock_db, patchers = _mock_env(expert=_current_expert(), redis=fake_redis)
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _preview(session, voice_preferences="x" * 4_001)
    assert isinstance(resp, ErrorResponse)
    assert fake_redis.store == {}
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_preview_strips_whitespace_so_diff_matches_persisted():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="  New voice.  ")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        assert [(c.field, c.after) for c in preview.changes] == [
            ("voice_preferences", "New voice.")
        ]
        resp = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ExpertSoulUpdatedResponse) and resp.applied is True
    mock_db.update_soul_fields_if_current.assert_awaited_once_with(
        _TEST_USER_ID,
        "exp-1",
        voice_preferences="New voice.",
        expected_voice_preferences="Old voice.",
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_other_user():
    """The proposal.user_id binding is the core authz boundary: an id issued
    to one user must be useless to any other."""
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        resp = await ConfirmExpertSoulUpdateTool()._execute(
            user_id="some-other-user",
            session=session,
            confirmation_id=preview.confirmation_id,
        )
        accepted = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ErrorResponse)
    assert isinstance(accepted, ExpertSoulUpdatedResponse)
    mock_db.update_soul_fields_if_current.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_rejects_other_expert():
    mock_db, patchers = _mock_env(expert=_current_expert())
    with patchers[0], patchers[1]:
        session_a = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session_a, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        session_b = make_session(_TEST_USER_ID, expert_id="exp-2")
        resp = await _confirm(session_b, confirmation_id=preview.confirmation_id)
        accepted = await _confirm(session_a, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ErrorResponse)
    assert isinstance(accepted, ExpertSoulUpdatedResponse)
    mock_db.update_soul_fields_if_current.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_confirm_recovers_from_corrupt_stored_proposal():
    fake_redis = _FakeRedis()
    fake_redis.store[proposal_key("corrupt-id")] = "{not valid json"
    mock_db, patchers = _mock_env(expert=_current_expert(), redis=fake_redis)
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        resp = await _confirm(session, confirmation_id="corrupt-id")
    assert isinstance(resp, ErrorResponse)
    assert "fresh preview" in resp.message
    mock_db.update_soul_fields_if_current.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_apply_failure_returns_recoverable_error():
    """The id is already consumed when the write runs, so a surprise failure
    must come back as guidance to re-preview, not an uncaught exception."""
    mock_db, patchers = _mock_env(expert=_current_expert())
    mock_db.update_soul_fields_if_current.side_effect = RuntimeError(
        "db transport down"
    )
    with patchers[0], patchers[1]:
        session = make_session(_TEST_USER_ID, expert_id="exp-1")
        preview = await _preview(session, voice_preferences="New voice.")
        assert isinstance(preview, ExpertSoulUpdatedResponse)
        resp = await _confirm(session, confirmation_id=preview.confirmation_id)
    assert isinstance(resp, ErrorResponse)
    assert "re-preview" in resp.message
