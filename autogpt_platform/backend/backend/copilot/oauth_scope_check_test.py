"""Tests for the post-OAuth scope check.

Covers the contract:

* a shortfall reaches the chat that asked for the connection, and nothing else
* a clean connect produces no chat noise and clears any stale warning
* two cards open for one provider are each judged against their own request
* the model-facing status is emitted regardless of the notice feature flag
* a failure anywhere is swallowed — the credential is already stored
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import oauth_scope_check
from backend.copilot.oauth_scope_check import (
    PendingConnect,
    _run_scope_check,
    record_pending_connect,
    schedule_scope_check,
    scope_status_lines,
    status_key,
)

_USER = "user-1"
_SESSION = "session-1"
_OTHER_SESSION = "session-2"


class _FakeRedis:
    """Async-Redis fake covering only the calls this module makes."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.strings: dict[str, str] = {}

    async def rpush(self, key: str, *values: str) -> int:
        self.lists.setdefault(key, []).extend(values)
        return len(self.lists[key])

    async def ltrim(self, key: str, start: int, stop: int) -> None:
        values = self.lists.get(key, [])
        self.lists[key] = values[start:] if stop == -1 else values[start : stop + 1]

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        values = self.lists.get(key, [])
        return list(values[start:] if stop == -1 else values[start : stop + 1])

    async def expire(self, key: str, seconds: int) -> int:
        return 1

    async def delete(self, *keys: str) -> int:
        for key in keys:
            self.lists.pop(key, None)
            self.hashes.pop(key, None)
            self.strings.pop(key, None)
        return len(keys)

    async def hset(self, key: str, field: str, value: str) -> int:
        self.hashes.setdefault(key, {})[field] = value
        return 1

    async def hdel(self, key: str, *fields: str) -> int:
        entries = self.hashes.get(key, {})
        return sum(1 for field in fields if entries.pop(field, None) is not None)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def set(self, key: str, value: str, *, nx: bool = False, ex: Any = None):
        if nx and key in self.strings:
            return None
        self.strings[key] = value
        return True

    def pipeline(self, transaction: bool = True) -> "_FakePipeline":
        return _FakePipeline(self)


class _FakePipeline:
    def __init__(self, parent: _FakeRedis) -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str):
        def _record(*args: Any) -> "_FakePipeline":
            self._ops.append((name, args))
            return self

        return _record

    async def execute(self) -> list[Any]:
        return [await getattr(self._parent, name)(*args) for name, args in self._ops]

    async def __aenter__(self) -> "_FakePipeline":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


def _session(user_id: str = _USER) -> MagicMock:
    info = MagicMock()
    info.user_id = user_id
    return info


def _patched(
    redis: _FakeRedis,
    *,
    enqueue: AsyncMock,
    flag_enabled: bool = True,
    session: MagicMock | None = _session(),
):
    return (
        patch.object(
            oauth_scope_check, "get_redis_async", new=AsyncMock(return_value=redis)
        ),
        patch.object(
            oauth_scope_check,
            "is_feature_enabled",
            new=AsyncMock(return_value=flag_enabled),
        ),
        patch(
            "backend.copilot.model.get_chat_session_metadata",
            new=AsyncMock(return_value=session),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
            new=enqueue,
        ),
    )


async def _seed(redis: _FakeRedis, session_id: str, scopes: list[str]) -> None:
    with patch.object(
        oauth_scope_check, "get_redis_async", new=AsyncMock(return_value=redis)
    ):
        await record_pending_connect(
            user_id=_USER,
            provider="github",
            session_id=session_id,
            requested_scopes=scopes,
        )


# ── Shortfall reaches the originating chat ──────────────────────────


@pytest.mark.asyncio
async def test_zero_scope_grant_notifies_the_originating_session():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_awaited_once()
    kwargs = enqueue.await_args.kwargs
    assert kwargs["session_id"] == _SESSION
    assert kwargs["user_id"] == _USER
    assert kwargs["timeout"] == 0
    message = kwargs["message"]
    # System-framed: the pending buffer has no author field.
    assert message.startswith("[System notice")
    assert 'provider="github"' in message
    assert "repo" in message
    assert "`alice`" in message
    # Provider-specific remediation, not the generic one.
    assert "Authorized OAuth Apps" in message


@pytest.mark.asyncio
async def test_partial_grant_names_only_the_missing_scope():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo", "workflow"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo"],
            provider_reports_scopes=True,
            username="alice",
        )

    assert 'missing="workflow"' in enqueue.await_args.kwargs["message"]


# ── Silence on success ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_grant_produces_no_chat_noise():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    enqueue = AsyncMock()

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo", "workflow"],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_reporting_provider_does_not_false_alarm():
    """Notion hardcodes ``scopes=[]``; that is silence, not refusal."""
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["read_content"])
    enqueue = AsyncMock()

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="notion",
            granted_scopes=[],
            provider_reports_scopes=False,
            username="alice",
        )

        enqueue.assert_not_awaited()
        assert await scope_status_lines(_USER, _SESSION) == []


@pytest.mark.asyncio
async def test_no_pending_connect_means_no_notice():
    """A connect made from the Integrations panel was never attributed to a
    chat; there is nobody to tell."""
    redis = _FakeRedis()
    enqueue = AsyncMock()

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_not_awaited()


# ── Two cards open for the same provider ────────────────────────────


@pytest.mark.asyncio
async def test_each_open_card_is_judged_against_its_own_request():
    """One chat asked for `repo`, another for `repo`+`workflow`. A grant of
    `repo` satisfies the first and shortchanges the second — and only the
    second is told."""
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    await _seed(redis, _OTHER_SESSION, ["repo", "workflow"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo"],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_awaited_once()
    assert enqueue.await_args.kwargs["session_id"] == _OTHER_SESSION


@pytest.mark.asyncio
async def test_two_cards_in_one_session_are_judged_against_their_union():
    """The model re-rendered a widened card into the *same* chat, so that chat
    has two pending records. The grant satisfies the first card and
    shortchanges the second — and the second is the one the user clicked.
    Judging only the older, narrower record would call this a clean connect,
    erase the warning, and never mention `workflow` again."""
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    await _seed(redis, _SESSION, ["repo", "workflow"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo"],
            provider_reports_scopes=True,
            username="alice",
        )
        lines = await scope_status_lines(_USER, _SESSION)

    enqueue.assert_awaited_once()
    assert enqueue.await_args.kwargs["session_id"] == _SESSION
    assert 'missing="workflow"' in enqueue.await_args.kwargs["message"]
    assert len(lines) == 1
    assert "workflow" in lines[0]


@pytest.mark.asyncio
async def test_pending_records_are_drained_so_a_replay_stays_quiet():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_list_is_bounded():
    redis = _FakeRedis()
    for index in range(oauth_scope_check._MAX_PENDING_PER_PROVIDER + 3):
        await _seed(redis, f"session-{index}", ["repo"])

    key = oauth_scope_check._pending_key(_USER, "github")
    assert len(redis.lists[key]) == oauth_scope_check._MAX_PENDING_PER_PROVIDER
    newest = PendingConnect.model_validate_json(redis.lists[key][-1])
    assert newest.session_id == "session-7"


# ── Model-facing status ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_status_is_written_even_when_the_notice_flag_is_off():
    """The flag gates interrupting the user, not the model's awareness — the
    model must still know not to re-fire connect_integration."""
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo", "workflow"])
    enqueue = AsyncMock()

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue, flag_enabled=False)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo"],
            provider_reports_scopes=True,
            username="alice",
        )

        enqueue.assert_not_awaited()
        lines = await scope_status_lines(_USER, _SESSION)

    assert len(lines) == 1
    assert "credential_scope_shortfall: github" in lines[0]
    assert "workflow" in lines[0]
    assert "Do not silently retry connect_integration" in lines[0]
    assert "Authorized OAuth Apps" in lines[0]


@pytest.mark.asyncio
async def test_a_clean_reconnect_clears_a_stale_shortfall():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo", "workflow"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo"],
            provider_reports_scopes=True,
            username="alice",
        )
        assert len(await scope_status_lines(_USER, _SESSION)) == 1

        await _seed(redis, _SESSION, ["repo", "workflow"])
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo", "workflow"],
            provider_reports_scopes=True,
            username="alice",
        )
        assert await scope_status_lines(_USER, _SESSION) == []


@pytest.mark.asyncio
async def test_status_lines_are_scoped_to_one_session():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )
        assert await scope_status_lines(_USER, _OTHER_SESSION) == []
        assert await scope_status_lines("other-user", _SESSION) == []


def test_status_key_is_namespaced_by_user():
    """Two users cannot read each other's shortfalls even on a shared
    session id."""
    assert status_key("u1", "s") != status_key("u2", "s")


def test_pending_key_is_namespaced_by_user():
    pending_key = oauth_scope_check._pending_key
    assert pending_key("u1", "github") != pending_key("u2", "github")


@pytest.mark.asyncio
async def test_unreadable_status_entry_is_skipped_not_raised():
    redis = _FakeRedis()
    redis.hashes[status_key(_USER, _SESSION)] = {"github": "not json"}

    with patch.object(
        oauth_scope_check, "get_redis_async", new=AsyncMock(return_value=redis)
    ):
        assert await scope_status_lines(_USER, _SESSION) == []


@pytest.mark.asyncio
async def test_status_read_failure_degrades_to_no_lines():
    """This runs on the per-turn hot path; a Redis blip must not break turns."""
    with patch.object(
        oauth_scope_check,
        "get_redis_async",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        assert await scope_status_lines(_USER, _SESSION) == []


# ── Ownership and failure handling ──────────────────────────────────


@pytest.mark.asyncio
async def test_session_owned_by_another_user_is_never_posted_into():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    enqueue = AsyncMock()

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue, session=None)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_failure_is_swallowed():
    """The OAuth callback has already stored the credential; nothing here is
    worth failing that for."""
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    enqueue = AsyncMock(side_effect=RuntimeError("rabbit down"))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_failure_is_swallowed():
    with patch.object(
        oauth_scope_check,
        "get_redis_async",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )


@pytest.mark.asyncio
async def test_recording_a_pending_connect_never_raises():
    """A Redis hiccup must not stop the setup card from rendering."""
    with patch.object(
        oauth_scope_check,
        "get_redis_async",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        await record_pending_connect(
            user_id=_USER,
            provider="github",
            session_id=_SESSION,
            requested_scopes=["repo"],
        )


@pytest.mark.asyncio
async def test_unparseable_pending_record_is_discarded():
    redis = _FakeRedis()
    redis.lists[oauth_scope_check._pending_key(_USER, "github")] = ["garbage"]
    enqueue = AsyncMock()

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )

    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_scope_check_runs_detached():
    run = AsyncMock()
    with patch.object(oauth_scope_check, "_run_scope_check", new=run):
        schedule_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )
        await asyncio.sleep(0)

    run.assert_awaited_once()


@pytest.mark.asyncio
async def test_notice_flag_key():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo"])
    flag = AsyncMock(return_value=False)

    with (
        patch.object(
            oauth_scope_check, "get_redis_async", new=AsyncMock(return_value=redis)
        ),
        patch.object(oauth_scope_check, "is_feature_enabled", new=flag),
    ):
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=[],
            provider_reports_scopes=True,
            username="alice",
        )

    assert flag.await_args.args[0].value == "copilot-oauth-scope-check"
    assert flag.await_args.args[1] == _USER


@pytest.mark.asyncio
async def test_status_payload_records_the_full_diff():
    redis = _FakeRedis()
    await _seed(redis, _SESSION, ["repo", "workflow"])
    enqueue = AsyncMock(return_value=("running", MagicMock()))

    p1, p2, p3, p4 = _patched(redis, enqueue=enqueue)
    with p1, p2, p3, p4:
        await _run_scope_check(
            user_id=_USER,
            provider="github",
            granted_scopes=["repo"],
            provider_reports_scopes=True,
            username="alice",
        )

    payload = json.loads(redis.hashes[status_key(_USER, _SESSION)]["github"])
    assert payload == {
        "coverage": "partial",
        "requested": ["repo", "workflow"],
        "granted": ["repo"],
        "missing": ["workflow"],
    }


def test_generic_remediation_for_unknown_provider():
    text = oauth_scope_check.remediation_for("some-provider")
    assert "consent screen" in text
    assert "GitHub" not in text
