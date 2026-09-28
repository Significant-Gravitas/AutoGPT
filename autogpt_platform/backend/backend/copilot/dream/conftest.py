"""Local conftest for copilot/dream tests.

Overrides the session-scoped ``server`` and ``graph_cleanup`` autouse fixtures
from backend/conftest.py so the dream unit tests do not boot the full
SpinTestServer (Postgres, RabbitMQ and every service). Mirrors
copilot/tools/conftest.py, and applies to the ``webcheck/`` tests too, since
pytest inherits conftests by directory.

Without the server there is no Redis either, and ``get_redis_async`` retries
for close to an hour before giving up, so every test in this directory gets an
in-memory stand-in by default. Tests that need their own fake keep patching
``backend.data.redis_client.get_redis_async`` as before; a later patch wins.

No Postgres either, and the DatabaseManager RPC client behind ``dream_db()``
would stall a write until the store's deadline, so the dream store writes
each pass's record to an in-memory ``FakeDreamDb`` that tests can read the
transitions back from. The master flag the guard reads answers on. And no
test reaches Anthropic from a pass's cleanup: there is no key to reach it
with, and a batch's status reads ended, unless a test sets its own.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from backend.data.dream_pass_models import (
    OPEN_STATUSES,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    """No-op server stub — dream tests don't need the full backend."""
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    """No-op graph cleanup stub."""
    yield


class FakeAsyncRedis:
    """In-memory stand-in for the ``decode_responses=True`` async client.

    Covers the surface the dream code uses: ``get``/``set`` (with ``nx``,
    ``xx`` and ``ex``), ``delete``, ``expire``, ``ttl``, ``exists``, ``incr``,
    the hash commands, and ``eval`` for the two single-key lock scripts in
    ``locks.py`` (compare-and-delete, compare-and-extend). TTLs are recorded,
    not enforced.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}

    @staticmethod
    def _s(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        nx: bool = False,
        xx: bool = False,
        ex: int | None = None,
        px: int | None = None,
        **_: Any,
    ) -> bool | None:
        if nx and key in self.store:
            return None
        if xx and key not in self.store:
            return None
        self.store[key] = self._s(value)
        if ex is not None:
            self.ttls[key] = int(ex)
        elif px is not None:
            self.ttls[key] = int(px) // 1000
        else:
            # A plain SET drops any expiry the key had, as Redis does.
            self.ttls.pop(key, None)
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self.store.pop(key, None) is not None:
                removed += 1
            if self.hashes.pop(key, None) is not None:
                removed += 1
            self.ttls.pop(key, None)
        return removed

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        if key not in self.store and key not in self.hashes:
            return False
        self.ttls[key] = int(ttl_seconds)
        return True

    async def ttl(self, key: str) -> int:
        if key not in self.store and key not in self.hashes:
            return -2
        return self.ttls.get(key, -1)

    async def exists(self, *keys: str) -> int:
        return sum(1 for key in keys if key in self.store or key in self.hashes)

    async def incr(self, key: str) -> int:
        value = int(self.store.get(key, "0")) + 1
        self.store[key] = str(value)
        return value

    async def hset(
        self,
        name: str,
        key: str | None = None,
        value: Any = None,
        mapping: dict[str, Any] | None = None,
    ) -> int:
        bucket = self.hashes.setdefault(name, {})
        added = 0
        items = dict(mapping or {})
        if key is not None:
            items[key] = value
        for field, item in items.items():
            if field not in bucket:
                added += 1
            bucket[field] = self._s(item)
        return added

    async def hget(self, name: str, key: str) -> str | None:
        return self.hashes.get(name, {}).get(key)

    async def hgetall(self, name: str) -> dict[str, str]:
        return dict(self.hashes.get(name, {}))

    async def hdel(self, name: str, *keys: str) -> int:
        bucket = self.hashes.get(name, {})
        removed = sum(1 for key in keys if bucket.pop(key, None) is not None)
        if not bucket:
            # Redis removes a hash once its last field is gone.
            self.hashes.pop(name, None)
            self.ttls.pop(name, None)
        return removed

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        keys = [self._s(a) for a in args[:numkeys]]
        argv = [self._s(a) for a in args[numkeys:]]
        if numkeys != 1 or not argv:
            raise NotImplementedError(
                "FakeAsyncRedis.eval only models the lock scripts"
            )
        key, token = keys[0], argv[0]
        if self.store.get(key) != token:
            return 0
        if '"del"' in script:
            del self.store[key]
            self.ttls.pop(key, None)
            return 1
        if '"expire"' in script:
            self.ttls[key] = int(argv[1])
            return 1
        raise NotImplementedError(f"FakeAsyncRedis.eval: unknown script {script!r}")

    async def aclose(self) -> None:
        return None


class FakeDreamDb:
    """In-memory stand-in for ``backend.data.dream_pass`` behind ``dream_db()``.

    Applies an update by the rules of ``backend/data/dream_pass_update.py``:
    a ``None`` field leaves its column, status and phase only move forward,
    a batch id lands only while the row has not moved past the update's
    phase, ``phase_outputs`` and ``operations`` merge one field at a time, a
    stop bumps the cancel generation, ``clear`` empties its columns, and a
    terminal row, another user's row (for an update naming the owner) or a
    row written after the update's instant is not written; an update marked
    ``closed_row`` writes a terminal row only. ``writes`` holds
    every draft and update that was written, in order, as sent. ``fail``
    makes every call raise, as an unreachable database would.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.writes: list[tuple[str, DreamPassDraft | DreamPassUpdate]] = []
        self.fail = False

    async def create_dream_pass(self, draft: DreamPassDraft) -> None:
        self._raise_if_down()
        if draft.id in self.rows:
            raise ValueError(f"duplicate dream pass {draft.id}")
        self.writes.append((draft.id, draft))
        self.seed(draft)

    async def update_dream_pass(self, pass_id: str, update: DreamPassUpdate) -> bool:
        self._raise_if_down()
        row = self.rows.get(pass_id)
        if row is None or not _writable(row, update):
            return False
        self.writes.append((pass_id, update))
        if update.provider_batch_id is not None and _order(row["phase"]) <= _order(
            update.phase
        ):
            row["provider_batch_id"] = update.provider_batch_id
        if update.status is not None and _order(update.status) > _order(row["status"]):
            row["status"] = update.status
        if update.phase is not None and _order(update.phase) > _order(row["phase"]):
            row["phase"] = update.phase
        for field, value in dict(update).items():
            if value is None or field in _NOT_WRITTEN_AS_IS:
                continue
            if field in ("phase_outputs", "operations"):
                row[field].update(
                    {k: v for k, v in dict(value).items() if v is not None}
                )
            else:
                row[field] = value
        row.update({column: None for column in update.clear})
        row["cancel_generation"] += int(update.bump_cancel_generation)
        row["updated_at"] = datetime.now(timezone.utc)
        return True

    async def get_dream_pass(self, pass_id: str) -> DreamPassRecord | None:
        self._raise_if_down()
        return self.record(pass_id) if pass_id in self.rows else None

    async def get_dream_pass_for_user(
        self, pass_id: str, user_id: str
    ) -> DreamPassRecord | None:
        row = await self.get_dream_pass(pass_id)
        return row if row is not None and row.user_id == user_id else None

    async def list_open_dream_passes(
        self, scope_key: str, limit: int | None = None
    ) -> list[DreamPassRecord]:
        """Open rows of the scope, the last inserted first, at most *limit*."""
        self._raise_if_down()
        return [
            self.record(pass_id)
            for pass_id, row in reversed(self.rows.items())
            if row["scope_key"] == scope_key and row["status"] in OPEN_STATUSES
        ][:limit]

    async def list_expired_dream_passes(
        self, expired_before: datetime, limit: int = 100
    ) -> list[DreamPassRecord]:
        """Open rows whose lease lapsed before *expired_before*, oldest
        lapse first, at most *limit*."""
        self._raise_if_down()
        lapsed = [
            (row["lease_expires_at"], pass_id)
            for pass_id, row in self.rows.items()
            if row["status"] in OPEN_STATUSES
            and row.get("lease_expires_at") is not None
            and row["lease_expires_at"] < expired_before
        ]
        return [self.record(pass_id) for _, pass_id in sorted(lapsed)[:limit]]

    async def list_dream_pass_cleanups(
        self, due_before: datetime, limit: int = 100
    ) -> list[DreamPassRecord]:
        """Closed rows marked for a cleanup that is due (marked, or holding
        no lease or one that lapsed, before *due_before*), longest pending
        first."""
        self._raise_if_down()
        pending = [
            (row["cleanup_pending_at"], pass_id)
            for pass_id, row in self.rows.items()
            if row["status"] not in OPEN_STATUSES
            and row.get("cleanup_pending_at") is not None
            and _due(row, due_before)
        ]
        return [self.record(pass_id) for _, pass_id in sorted(pending)[:limit]]

    async def delete_old_dream_passes(
        self, created_before: datetime, limit: int = 1000
    ) -> int:
        """Delete at most *limit* closed rows created before *created_before*,
        none whose cleanup is pending."""
        self._raise_if_down()
        old = [
            pass_id
            for pass_id, row in self.rows.items()
            if row["status"] not in OPEN_STATUSES
            and row.get("cleanup_pending_at") is None
            and row["created_at"] < created_before
        ][:limit]
        for pass_id in old:
            del self.rows[pass_id]
        return len(old)

    def seed(self, draft: DreamPassDraft, **columns: Any) -> None:
        """A row as an earlier step (another process) would have left it;
        *columns* set the rest of it (a lease, when it was last written)."""
        now = datetime.now(timezone.utc)
        self.rows[draft.id] = {
            **dict(draft),
            "phase_outputs": {},
            "operations": {},
            "cancel_generation": 0,
            "created_at": now,
            "updated_at": now,
            **columns,
        }

    def statuses(self, pass_id: str) -> list[Any]:
        return [w.status for pid, w in self.writes if pid == pass_id and w.status]

    def phases(self, pass_id: str) -> list[Any]:
        """The steps the row went through, a step written twice in a row once."""
        steps = [w.phase for pid, w in self.writes if pid == pass_id and w.phase]
        return [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]]

    def record(self, pass_id: str) -> DreamPassRecord:
        """The row as ``get_dream_pass`` would read it back."""
        row = self.rows[pass_id]
        return DreamPassRecord.model_validate(
            {
                **{name: row.get(name) for name in DreamPassRecord.model_fields},
                "phase_outputs": DreamPhaseOutputs(**row["phase_outputs"]),
                "operations": DreamPassOperations(**row["operations"]),
            }
        )

    def _raise_if_down(self) -> None:
        if self.fail:
            raise ConnectionError("dream pass database unreachable")


# Update fields that are not a column written as sent: the forward-only
# columns, the generation bump, the conditions and the columns to empty.
_NOT_WRITTEN_AS_IS = frozenset(
    {
        "status",
        "phase",
        "provider_batch_id",
        "bump_cancel_generation",
        "owner_user_id",
        "not_updated_since",
        "clear",
        "closed_row",
    }
)


def _writable(row: dict[str, Any], update: DreamPassUpdate) -> bool:
    """Whether *update* may write *row*: open (closed, for a ``closed_row``
    update), the owner's when it names one, not written after its instant
    when it names one."""
    return (
        (row["status"] in OPEN_STATUSES) is not update.closed_row
        and update.owner_user_id in (None, row["user_id"])
        and (
            update.not_updated_since is None
            or row["updated_at"] <= update.not_updated_since
        )
    )


def _due(row: dict[str, Any], due_before: datetime) -> bool:
    """Whether a marked row's cleanup is due, as the cleanup scan decides."""
    lease = row.get("lease_expires_at")
    return row["cleanup_pending_at"] <= due_before or (
        lease is None or lease <= due_before
    )


def _order(value: Enum | None) -> int:
    """Where *value* sits in its enum's declared order, as Postgres compares
    enums."""
    assert value is not None
    return list(type(value)).index(value)


@pytest.fixture(autouse=True)
def fake_dream_db(monkeypatch: pytest.MonkeyPatch) -> FakeDreamDb:
    """Give every dream test an in-memory pass record behind the store."""
    fake = FakeDreamDb()
    monkeypatch.setattr("backend.copilot.dream.store.dream_db", lambda: fake)
    return fake


class StalledDreamDb:
    """A DatabaseManager that never answers: every record write and read
    hangs until the store's deadline gives up on it and cancels it."""

    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0

    async def create_dream_pass(self, draft: DreamPassDraft) -> None:
        await self._hang()

    async def update_dream_pass(self, pass_id: str, update: DreamPassUpdate) -> bool:
        await self._hang()
        return True

    async def get_dream_pass(self, pass_id: str) -> DreamPassRecord | None:
        await self._hang()
        return None

    async def list_open_dream_passes(
        self, scope_key: str, limit: int | None = None
    ) -> list[DreamPassRecord]:
        await self._hang()
        return []

    async def _hang(self) -> None:
        self.started += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@pytest.fixture
def stalled_dream_db(monkeypatch: pytest.MonkeyPatch) -> StalledDreamDb:
    """Put a DatabaseManager that never answers behind the store, with a
    short write deadline so a test sees it expire."""
    stalled = StalledDreamDb()
    monkeypatch.setattr("backend.copilot.dream.store.dream_db", lambda: stalled)
    monkeypatch.setattr(
        "backend.copilot.dream.store.RECORD_WRITE_TIMEOUT_SECONDS", 0.05
    )
    return stalled


@pytest.fixture(autouse=True)
def dream_pass_flag(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The master flag answers on, authoritatively, at every pass's guard;
    without this the read would reach the configured flag vendor. Tests of
    the flag set the mock's return value or side effect."""
    flag = AsyncMock(return_value=(True, True))
    monkeypatch.setattr("backend.copilot.dream.guard.evaluate_feature_flag", flag)
    return flag


@pytest.fixture(autouse=True)
def batch_status(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """No Anthropic key behind a pass's cleanup, and a batch status that reads
    ended: the cleanup's provider step never reaches the network. Tests of
    that step set their own key, cancel and status."""
    monkeypatch.setattr(
        "backend.copilot.dream.provider_batch.anthropic_api_key", lambda: None
    )
    status = AsyncMock(return_value="ended")
    monkeypatch.setattr("backend.copilot.dream.provider_batch.poll_batch", status)
    return status


@pytest.fixture(autouse=True)
def fake_dream_redis(monkeypatch: pytest.MonkeyPatch) -> FakeAsyncRedis:
    """Give every dream test an in-memory Redis unless it patches its own."""
    fake = FakeAsyncRedis()

    async def _get_redis_async() -> FakeAsyncRedis:
        return fake

    monkeypatch.setattr(
        "backend.data.redis_client.get_redis_async", _get_redis_async, raising=True
    )
    return fake


@pytest.fixture(autouse=True)
def stub_dream_session_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the dream summary session to the platform route.

    ``apply._create_dream_session`` resolves the owner's default chat route,
    which reads the user row through the database accessors. Without the
    test server those fall back to the DatabaseManager RPC client, whose
    connection retry runs for close to an hour. No dream test asserts on the
    route, so answer "platform, no credential" up front.
    """

    async def _platform_route(user_id: str) -> tuple[str, None]:
        return ("platform", None)

    monkeypatch.setattr(
        "backend.copilot.dream.apply.resolve_default_chat_route", _platform_route
    )
