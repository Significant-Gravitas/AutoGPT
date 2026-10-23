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
import gc
from collections.abc import Iterator
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from backend.copilot import usage_activation_fixtures
from backend.copilot.trial_cost_context import TrialCostContext
from backend.data.dream_pass_models import (
    DREAM_TEXT_FIELDS,
    ERASED_TEXT,
    OPEN_STATUSES,
    UNKNOWN_READS,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

from .pass_errors import PASS_ERROR_STAGES, UNRECORDED, is_pass_error


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():
    """No-op server stub — dream tests don't need the full backend."""
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():
    """No-op graph cleanup stub."""
    yield


@pytest.fixture(autouse=True)
def no_gc_during_test() -> Iterator[None]:
    """These tests time the loop against budgets of tenths of a second, and a
    full collection over the suite's heap stalls it for over a second on CI."""
    enabled = gc.isenabled()
    gc.disable()
    yield
    if enabled:
        gc.enable()


class FakeAsyncRedis:
    """In-memory stand-in for the ``decode_responses=True`` async client.

    Covers the surface the dream code uses: ``get``/``set`` (with ``nx``,
    ``xx`` and ``ex``), ``delete``, ``expire``, ``ttl``, ``exists``, ``incr``,
    the hash commands, the set commands the reaper's sweep of pending dream
    records uses (``graphiti/provenance_pending.py``), and ``eval`` for the
    two single-key lock scripts in ``locks.py`` (compare-and-delete,
    compare-and-extend). TTLs are recorded, not enforced.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.ttls: dict[str, int] = {}

    async def sadd(self, name: str, *values: Any) -> int:
        bucket = self.sets.setdefault(name, set())
        added = {self._s(value) for value in values} - bucket
        bucket |= added
        return len(added)

    async def srem(self, name: str, *values: Any) -> int:
        bucket = self.sets.get(name, set())
        removed = {self._s(value) for value in values} & bucket
        bucket -= removed
        return len(removed)

    async def srandmember(self, name: str, number: int) -> list[str]:
        return sorted(self.sets.get(name, set()))[:number]

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
    a ``None`` field leaves its column, status, phase and the lease's
    expiry only move forward,
    a batch id lands only while the row has not moved past the update's
    phase, ``phase_outputs`` and ``operations`` merge one field at a time, a
    stop bumps the cancel generation, ``clear`` empties its columns, and a
    terminal row, another user's row (for an update naming the owner) or a
    row written after the update's instant is not written; an update marked
    ``closed_row`` writes a terminal row only. The read set only grows, and an
    error that is not one of the fixed codes is recorded as ``UNRECORDED``.
    An update that closes a row a hard forget marked (``erased_at``) blanks
    its text and error, as ``erase_dream_passes`` does to a closed one.
    ``writes`` holds every draft and update that was written, in order, as
    sent. ``fail`` makes every call raise, as an unreachable database would.
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
        self._insert(draft)

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
        if update.read_uuids is not None:
            row["read_uuids"] = sorted({*row["read_uuids"], *update.read_uuids})
        for field, value in dict(update).items():
            if value is None or field in _NOT_WRITTEN_AS_IS:
                continue
            if field == "error":
                row["error"] = recorded_error(value)
            elif field in ("phase_outputs", "operations"):
                row[field].update(
                    {k: v for k, v in dict(value).items() if v is not None}
                )
            elif field == "lease_expires_at" and row.get(field) is not None:
                row[field] = max(row[field], value)
            else:
                row[field] = value
        row.update({column: None for column in update.clear})
        row["cancel_generation"] += int(update.bump_cancel_generation)
        row["updated_at"] = datetime.now(timezone.utc)
        closes = update.closed_row or (
            update.status is not None and update.status not in OPEN_STATUSES
        )
        if row.get("erased_at") is not None and closes:
            _blank_row(row)
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

    async def erase_dream_passes(
        self, user_id: str, scope_key: str, *, reached: list[str], named: list[str]
    ) -> list[str]:
        """Mark the user's passes of the scope a hard forget reached (one
        that read a reached uuid or may have read anything, open, closed with
        a cleanup pending, named, or marked before), blank those marked now
        and closed, and keep every
        error of the scope only as a fixed code; the ids marked, now or
        before. ``data/dream_pass.ERASE_SQL``."""
        self._raise_if_down()
        marked: list[str] = []
        for pass_id, row in self.rows.items():
            if (row["user_id"], row["scope_key"]) != (user_id, scope_key):
                continue
            row["error"] = blank_error(row.get("error"))
            if not (
                row.get("erased_at") is not None
                or row["status"] in OPEN_STATUSES
                or row.get("cleanup_pending_at") is not None
                or pass_id in named
                or set(row["read_uuids"]) & set(reached)
                or UNKNOWN_READS in row["read_uuids"]
            ):
                continue
            marked.append(pass_id)
            if row.get("erased_at") is None:
                row["erased_at"] = datetime.now(timezone.utc)
                if row["status"] not in OPEN_STATUSES:
                    _blank_row(row)
        return marked

    async def list_expired_dream_passes(
        self,
        expired_before: datetime,
        limit: int = 100,
        after_lapse: datetime | None = None,
        after_id: str | None = None,
    ) -> list[DreamPassRecord]:
        """Open rows whose lease lapsed before *expired_before*, in lapse
        order (the lapse, then the id), after *after_lapse* and *after_id*
        when given, at most *limit*."""
        self._raise_if_down()
        lapsed = sorted(
            (row["lease_expires_at"], pass_id)
            for pass_id, row in self.rows.items()
            if row["status"] in OPEN_STATUSES
            and row.get("lease_expires_at") is not None
            and row["lease_expires_at"] < expired_before
        )
        if after_lapse is not None and after_id is not None:
            lapsed = [key for key in lapsed if key > (after_lapse, after_id)]
        return [self.record(pass_id) for _, pass_id in lapsed[:limit]]

    async def list_dream_pass_cleanups(
        self,
        due_before: datetime,
        limit: int = 100,
        marked_through: datetime | None = None,
        after_mark: datetime | None = None,
        after_id: str | None = None,
    ) -> list[DreamPassRecord]:
        """Closed rows marked for a cleanup that is due (marked, or holding
        no lease or one that lapsed, before *due_before*), in mark order (the
        mark, then the id), marked by *marked_through* and after *after_mark*
        and *after_id* when given."""
        self._raise_if_down()
        pending = sorted(
            (row["cleanup_pending_at"], pass_id)
            for pass_id, row in self.rows.items()
            if row["status"] not in OPEN_STATUSES
            and row.get("cleanup_pending_at") is not None
            and _due(row, due_before)
            and (marked_through is None or row["cleanup_pending_at"] <= marked_through)
        )
        if after_mark is not None and after_id is not None:
            pending = [key for key in pending if key > (after_mark, after_id)]
        return [self.record(pass_id) for _, pass_id in pending[:limit]]

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
        *columns* set the rest of it (a lease, when it was last written).

        Unless the draft or *columns* say otherwise, the row carries the
        usage attribution its pass would have started under in these tests
        (``usage_snapshot``: the owner, ready, untrialed), as ``start_pass``
        records it; ``cost_context=None`` seeds a row that carries none."""
        if draft.cost_context is None:
            columns = {
                "cost_context": TrialCostContext(
                    user_id=draft.user_id, trial_id=None
                ).model_dump(mode="json"),
                **columns,
            }
        self._insert(draft, **columns)

    def _insert(self, draft: DreamPassDraft, **columns: Any) -> None:
        now = datetime.now(timezone.utc)
        self.rows[draft.id] = {
            **dict(draft),
            "phase_outputs": {},
            "operations": {},
            "read_uuids": [],
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
        "read_uuids",
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


def _blank_row(row: dict[str, Any]) -> None:
    """Blank the dream's text in *row*'s JSON columns, and its error unless
    it is a fixed code."""
    row["phase_outputs"], row["operations"] = blank_text(_json_columns(row))
    row["error"] = blank_error(row.get("error"))


def _json_columns(row: dict[str, Any]) -> list[dict[str, Any]]:
    """*row*'s phase outputs and operations as the database keeps them."""
    return [
        DreamPhaseOutputs(**row["phase_outputs"]).model_dump(mode="json"),
        DreamPassOperations(**row["operations"]).model_dump(mode="json"),
    ]


def blank_text(value: Any) -> Any:
    """*value* with every string under one of ``DREAM_TEXT_FIELDS`` reading
    ``ERASED_TEXT``, as the migration's ``dream_text_blanked`` does."""
    if isinstance(value, dict):
        return {
            key: (
                ERASED_TEXT
                if key in DREAM_TEXT_FIELDS and isinstance(item, str)
                else blank_text(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [blank_text(item) for item in value]
    return value


def blank_error(error: str | None) -> str | None:
    """*error* when it is one of the fixed codes, else ``ERASED_TEXT`` after
    its stage when that is one of the fixed stages, as the migration's
    ``dream_error_blanked`` does."""
    if error is None or is_pass_error(error):
        return error
    stage, separator, _ = error.partition(": ")
    if separator and stage in PASS_ERROR_STAGES:
        return f"{stage}: {ERASED_TEXT}"
    return ERASED_TEXT


def recorded_error(error: str) -> str:
    """*error* as the store records it: a fixed code, else ``UNRECORDED``."""
    return error if is_pass_error(error) else UNRECORDED


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


# A pass captures its usage attribution before its first provider call
# (#15208's ``attributed_usage``); every dream test gets the ready, untrialed
# usage state unless it sets its own.
usage_snapshot = usage_activation_fixtures.usage_snapshot


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
