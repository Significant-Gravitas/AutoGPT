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
retries just as long, so the dream store writes each pass's record to an
in-memory ``FakeDreamDb`` that tests can read the transitions back from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
import pytest_asyncio

from backend.data.dream_pass import (
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

    Keeps each pass's row the way the data module does: a ``None`` field
    leaves its column, ``phase_outputs`` and ``operations`` merge one field at
    a time, and a terminal row is final. ``writes`` holds every draft and
    update that landed, in order. ``fail`` makes every call raise, as an
    unreachable database would.
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
        self.rows[draft.id] = {**dict(draft), "phase_outputs": {}, "operations": {}}

    async def update_dream_pass(self, pass_id: str, update: DreamPassUpdate) -> bool:
        self._raise_if_down()
        row = self.rows.get(pass_id)
        if row is None or row["status"] not in OPEN_STATUSES:
            return False
        self.writes.append((pass_id, update))
        for field, value in dict(update).items():
            if value is None:
                continue
            if field in ("phase_outputs", "operations"):
                row[field].update(
                    {k: v for k, v in dict(value).items() if v is not None}
                )
            else:
                row[field] = value
        return True

    def seed(self, draft: DreamPassDraft) -> None:
        """A row as an earlier step (another process) would have left it."""
        self.rows[draft.id] = {**dict(draft), "phase_outputs": {}, "operations": {}}

    def statuses(self, pass_id: str) -> list[Any]:
        return [w.status for pid, w in self.writes if pid == pass_id and w.status]

    def phases(self, pass_id: str) -> list[Any]:
        """The steps the row went through, a step written twice in a row once."""
        steps = [w.phase for pid, w in self.writes if pid == pass_id and w.phase]
        return [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]]

    def record(self, pass_id: str) -> DreamPassRecord:
        """The row as ``get_dream_pass`` would read it back."""
        row = self.rows[pass_id]
        stamp = row.get("started_at") or datetime.now(timezone.utc)
        return DreamPassRecord.model_validate(
            {
                **{name: row.get(name) for name in DreamPassRecord.model_fields},
                "cancel_generation": 0,
                "created_at": stamp,
                "updated_at": stamp,
                "phase_outputs": DreamPhaseOutputs(**row["phase_outputs"]),
                "operations": DreamPassOperations(**row["operations"]),
            }
        )

    def _raise_if_down(self) -> None:
        if self.fail:
            raise ConnectionError("dream pass database unreachable")


@pytest.fixture(autouse=True)
def fake_dream_db(monkeypatch: pytest.MonkeyPatch) -> FakeDreamDb:
    """Give every dream test an in-memory pass record behind the store."""
    fake = FakeDreamDb()
    monkeypatch.setattr("backend.copilot.dream.store.dream_db", lambda: fake)
    return fake


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
