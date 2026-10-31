"""The running-box limits, driven through the real create and connect paths
against a fake E2B listing and the real Redis that holds the reservations."""

import asyncio
import contextlib
import itertools
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from e2b import SandboxState

from backend.copilot.tools import e2b_sandbox
from backend.copilot.tools.e2b_sandbox import SandboxOwner, connect_owned
from backend.data.redis_client import get_redis_async
from backend.util import sandbox_limits
from backend.util.e2b_network import EgressOwner, connect_sandbox, create_sandbox
from backend.util.sandbox_limits import ENV_KEY, USER_KEY, SandboxLimitError, start_slot
from backend.util.sandbox_metadata import SERVICE, SERVICE_FIELD, deployment_env

_ENV = deployment_env()
_OTHER_ENV = "dev" if _ENV == "prod" else "prod"
_USER = "user-a"
_OTHER_USER = "user-b"


class _FakeE2B:
    """E2B as the limits see it: boxes with stamps and a state.

    The listing ignores the metadata filter, so a count that trusted the filter
    would take in every deployment's and every account's running boxes.
    """

    def __init__(self) -> None:
        self.boxes: dict[str, SimpleNamespace] = {}
        # Started, but not in E2B's listing yet.
        self.unlisted: set[str] = set()
        self.on_list: AsyncMock = AsyncMock()
        self._ids = itertools.count()

    def add(
        self,
        *,
        env: str = _ENV,
        user: str | None = _USER,
        state: SandboxState = SandboxState.RUNNING,
        metadata: dict[str, str] | None = None,
    ) -> str:
        sandbox_id = f"sb-{next(self._ids)}"
        stamps = {SERVICE_FIELD: SERVICE, ENV_KEY: env}
        if user:
            stamps[USER_KEY] = user
        self.boxes[sandbox_id] = SimpleNamespace(
            sandbox_id=sandbox_id, state=state, metadata=metadata or stamps
        )
        return sandbox_id

    def running(self) -> set[str]:
        return {i for i, box in self.boxes.items() if box.state == SandboxState.RUNNING}

    async def create(self, *, metadata: dict[str, str], **_) -> SimpleNamespace:
        sandbox_id = f"sb-{next(self._ids)}"
        self.boxes[sandbox_id] = SimpleNamespace(
            sandbox_id=sandbox_id, state=SandboxState.RUNNING, metadata=metadata
        )
        return SimpleNamespace(sandbox_id=sandbox_id)

    async def connect(self, sandbox_id: str, **_) -> SimpleNamespace:
        self.boxes[sandbox_id].state = SandboxState.RUNNING
        return SimpleNamespace(sandbox_id=sandbox_id)

    async def get_info(self, sandbox_id: str, **_) -> SimpleNamespace:
        return self.boxes[sandbox_id]

    def list(self, *, query, **_) -> "_Pages":
        return _Pages(self, query.state)


class _Pages:
    def __init__(self, e2b: _FakeE2B, states: list[SandboxState]) -> None:
        self._e2b = e2b
        self._states = states
        self.has_next = True

    async def next_items(self) -> list[SimpleNamespace]:
        # Read before ``on_list`` can hold the caller: what a slow reply returns.
        listed = [
            box
            for box in self._e2b.boxes.values()
            if box.state in self._states and box.sandbox_id not in self._e2b.unlisted
        ]
        await self._e2b.on_list()
        self.has_next = False
        return listed


@pytest.fixture
async def e2b(monkeypatch: pytest.MonkeyPatch):
    prefix = f"test:{uuid.uuid4().hex}:e2b:sandbox_starts:"
    monkeypatch.setattr(sandbox_limits, "_KEY_PREFIX", prefix)
    monkeypatch.setenv("E2B_EGRESS_PROXY_ADDRESS", "")
    fake = _FakeE2B()
    monkeypatch.setattr(sandbox_limits, "AsyncSandbox", fake)
    yield fake
    redis = await get_redis_async()
    await redis.delete(
        *(
            f"{prefix}{{{_ENV}}}{suffix}"
            for suffix in ("", f":{_USER}", f":{_OTHER_USER}")
        )
    )


def _limits(monkeypatch: pytest.MonkeyPatch, *, deployment: int = 0, account: int = 0):
    monkeypatch.setenv("E2B_DEPLOYMENT_SANDBOX_LIMIT", str(deployment))
    monkeypatch.setenv("E2B_ACCOUNT_SANDBOX_LIMIT", str(account))


def _owner(user: str | None = _USER) -> EgressOwner:
    return EgressOwner(kind="session", id=f"s-{uuid.uuid4().hex[:8]}", user_id=user)


async def test_an_account_at_its_cap_is_refused_and_another_account_is_not(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch
):
    _limits(monkeypatch, account=2)
    for _ in range(2):
        await create_sandbox(e2b, _owner(), api_key="k")

    with pytest.raises(SandboxLimitError, match="running on your account: 2 at once"):
        await create_sandbox(e2b, _owner(), api_key="k")
    await create_sandbox(e2b, _owner(_OTHER_USER), api_key="k")

    assert len(e2b.running()) == 3


async def test_the_deployment_ceiling_counts_only_this_deployments_boxes(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch
):
    _limits(monkeypatch, deployment=2)
    for _ in range(3):
        e2b.add(env=_OTHER_ENV)
    e2b.add(user=_OTHER_USER)

    await create_sandbox(e2b, _owner(), api_key="k")
    with pytest.raises(SandboxLimitError, match="Too many sandboxes are running right"):
        await create_sandbox(e2b, _owner(), api_key="k")


async def test_concurrent_starts_for_the_last_place_start_exactly_one(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch
):
    """Both starts are held at the listing until the other gets there too, so
    a count read from E2B alone would let both through."""
    _limits(monkeypatch, account=1)
    barrier = asyncio.Barrier(2)

    async def both_at_the_listing() -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(barrier.wait(), timeout=2)

    e2b.on_list.side_effect = both_at_the_listing

    results = await asyncio.gather(
        *(create_sandbox(e2b, _owner(), api_key="k") for _ in range(2)),
        return_exceptions=True,
    )

    assert len(e2b.running()) == 1
    assert [type(r) for r in results].count(SandboxLimitError) == 1


async def test_a_box_counts_before_e2b_lists_it_and_once_after(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch
):
    _limits(monkeypatch, account=1)
    e2b.create = _created_unlisted(e2b, e2b.create)

    await create_sandbox(e2b, _owner(), api_key="k")
    with pytest.raises(SandboxLimitError):
        await create_sandbox(e2b, _owner(), api_key="k")

    # Listed now, and still in its grace period: one box, not two.
    e2b.unlisted.clear()
    [started] = e2b.running()
    await connect_sandbox(e2b, started, _owner(), api_key="k")


async def test_a_connect_that_wakes_a_paused_box_takes_a_place_and_one_to_a_running_box_does_not(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch
):
    """The copilot reads the box's state before connecting: waking a paused box
    past the cap is refused, using one that is already running is not, even
    with the account over its cap."""
    _limits(monkeypatch, account=1)
    owner = SandboxOwner(kind="expert", id="exp-1")
    stamp = owner.creation_metadata(user_id=_USER)
    paused = e2b.add(state=SandboxState.PAUSED, metadata=stamp)
    awake = e2b.add(metadata=stamp)
    e2b.add()
    monkeypatch.setattr(e2b_sandbox, "AsyncSandbox", e2b)
    monkeypatch.setattr(e2b_sandbox, "_keep_grants", AsyncMock())

    with pytest.raises(SandboxLimitError):
        await connect_owned(paused, owner, "k", user_id=_USER)
    await connect_owned(awake, owner, "k", user_id=_USER)

    assert e2b.boxes[paused].state == SandboxState.PAUSED


async def test_a_block_on_the_users_own_e2b_key_is_not_counted(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch
):
    _limits(monkeypatch, account=1)
    monkeypatch.setenv("E2B_API_KEY", "platform-key")
    e2b.add()
    block = EgressOwner(kind="block", id=_USER, user_id=_USER)

    await create_sandbox(e2b, block, api_key="users-own-key")
    with pytest.raises(SandboxLimitError):
        await create_sandbox(e2b, block, api_key="platform-key")


@pytest.mark.parametrize(
    "deployment, refused", [(5, True), (0, False)], ids=["ceiling", "account-only"]
)
async def test_a_start_whose_count_fails_is_refused_only_under_a_ceiling(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch, deployment: int, refused: bool
):
    _limits(monkeypatch, deployment=deployment, account=5)
    e2b.on_list.side_effect = RuntimeError("E2B is down")

    with patch.object(e2b, "create", wraps=e2b.create) as create:
        if refused:
            with pytest.raises(SandboxLimitError, match="Couldn't check"):
                await create_sandbox(e2b, _owner(), api_key="k")
        else:
            await create_sandbox(e2b, _owner(), api_key="k")

    assert create.await_count == (0 if refused else 1)


async def test_a_start_that_fails_gives_its_place_back(
    e2b: _FakeE2B, monkeypatch: pytest.MonkeyPatch
):
    _limits(monkeypatch, account=1)

    with pytest.raises(RuntimeError):
        async with start_slot("k", _USER):
            raise RuntimeError("E2B refused the create")

    await create_sandbox(e2b, _owner(), api_key="k")


def _created_unlisted(e2b: _FakeE2B, create):
    async def _create(**kwargs):
        box = await create(**kwargs)
        e2b.unlisted.add(box.sandbox_id)
        return box

    return _create
