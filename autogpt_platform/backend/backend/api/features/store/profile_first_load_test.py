"""A new user's marketplace profile, read before account setup has made it.

Better Auth issues the session as soon as the identity exists, when the auth
hook has inserted only a bare ``User`` row. The Profile comes from account
setup (``POST /auth/user``), so the first page's ``GET /store/profile`` 404'd
whenever it beat setup, and for good if setup's best-effort insert failed.
The read now creates the default profile itself. These run on the real
database: the race is between real inserts on ``Profile.userId``'s unique index.
"""

import asyncio
import random
import uuid
from contextvars import ContextVar
from typing import Literal

import pytest
from prisma.actions import ProfileActions
from pydantic import BaseModel, ConfigDict

from backend.api.features.store import db as store_db
from backend.api.features.store import model as store_model
from backend.api.features.store.routes import get_profile
from backend.data import profile as profile_module
from backend.data.db import prisma
from backend.data.user import get_or_create_user_with_status
from backend.util.test import SpinTestServer

_STALL = 5.0
_SEEDS = 200

_Role = Literal["read", "setup", "save"]
_ROLES: tuple[_Role, ...] = ("read", "setup", "save")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_profile_read_while_setup_creates_it_gets_the_one_profile(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    user_id, email = await _signed_up_user()

    # Hold account setup just before it inserts the profile.
    entered = asyncio.Event()
    release = asyncio.Event()
    create = ProfileActions.create

    async def hold_the_first_insert(self, *args, **kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await create(self, *args, **kwargs)

    monkeypatch.setattr(ProfileActions, "create", hold_the_first_insert)
    setup = asyncio.create_task(get_or_create_user_with_status(_claims(user_id, email)))
    try:
        await _wait_for(entered, "account setup to start inserting the profile")
        profile = await asyncio.wait_for(get_profile(user_id=user_id), timeout=_STALL)
    finally:
        release.set()
        await asyncio.wait_for(setup, timeout=_STALL)

    assert profile.name == email.split("@")[0]
    rows = await prisma.profile.find_many(where={"userId": user_id})
    assert [row.username for row in rows] == [profile.username]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_profile_read_heals_an_account_whose_setup_could_not_create_it(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    user_id, email = await _signed_up_user()

    async def fail(self, *args, **kwargs):
        raise RuntimeError("injected: the profile insert failed")

    with monkeypatch.context() as patched:
        patched.setattr(ProfileActions, "create", fail)
        # Setup treats the profile as best-effort, so it still succeeds, and
        # its personal org means no later org-scoped request re-runs it.
        await get_or_create_user_with_status(_claims(user_id, email))
    assert await prisma.profile.find_unique(where={"userId": user_id}) is None

    profile = await get_profile(user_id=user_id)

    assert profile.name == email.split("@")[0]
    assert await prisma.profile.count(where={"userId": user_id}) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_first_page_profile_reads_never_404_under_random_interleavings(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """Reads race account setup (whose profile insert may fail) and the user
    saving their profile, with random yields before every Profile query and
    handles that collide with another user's. Every read gets a profile, the
    user ends with exactly one, and a read never replaces what they saved."""
    _interleave_profile_queries(monkeypatch, await _a_taken_handle())
    run = uuid.uuid4().hex[:6]

    for seed in range(_SEEDS):
        user_id, email = await _signed_up_user()
        racers = _plan(seed)
        failures, seen = await _race(run, seed, user_id, email, racers)

        context = f"seed {seed}, racers {[racer.role for racer in racers]}"
        assert not failures, f"{context}: {failures}"
        rows = await prisma.profile.find_many(where={"userId": user_id})
        assert len(rows) == 1, f"{context}: {len(rows)} profiles"
        saved = {_saved_name(seed, i) for i, r in enumerate(racers) if r.role == "save"}
        default = email.split("@")[0]
        assert rows[0].name in (
            saved or {default}
        ), f"{context}: ended as {rows[0].name!r}"
        assert set(seen) <= saved | {default}, f"{context}: reads saw {seen}"


_RACER: ContextVar["_Racer | None"] = ContextVar("first_load_racer", default=None)


class _Racer(BaseModel):
    """One request of a new user's first page load, and what goes wrong for it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: _Role
    jitter: random.Random
    setup_insert_fails: bool
    handle_taken: bool


def _plan(seed: int) -> list[_Racer]:
    rng = random.Random(seed)
    roles: list[_Role] = [
        "read",
        *rng.choices(_ROLES, weights=[5, 3, 2], k=rng.randint(1, 4)),
    ]
    rng.shuffle(roles)
    return [
        _Racer(
            role=role,
            jitter=random.Random(rng.getrandbits(32)),
            setup_insert_fails=rng.random() < 0.3,
            handle_taken=rng.random() < 0.2,
        )
        for role in roles
    ]


async def _race(
    run: str, seed: int, user_id: str, email: str, racers: list[_Racer]
) -> tuple[list[str], list[str]]:
    """Run *racers* at once; return what failed and the names the reads saw."""
    seen: list[str] = []

    async def act(index: int, racer: _Racer) -> None:
        _RACER.set(racer)
        await _yield_randomly()
        if racer.role == "read":
            seen.append((await get_profile(user_id=user_id)).name)
        elif racer.role == "setup":
            await get_or_create_user_with_status(_claims(user_id, email))
        else:
            await store_db.update_profile(user_id, _saved(run, seed, index))

    results = await asyncio.wait_for(
        asyncio.gather(
            *(act(i, racer) for i, racer in enumerate(racers)),
            return_exceptions=True,
        ),
        timeout=_STALL * 4,
    )
    failures = [
        f"{racer.role}: {result!r}"
        for racer, result in zip(racers, results)
        if result is not None
    ]
    return failures, seen


def _saved(run: str, seed: int, index: int) -> store_model.Profile:
    return store_model.Profile(
        username=f"saved-{run}-{seed}-{index}",
        name=_saved_name(seed, index),
        description="Saved from the profile page",
        links=[],
        avatar_url=None,
    )


def _saved_name(seed: int, index: int) -> str:
    return f"Saved {seed}-{index}"


def _interleave_profile_queries(
    monkeypatch: pytest.MonkeyPatch, taken_handle: str
) -> None:
    """Yield a random number of times before every Profile query, fail account
    setup's insert when its racer says so, and offer an already-taken handle
    first when the racer says so."""
    find_unique = _after_yielding(ProfileActions.find_unique)
    monkeypatch.setattr(ProfileActions, "find_unique", find_unique)
    find_first = _after_yielding(ProfileActions.find_first)
    monkeypatch.setattr(ProfileActions, "find_first", find_first)
    monkeypatch.setattr(
        ProfileActions, "update", _after_yielding(ProfileActions.update)
    )
    create = ProfileActions.create

    async def insert_or_fail(self, *args, **kwargs):
        await _yield_randomly()
        racer = _RACER.get()
        if racer is not None and racer.role == "setup" and racer.setup_insert_fails:
            raise RuntimeError("injected: account setup's profile insert failed")
        return await create(self, *args, **kwargs)

    monkeypatch.setattr(ProfileActions, "create", insert_or_fail)
    generate = profile_module._generate_profile_username

    async def taken_handle_first() -> str:
        racer = _RACER.get()
        if racer is not None and racer.handle_taken:
            racer.handle_taken = False
            return taken_handle
        return await generate()

    monkeypatch.setattr(
        profile_module, "_generate_profile_username", taken_handle_first
    )


def _after_yielding(query):
    async def yielding(self, *args, **kwargs):
        await _yield_randomly()
        return await query(self, *args, **kwargs)

    return yielding


async def _yield_randomly() -> None:
    racer = _RACER.get()
    if racer is None:
        return
    for _ in range(racer.jitter.randint(0, 3)):
        await asyncio.sleep(0)


async def _signed_up_user() -> tuple[str, str]:
    """A user as sign-up leaves them before account setup: a bare ``User`` row."""
    user_id = str(uuid.uuid4())
    email = f"first-load-{uuid.uuid4().hex[:8]}@example.com"
    await prisma.user.create(data={"id": user_id, "email": email})
    return user_id, email


async def _a_taken_handle() -> str:
    """A handle another user's profile already holds."""
    squatter_id, _ = await _signed_up_user()
    handle = f"taken-{uuid.uuid4().hex[:8]}"
    await prisma.profile.create(
        data={
            "userId": squatter_id,
            "name": "Squatter",
            "username": handle,
            "description": "",
            "links": [],
            "avatarUrl": "",
        }
    )
    return handle


def _claims(user_id: str, email: str) -> dict:
    return {"sub": user_id, "email": email}


async def _wait_for(event: asyncio.Event, what: str) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=_STALL)
    except TimeoutError:
        pytest.fail(f"timed out after {_STALL}s waiting for {what}")
