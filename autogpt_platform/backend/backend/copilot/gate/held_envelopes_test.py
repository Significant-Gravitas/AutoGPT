"""An approved held call runs only under the envelope it was held under.

The fold runs every answered card at the start of whichever turn comes next,
so one chat's cards can belong to different turns: a delegated turn of one
tree, a later root, a turn whose tree has since closed. This drives the real
fold through seeded interleavings of folding turns, answers landing, records
rebuilt from their cards, trees expiring or spending out, and Redis and
Postgres failing, over in-memory stand-ins for both, and checks every run
against the call's own records.
"""

import asyncio
import logging
import random
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.context import get_current_envelope, set_execution_context
from backend.copilot.gate import held
from backend.copilot.gate import review as review_store
from backend.copilot.model import ChatSession
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.tree import MAX_DEPTH, TurnEnvelope, narrowed_by, root_envelope
from backend.util.encryption import JSONCryptor

_SEEDS = 300
_USER = "user-1"
_PARENT = "parent-session"
_TOOLS = ("post_to_chat_platform", "create_agent", "run_agent", "message_session")
_TREES = ("tree-a", "tree-b", "tree-c")
_LEDGER = "copilot:tree:"
_REFUSED = "Propose it again if it is still needed."
# The folding turn an operation runs in, so a read or a claim is credited to it.
_fold: ContextVar[int | None] = ContextVar("_fold", default=None)

Run = tuple[int, int | None, str, TurnEnvelope | None]


class _World:
    """Redis hashes and review rows, with a step counter, random yields, and
    random events between any two operations once the setup is done."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.calm = True
        self.step = 0
        self.hashes: dict[str, dict[str, Any]] = {}
        self.rows: dict[str, SimpleNamespace] = {}
        # (step, fold, tree, admitted): every read of a tree's ledger.
        self.checks: list[tuple[int, int | None, str, bool]] = []
        # Every record of a held call written once play began: rebuilt from its
        # card, or put back after a failure.
        self.written: dict[str, list[held.HeldCall]] = {}
        # (step, fold, review id, envelope in effect): every run.
        self.runs: list[Run] = []

    async def tick(self, may_fail: bool = True) -> None:
        self.step += 1
        if self.calm:
            return
        for _ in range(self.rng.randint(0, 2)):
            await asyncio.sleep(0)
        roll = self.rng.random()
        if roll < 0.05:
            # A tree's ledger expires.
            self.hashes.pop(_LEDGER + self.rng.choice(_TREES), None)
        elif roll < 0.08:
            ledger = self.hashes.get(_LEDGER + self.rng.choice(_TREES))
            if ledger:
                ledger["spent"] = ledger["ceiling"]
        elif roll < 0.12:
            # An answer lands while the turns fold.
            waiting = [
                r for r in self.rows.values() if r.status == ReviewStatus.WAITING
            ]
            if waiting:
                row = self.rng.choice(waiting)
                row.status, row.reviewed_at = ReviewStatus.APPROVED, datetime.now(UTC)
        if may_fail and self.rng.random() < 0.03:
            raise ConnectionError("injected")


class _Redis:
    def __init__(self, world: _World) -> None:
        self.w = world

    async def hgetall(self, key: str) -> dict[str, Any]:
        await self.w.tick()
        return dict(self.w.hashes.get(key, {}))

    async def hdel(self, key: str, field: str) -> int:
        await self.w.tick()
        return int(self.w.hashes.get(key, {}).pop(field, None) is not None)

    async def hmget(self, key: str, fields: list[str]) -> list[Any]:
        await self.w.tick()
        bucket = self.w.hashes.get(key, {})
        if key.startswith(_LEDGER):
            ceiling, spent = bucket.get("ceiling"), bucket.get("spent")
            admitted = ceiling is not None and int(spent) < int(ceiling)
            tree = key.removeprefix(_LEDGER)
            self.w.checks.append((self.w.step, _fold.get(), tree, admitted))
        return [bucket.get(f) for f in fields]

    def pipeline(self, transaction: bool = True) -> "_Pipeline":
        return _Pipeline(self.w)


class _Pipeline:
    def __init__(self, world: _World) -> None:
        self.w = world
        self.writes: list[tuple[str, str, Any]] = []

    async def __aenter__(self) -> "_Pipeline":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def hset(self, key: str, field: str, value: Any) -> None:
        self.writes.append((key, field, value))

    def expire(self, key: str, seconds: int) -> None:
        pass

    async def execute(self) -> list[int]:
        await self.w.tick()
        for key, field, value in self.writes:
            self.w.hashes.setdefault(key, {})[field] = value
            if not self.w.calm and key.startswith(held._KEY):
                record = held.HeldCall.model_validate(JSONCryptor().decrypt(value))
                self.w.written.setdefault(field, []).append(record)
        return [1] * len(self.writes)


class _Reviews:
    def __init__(self, world: _World) -> None:
        self.w = world

    async def get_reviews_by_node_exec_ids(self, ids: list[str], _user: str):
        await self.w.tick()
        # A snapshot, as Postgres returns.
        rows = self.w.rows
        return {i: SimpleNamespace(**vars(rows[i])) for i in ids if i in rows}

    async def delete_review_by_node_exec_id(self, review_id: str, _user: str) -> int:
        await self.w.tick()
        return int(self.w.rows.pop(review_id, None) is not None)


class _Tool:
    """Spends the approval first, as the gate does, then records what it ran
    under; now and then it fails after running."""

    def __init__(self, world: _World, name: str) -> None:
        self.w = world
        self.name = name

    async def execute(self, user_id: str, session: ChatSession, _id: str, **args):
        review_id = review_store.review_id_for(
            session.session_id, user_id, self.name, args
        )
        if not await review_store.consume(review_id, user_id):
            return SimpleNamespace(output="not approved")
        self.w.runs.append(
            (self.w.step, _fold.get(), review_id, get_current_envelope())
        )
        await self.w.tick(may_fail=False)
        if self.w.rng.random() < 0.03:
            raise RuntimeError("failed after running")
        return SimpleNamespace(output=f"ran {review_id}")


@pytest.mark.asyncio
async def test_a_held_call_runs_only_under_its_own_unexpired_envelope(
    caplog: pytest.LogCaptureFixture,
):
    """Each seed is one chat: calls held under roots, children of open, spent
    and closed trees, malformed envelopes and none, some of their records gone
    or unreadable, answered in any order and folded by concurrent turns of any
    envelope. A failing seed is named."""
    # Every injected failure logs a traceback; formatting them is most of the run.
    for name in ("backend.copilot.gate", "backend.copilot.tree"):
        caplog.set_level(logging.CRITICAL, logger=name)
    ran = refused = 0
    for seed in range(_SEEDS):
        seed_ran, seed_refused = await _seed(seed)
        ran += seed_ran
        refused += seed_refused
    # Not vacuous: across the seeds, calls both run and are refused for their
    # envelope.
    assert ran > _SEEDS // 3 and refused > _SEEDS // 3, (ran, refused)


async def _seed(seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    world = _World(rng)
    redis, reviews = _Redis(world), _Reviews(world)
    tools = {name: _Tool(world, name) for name in _TOOLS}
    with (
        patch(
            "backend.copilot.gate.held.get_redis_async",
            AsyncMock(return_value=redis),
        ),
        patch("backend.copilot.tree.get_redis_async", AsyncMock(return_value=redis)),
        patch("backend.copilot.tree.is_feature_enabled", AsyncMock(return_value=False)),
        patch("backend.copilot.gate.held.review_db", return_value=reviews),
        patch("backend.copilot.gate.review.review_db", return_value=reviews),
        patch("backend.copilot.gate.chat_rules.set_ask", AsyncMock()),
        patch("backend.copilot.tools.get_tool", side_effect=tools.get),
    ):
        return await _play(seed, rng, world)


async def _play(seed: int, rng: random.Random, world: _World) -> tuple[int, int]:
    own_chat = rng.random() < 0.5
    session = ChatSession.new(
        _USER,
        dry_run=False,
        builder_graph_id="graph-1" if rng.random() < 0.3 else None,
        delegated_by_session_id=None if own_chat else _PARENT,
    )
    calls, records, never = await _hold_calls(seed, rng, world, session)
    turns: dict[int, TurnEnvelope | None] = {}
    delivered: list[str] = []

    async def fold(index: int) -> None:
        _fold.set(index)
        turns[index] = turn = rng.choice(
            [
                None,
                root_envelope(f"turn-{seed}-{index}", session_id=session.session_id),
                *(c.envelope for c in calls.values() if c.envelope),
            ]
        )
        set_execution_context(_USER, session, envelope=turn)
        rows = await held.resolve_answered(_USER, session)
        delivered.extend(row.content for row in rows)

    async def answer() -> None:
        # The answers' route hands the wake the cards it answered, and the wake
        # rebuilds from them every call whose record is gone or unreadable.
        rows = [
            SimpleNamespace(**vars(r))
            for r in world.rows.values()
            if r.status != ReviewStatus.WAITING
        ]
        await held._restore(_USER, session.session_id, rows)

    world.calm = False
    first = rng.randint(1, 3)
    await asyncio.gather(answer(), *(fold(i) for i in range(first)))
    # A later turn picks up what landed or came back meanwhile.
    await fold(first)
    world.calm = True

    ran = [run[2] for run in world.runs]
    assert len(ran) == len(set(ran)), f"seed {seed}: a call ran twice"
    for run in world.runs:
        _check(seed, run, calls, records, never, turns, world, session)
    refused = [text for text in delivered if _REFUSED in text]
    return len(world.runs), len(refused)


async def _hold_calls(
    seed: int, rng: random.Random, world: _World, session: ChatSession
) -> tuple[dict[str, held.HeldCall], dict[str, str], set[str]]:
    """Calls held in one chat by turns of different envelopes, the trees they
    belong to, their cards, and their records: kept, gone, or unreadable. Also
    the cards that must never run."""
    now = datetime.now(UTC)
    for tree in _TREES:
        state = rng.choice(["open", "open", "spent", "closed"])
        if state != "closed":
            world.hashes[_LEDGER + tree] = {
                "ceiling": "1000000",
                "spent": "1000000" if state == "spent" else "0",
                "max_nodes": "10",
                "nodes": "1",
            }
    calls: dict[str, held.HeldCall] = {}
    records: dict[str, str] = {}
    never: set[str] = set()
    for index in range(rng.randint(2, 6)):
        tool, args = rng.choice(_TOOLS), {"text": f"{seed}-{index}"}
        review_id = review_store.review_id_for(session.session_id, _USER, tool, args)
        call = held.HeldCall(
            review_id=review_id,
            tool_name=tool,
            tool_call_id=f"call-{index}",
            args=args,
            held_at=now + timedelta(seconds=index),
            envelope=_envelope(rng, index, session.session_id),
        )
        calls[review_id] = call
        records[review_id] = await _record(rng, session.session_id, call, world)
        status, age = rng.choice(
            [
                (ReviewStatus.APPROVED, timedelta(minutes=5)),
                (ReviewStatus.APPROVED, timedelta(minutes=5)),
                (ReviewStatus.APPROVED, timedelta(hours=2)),
                (ReviewStatus.REJECTED, timedelta(minutes=5)),
                (ReviewStatus.WAITING, timedelta(0)),
            ]
        )
        if status == ReviewStatus.REJECTED or age > review_store.APPROVAL_TTL:
            never.add(review_id)
        world.rows[review_id] = SimpleNamespace(
            node_exec_id=review_id,
            session_id=session.session_id,
            status=status,
            # The card's copy of the call, which a lost record is rebuilt from.
            payload={"arguments": args, "tool_call_id": call.tool_call_id},
            reviewed_at=None if status == ReviewStatus.WAITING else now - age,
            updated_at=None,
            created_at=now - timedelta(hours=3),
        )
    return calls, records, never


async def _record(
    rng: random.Random, session_id: str, call: held.HeldCall, world: _World
) -> str:
    """Keep the call's record, lose it, or leave one that no longer reads back."""
    kind = rng.choice(["kept"] * 6 + ["gone", "unreadable"])
    if kind == "kept":
        assert await held.remember(session_id, call)
    elif kind == "unreadable":
        garbled = {**call.model_dump(mode="json"), "envelope": {"depth": "deep"}}
        world.hashes.setdefault(held._key(session_id), {})[call.review_id] = rng.choice(
            ["not a token", JSONCryptor().encrypt(garbled)]
        )
    return kind


def _envelope(rng: random.Random, index: int, session_id: str) -> TurnEnvelope | None:
    """A root, a child of one of the trees, nothing recorded, or a malformed
    child that no derivation could produce."""
    kind = rng.choice(["root", "child", "child", "lost", "malformed"])
    if kind == "lost":
        return None
    if kind == "root":
        root = root_envelope(f"root-{index}", session_id=session_id)
        if rng.random() < 0.3:
            return root.model_copy(update={"tools": frozenset(rng.sample(_TOOLS, 2))})
        return root
    now = datetime.now(UTC)
    child = TurnEnvelope(
        tree_id=rng.choice(_TREES),
        depth=rng.randint(1, MAX_DEPTH),
        tainted=rng.random() < 0.5,
        tools=frozenset(rng.sample(_TOOLS, rng.randint(1, len(_TOOLS)))),
        deadline_at=rng.choice(
            [None, now - timedelta(minutes=1), now + timedelta(hours=1)]
        ),
        spend_session_id=_PARENT,
    )
    if kind == "child":
        return child
    broken = rng.choice(
        [
            {"depth": -1},
            {"depth": MAX_DEPTH + 1},
            {"tree_id": ""},
            {"tools": None},
            {"deadline_at": (now + timedelta(hours=1)).replace(tzinfo=None)},
        ]
    )
    return child.model_copy(update=broken)


def _check(
    seed: int,
    run: Run,
    calls: dict[str, held.HeldCall],
    records: dict[str, str],
    never: set[str],
    turns: dict[int, TurnEnvelope | None],
    world: _World,
    session: ChatSession,
) -> None:
    """The run was under one of the call's own records: the one it was held
    with or one written for it since, never another call's or turn's. Under a
    recorded envelope, as the session allows it now, unexpired, its tree read
    live by the same fold; under a record rebuilt from its card with nothing
    recorded, as a root of the user's own chat. A record that no longer read
    back never runs, nor does one whose details were lost."""
    step, fold_index, review_id, in_effect = run
    call = calls[review_id]
    where = f"seed {seed}: {call.tool_name} held under {call.envelope}"
    assert review_id not in never, f"{where} ran on a stale or declined card"
    assert records[review_id] != "unreadable", f"{where} ran, its record unreadable"
    kept = [call] if records[review_id] == "kept" else []
    runnable = [r for r in kept + world.written.get(review_id, []) if not r.lost]
    permissions = resolve_session_permissions(session)
    matches = [
        r.envelope
        for r in runnable
        if r.envelope is not None and in_effect == narrowed_by(r.envelope, permissions)
    ]
    if not matches:
        turn = turns[fold_index] if fold_index is not None else None
        assert any(r.envelope is None for r in runnable), f"{where} ran as {in_effect}"
        assert session.metadata.delegated_by_session_id is None, f"{where} ran"
        assert turn is None or turn.depth == 0, f"{where} ran under {turn}"
        # A root stands in, the folding turn's or, outside one, its own: as the
        # chat's permissions allow it now.
        assert in_effect is not None, f"{where} ran unbounded"
        root = turn or root_envelope(in_effect.tree_id, session_id=session.session_id)
        assert in_effect == narrowed_by(
            root, permissions
        ), f"{where} ran as {in_effect}"
        assert in_effect.permits(call.tool_name), f"{where} ran a tool it may not"
        return
    envelope = matches[0]
    assert envelope.tree_id and 0 <= envelope.depth <= MAX_DEPTH, f"{where} ran"
    assert envelope.depth == 0 or envelope.tools is not None, f"{where} ran"
    deadline = envelope.deadline_at
    assert deadline is None or deadline.tzinfo is not None, f"{where} ran"
    assert deadline is None or deadline > datetime.now(UTC), f"{where} ran late"
    assert in_effect is not None and in_effect.permits(
        call.tool_name
    ), f"{where} ran a tool it may not"
    if envelope.depth:
        reads = [
            admitted
            for at, fold, tree, admitted in world.checks
            if fold == fold_index and tree == envelope.tree_id and at <= step
        ]
        assert reads and reads[-1], f"{where} ran without its tree read live"
