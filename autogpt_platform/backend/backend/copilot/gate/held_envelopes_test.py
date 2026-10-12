"""An approved held call runs only under the envelope it was held under.

The fold runs every answered card at the start of whichever turn comes next,
so one chat's cards can belong to different turns: a delegated turn of one
tree, a later root, a turn whose tree has since closed. This drives the real
fold through seeded interleavings of folding turns, answers landing, trees
expiring or spending out, and Redis and Postgres failing, over in-memory
stand-ins for both, and checks every run against the call's own envelope.
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

_SEEDS = 300
_USER = "user-1"
_PARENT = "parent-session"
_TOOLS = ("post_to_chat_platform", "create_agent", "run_agent", "message_session")
_TREES = ("tree-a", "tree-b", "tree-c")
_LEDGER = "copilot:tree:"
# The folding turn an operation runs in, so a ledger read is credited to it.
_fold: ContextVar[int | None] = ContextVar("_fold", default=None)


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
        # (step, fold, review id, envelope in effect): every run.
        self.runs: list[tuple[int, int | None, str, TurnEnvelope | None]] = []

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
    and closed trees, malformed envelopes and none, answered in any order and
    folded by concurrent turns of any envelope. A failing seed is named."""
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
    calls, never = await _hold_calls(seed, rng, world, session)
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

    world.calm = False
    first = rng.randint(1, 3)
    await asyncio.gather(*(fold(i) for i in range(first)))
    # A later turn picks up what landed or came back meanwhile.
    await fold(first)
    world.calm = True

    ran = [run[2] for run in world.runs]
    assert len(ran) == len(set(ran)), f"seed {seed}: a call ran twice"
    for run in world.runs:
        _check(seed, run, calls, never, turns, world, session)
    refused = [text for text in delivered if "can no longer carry it out" in text]
    return len(world.runs), len(refused)


async def _hold_calls(
    seed: int, rng: random.Random, world: _World, session: ChatSession
) -> tuple[dict[str, held.HeldCall], set[str]]:
    """Calls held in one chat by turns of different envelopes, the trees they
    belong to, and their cards; and the cards that must never run."""
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
        assert await held.remember(session.session_id, call)
        calls[review_id] = call
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
            payload={},
            reviewed_at=None if status == ReviewStatus.WAITING else now - age,
            updated_at=None,
            created_at=now - timedelta(hours=3),
        )
    return calls, never


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
    run: tuple[int, int | None, str, TurnEnvelope | None],
    calls: dict[str, held.HeldCall],
    never: set[str],
    turns: dict[int, TurnEnvelope | None],
    world: _World,
    session: ChatSession,
) -> None:
    """The run was the call's own: under its envelope as the session allows it
    now, unexpired, its tree read live by the same fold; or, with nothing
    recorded, under a root of the user's own chat."""
    step, fold_index, review_id, in_effect = run
    call = calls[review_id]
    where = f"seed {seed}: {call.tool_name} held under {call.envelope}"
    assert review_id not in never, f"{where} ran on a stale or declined card"
    envelope = call.envelope
    if envelope is None:
        turn = turns[fold_index] if fold_index is not None else None
        assert session.metadata.delegated_by_session_id is None, f"{where} ran"
        assert turn is None or turn.depth == 0, f"{where} ran under {turn}"
        assert in_effect == turn, f"{where} ran under {in_effect}"
        return
    assert envelope.tree_id and 0 <= envelope.depth <= MAX_DEPTH, f"{where} ran"
    assert envelope.depth == 0 or envelope.tools is not None, f"{where} ran"
    deadline = envelope.deadline_at
    assert deadline is None or deadline.tzinfo is not None, f"{where} ran"
    assert deadline is None or deadline > datetime.now(UTC), f"{where} ran late"
    own = narrowed_by(envelope, resolve_session_permissions(session))
    assert in_effect == own, f"{where} ran under {in_effect}"
    assert own.permits(call.tool_name), f"{where} ran a tool it may not"
    if envelope.depth:
        reads = [
            admitted
            for at, fold, tree, admitted in world.checks
            if fold == fold_index and tree == envelope.tree_id and at <= step
        ]
        assert reads and reads[-1], f"{where} ran without its tree read live"
