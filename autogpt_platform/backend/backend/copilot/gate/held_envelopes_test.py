"""An approval is spent only under the authority it was given under.

The fold runs every answered card at the start of whichever turn comes next,
and any turn may propose the same call again, so one chat's approvals meet
turns of other authorities: a delegated turn of one tree, a later root, a turn
whose tree has since closed or whose deadline has passed. This drives the real
gate and fold through seeded interleavings of holds, answers, folds, calls
proposed again, records lost or garbled and rebuilt from their cards, trees
expiring or spending out, and Redis and Postgres failing (reads, writes and the
deletes that spend an approval), over in-memory stand-ins for both. Every
approval spent is checked against the authority its card was opened under.
"""

import asyncio
import itertools
import json
import logging
import random
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.context import get_current_envelope, set_execution_context
from backend.copilot.gate import check_action, held
from backend.copilot.gate import review as review_store
from backend.copilot.gate.headline import Headline
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
# What a refusal says when it invites the call again.
_INVITES = ("Propose it again", "Propose the call again", "send the request again")
# The turn an operation runs in, so a read, a spend or a claim is credited to it.
_turn: ContextVar[int | None] = ContextVar("_turn", default=None)
_again: ContextVar[bool] = ContextVar("_again", default=False)

# (step, turn, review id, envelope in effect, proposed again, approval spent)
Run = tuple[int, int | None, str, TurnEnvelope | None, bool, int]


class _World:
    """Redis hashes and review rows, with a step counter, random yields, and
    random events between any two operations once the setup is done. Each
    review row carries a generation: one approval, spent at most once."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.calm = True
        self.step = 0
        self.gens = itertools.count(1)
        self.hashes: dict[str, dict[str, Any]] = {}
        self.rows: dict[str, SimpleNamespace] = {}
        # The authority each approval's card was opened under, by generation.
        self.authority: dict[int, TurnEnvelope | None] = {}
        # Generations declined or approved too long ago: never spendable.
        self.never: set[int] = set()
        # Generations a fold read before inviting their call again.
        self.invited: dict[str, set[int]] = {}
        # The generation each turn last read, by review id; and the approval
        # each turn last spent, which is the one a call it runs rides on.
        self.read: dict[tuple[int | None, str], int] = {}
        self.spent: dict[int | None, tuple[str, int]] = {}
        # (step, turn, tree, admitted): every read of a tree's ledger.
        self.checks: list[tuple[int, int | None, str, bool]] = []
        # Every record of a held call ever written, by review id.
        self.written: dict[str, list[held.HeldCall]] = {}
        # The tool and arguments behind every card, as the model would repeat them.
        self.calls: dict[str, tuple[str, dict]] = {}
        self.runs: list[Run] = []
        # Calls the gate let through on no approval at all: never in Ask First.
        self.unapproved: list[tuple[TurnEnvelope | None, dict]] = []
        # Approvals a call of another tool or other arguments ran on.
        self.misspent: list[str] = []
        # Postgres operations that fail until a scripted test lets them pass.
        self.failing: set[str] = set()

    async def tick(self, may_fail: bool = True, failure_rate: float = 0.04) -> None:
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
        elif roll < 0.14:
            # An answer lands while the turns run.
            waiting = [
                r for r in self.rows.values() if r.status == ReviewStatus.WAITING
            ]
            if waiting:
                row = self.rng.choice(waiting)
                row.status, row.reviewed_at = ReviewStatus.APPROVED, datetime.now(UTC)
        if may_fail and self.rng.random() < failure_rate:
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
            self.w.checks.append((self.w.step, _turn.get(), tree, admitted))
        return [bucket.get(f) for f in fields]

    def pipeline(self, transaction: bool = True) -> "_Pipeline":
        return _Pipeline(self.w)


class _Pipeline:
    def __init__(self, world: _World) -> None:
        self.w = world
        self.writes: list[tuple[str, str, Any, bool]] = []

    async def __aenter__(self) -> "_Pipeline":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def hset(self, key: str, field: str, value: Any) -> None:
        self.writes.append((key, field, value, True))

    def hsetnx(self, key: str, field: str, value: Any) -> None:
        self.writes.append((key, field, value, False))

    def expire(self, key: str, seconds: int) -> None:
        pass

    async def execute(self) -> list[int]:
        await self.w.tick()
        done = []
        for key, field, value, overwrite in self.writes:
            bucket = self.w.hashes.setdefault(key, {})
            if field in bucket and not overwrite:
                done.append(0)
                continue
            bucket[field] = value
            done.append(1)
            if key.startswith(held._KEY):
                record = held.HeldCall.model_validate(JSONCryptor().decrypt(value))
                self.w.written.setdefault(field, []).append(record)
        return done


class _Reviews:
    def __init__(self, world: _World) -> None:
        self.w = world

    async def get_reviews_by_node_exec_ids(self, ids: list[str], _user: str):
        await self.w.tick()
        if "read" in self.w.failing:
            raise ConnectionError("read fails")
        found = {}
        for review_id in ids:
            row = self.w.rows.get(review_id)
            if row is not None:
                self.w.read[(_turn.get(), review_id)] = row.gen
                # A snapshot, as Postgres returns.
                found[review_id] = SimpleNamespace(**vars(row))
        return found

    async def delete_review_by_node_exec_id(self, review_id: str, _user: str) -> int:
        # Spending an approval is what a refusal must not report before it is
        # done, so it fails more often than the rest.
        await self.w.tick(failure_rate=0.15)
        if "delete" in self.w.failing:
            raise ConnectionError("delete fails")
        row = self.w.rows.pop(review_id, None)
        if row is None:
            return 0
        self.w.spent[_turn.get()] = (review_id, row.gen)
        return 1


async def _open_review(world: _World, review_id, _user, session, tool, args, *_a, **kw):
    """The card's row, opened under the authority of the call that parks it."""
    await world.tick()
    world.calls.setdefault(review_id, (tool, args))
    if review_id not in world.rows:
        gen = next(world.gens)
        world.authority[gen] = get_current_envelope()
        world.rows[review_id] = SimpleNamespace(
            node_exec_id=review_id,
            session_id=session.session_id,
            status=ReviewStatus.WAITING,
            payload={"arguments": args, "tool_call_id": kw.get("tool_call_id", "")},
            reviewed_at=None,
            updated_at=None,
            created_at=datetime.now(UTC) - timedelta(hours=3),
            gen=gen,
        )
    return Headline(ask="Do it")


class _Tool:
    """``BaseTool.execute`` where it matters here: the turn's envelope must
    allow the tool, then the real gate decides, and a gate that fails refuses.
    Records every run with the approval it spent; now and then it fails after
    running."""

    def __init__(self, world: _World, name: str) -> None:
        self.w = world
        self.name = name

    async def execute(self, user_id: str, session: ChatSession, call_id: str, **args):
        envelope = get_current_envelope()
        if envelope is not None and not envelope.permits(self.name):
            return SimpleNamespace(output='{"type": "error"}')
        try:
            decision = await check_action(self.name, args, user_id, session, call_id)
        except Exception:
            return SimpleNamespace(output='{"type": "error"}')
        if not decision.allowed:
            body = {
                "type": "approval_required",
                "review_id": decision.review_id,
                "reason": decision.reason,
            }
            return SimpleNamespace(output=json.dumps(body))
        if not decision.approved:
            # The gate is off: the call runs, and no approval is spent.
            self.w.unapproved.append((envelope, args))
            return SimpleNamespace(output="ran without an approval")
        # The approval the gate just spent, as the review store saw it: a turn
        # runs one call at a time, so no other spend of its comes between.
        turn = _turn.get()
        review_id, gen = self.w.spent[turn]
        if self.w.calls[review_id] != (self.name, args):
            self.w.misspent.append(review_id)
        self.w.runs.append((self.w.step, turn, review_id, envelope, _again.get(), gen))
        await self.w.tick(may_fail=False)
        if self.w.rng.random() < 0.03:
            # As BaseTool.execute reports a tool that raised after it ran.
            return SimpleNamespace(output='{"type": "error"}')
        return SimpleNamespace(output=f"ran {review_id}")


@pytest.mark.asyncio
async def test_an_approval_is_spent_only_under_its_own_unexpired_authority(
    caplog: pytest.LogCaptureFixture,
):
    """Each seed is one chat in Ask First: calls held under roots, expired
    roots, children of open, spent and closed trees, malformed envelopes and
    none, some records gone or unreadable, answered in any order, folded and
    proposed again by concurrent turns of any envelope. A failing seed is
    named."""
    # Every injected failure logs a traceback; formatting them is most of the run.
    for name in ("backend.copilot.gate", "backend.copilot.tree"):
        caplog.set_level(logging.CRITICAL, logger=name)
    totals = [0, 0, 0]
    for seed in range(_SEEDS):
        for i, count in enumerate(await _seed(seed)):
            totals[i] += count
    folded, again, refused = totals
    # Not vacuous: approvals are spent by folds and by calls proposed again,
    # and folds refuse calls their authority no longer covers.
    assert folded > _SEEDS // 5 and again > _SEEDS // 5, totals
    assert refused > _SEEDS // 2, totals


async def _seed(seed: int) -> tuple[int, int, int]:
    rng = random.Random(seed)
    world = _World(rng)
    with stand_ins(world) as tools:
        return await _play(seed, rng, world, tools)


@contextmanager
def stand_ins(world: _World) -> Iterator[dict[str, _Tool]]:
    """Redis, the review store and the tools as ``world`` holds them, the gate
    on, and no chat rule set: every action asks in Ask First."""
    redis, reviews = _Redis(world), _Reviews(world)
    tools = {name: _Tool(world, name) for name in _TOOLS}

    async def open_review(*args: Any, **kwargs: Any) -> Headline:
        return await _open_review(world, *args, **kwargs)

    with (
        patch(
            "backend.copilot.gate.held.get_redis_async",
            AsyncMock(return_value=redis),
        ),
        patch("backend.copilot.tree.get_redis_async", AsyncMock(return_value=redis)),
        patch("backend.copilot.tree.is_feature_enabled", AsyncMock(return_value=False)),
        patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)),
        patch("backend.copilot.gate.held.review_db", return_value=reviews),
        patch("backend.copilot.gate.review.review_db", return_value=reviews),
        patch("backend.copilot.gate.review.open_review", open_review),
        patch("backend.copilot.gate.chat_rules.set_ask", AsyncMock()),
        patch("backend.copilot.gate.chat_rules.rule_for", AsyncMock(return_value=None)),
        patch("backend.copilot.tools.get_tool", side_effect=tools.get),
    ):
        yield tools


async def _play(
    seed: int, rng: random.Random, world: _World, tools: dict[str, _Tool]
) -> tuple[int, int, int]:
    own_chat = rng.random() < 0.5
    session = ChatSession.new(
        _USER,
        dry_run=False,
        builder_graph_id="graph-1" if rng.random() < 0.3 else None,
        delegated_by_session_id=None if own_chat else _PARENT,
    )
    session.metadata.autopilot_mode = "ask_first"
    _open_trees(rng, world)
    plans, records = await _hold_calls(seed, rng, world, session, tools)
    envelopes = [e for _, _, e in plans.values() if e is not None]
    turns: dict[int, TurnEnvelope | None] = {}
    refused = 0

    async def turn(index: int) -> None:
        nonlocal refused
        _turn.set(index)
        _again.set(False)
        turns[index] = here = rng.choice(
            [
                None,
                root_envelope(f"turn-{seed}-{index}", session_id=session.session_id),
                *envelopes,
            ]
        )
        set_execution_context(_USER, session, envelope=here)
        invited = []
        for row in await held.resolve_answered(_USER, session):
            meta = (row.metadata or {})["held_call"]
            if any(invite in row.content for invite in _INVITES):
                refused += 1
                gen = world.read.get((index, meta["review_id"]))
                world.invited.setdefault(meta["review_id"], set()).add(gen or 0)
                invited.append(meta["review_id"])
        # The model mostly does as invited, then proposes other calls again,
        # some exactly as held before.
        _again.set(True)
        again = [rid for rid in invited if rng.random() < 0.8]
        again += [
            rng.choice(list(plans)) for _ in range(rng.randint(0, 3) if plans else 0)
        ]
        for review_id in again:
            tool, args = world.calls[review_id]
            await tools[tool].execute(_USER, session, f"again-{index}", **args)

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
    await asyncio.gather(answer(), *(turn(i) for i in range(first)))
    # A later turn picks up what landed or came back meanwhile; a task of its
    # own, as each turn is, so what it sets stays its own.
    await asyncio.create_task(turn(first))
    world.calm = True

    assert not world.unapproved, f"seed {seed}: an action ran without an approval"
    assert not world.misspent, f"seed {seed}: a call ran on another call's approval"
    spent = [run[5] for run in world.runs]
    assert len(spent) == len(set(spent)), f"seed {seed}: an approval was spent twice"
    for run in world.runs:
        _check(seed, run, records, turns, world, session)
    again = sum(1 for run in world.runs if run[4])
    return len(world.runs) - again, again, refused


def _open_trees(rng: random.Random, world: _World) -> None:
    for tree in _TREES:
        state = rng.choice(["open", "open", "open", "spent", "closed"])
        if state != "closed":
            world.hashes[_LEDGER + tree] = {
                "ceiling": "1000000",
                "spent": "1000000" if state == "spent" else "0",
                "max_nodes": "10",
                "nodes": "1",
            }


async def _hold_calls(
    seed: int,
    rng: random.Random,
    world: _World,
    session: ChatSession,
    tools: dict[str, _Tool],
) -> tuple[dict[str, tuple[str, dict, TurnEnvelope | None]], dict[str, str]]:
    """Calls held through the gate by turns of different envelopes; their
    cards answered or not, and their records kept, gone or unreadable."""
    plans: dict[str, tuple[str, dict, TurnEnvelope | None]] = {}
    records: dict[str, str] = {}
    now = datetime.now(UTC)
    for index in range(rng.randint(2, 6)):
        tool, args = rng.choice(_TOOLS), {"text": f"{seed}-{index}"}
        envelope = _envelope(rng, index, session.session_id, tool)

        async def hold(envelope=envelope, tool=tool, args=args, index=index):
            set_execution_context(_USER, session, envelope=envelope)
            return await tools[tool].execute(_USER, session, f"call-{index}", **args)

        review_id = json.loads((await asyncio.create_task(hold())).output).get(
            "review_id"
        )
        if not review_id:
            # Its own envelope does not allow the tool: nothing was held.
            continue
        plans[review_id] = (tool, args, envelope)
        records[review_id] = _record(rng, session.session_id, review_id, world)
        row = world.rows[review_id]
        status, age = rng.choice(
            [
                (ReviewStatus.APPROVED, timedelta(minutes=5)),
                (ReviewStatus.APPROVED, timedelta(minutes=5)),
                (ReviewStatus.APPROVED, timedelta(minutes=5)),
                (ReviewStatus.APPROVED, timedelta(hours=2)),
                (ReviewStatus.REJECTED, timedelta(minutes=5)),
                (ReviewStatus.WAITING, timedelta(0)),
            ]
        )
        row.status = status
        row.reviewed_at = None if status == ReviewStatus.WAITING else now - age
        if status == ReviewStatus.REJECTED or age > review_store.APPROVAL_TTL:
            world.never.add(row.gen)
    return plans, records


def _record(rng: random.Random, session_id: str, review_id: str, world: _World) -> str:
    """Keep the record the gate wrote, lose it, or garble it past reading."""
    kind = rng.choice(["kept"] * 6 + ["gone", "unreadable"])
    bucket = world.hashes[held._key(session_id)]
    if kind == "gone":
        bucket.pop(review_id)
    elif kind == "unreadable":
        garbled = {"review_id": review_id, "envelope": {"depth": "deep"}}
        bucket[review_id] = rng.choice(["not a token", JSONCryptor().encrypt(garbled)])
    return kind


def _envelope(
    rng: random.Random, index: int, session_id: str, tool: str
) -> TurnEnvelope | None:
    """A root, tainted or not, a root past a deadline, a child of one of the
    trees, usually allowed the tool it holds, nothing recorded, or a malformed
    child that no derivation could produce."""
    kind = rng.choice(
        ["root", "root", "late root", "child", "child", "child", "lost", "malformed"]
    )
    now = datetime.now(UTC)
    if kind == "lost":
        return None
    if kind in ("root", "late root"):
        root = root_envelope(
            f"root-{index}", tainted=rng.random() < 0.3, session_id=session_id
        )
        if kind == "late root":
            return root.model_copy(update={"deadline_at": now - timedelta(minutes=1)})
        if rng.random() < 0.3:
            return root.model_copy(update={"tools": frozenset(rng.sample(_TOOLS, 2))})
        return root
    tools = set(rng.sample(_TOOLS, rng.randint(1, len(_TOOLS))))
    if rng.random() < 0.8:
        tools.add(tool)
    child = TurnEnvelope(
        tree_id=rng.choice(_TREES),
        depth=rng.randint(1, MAX_DEPTH),
        tainted=rng.random() < 0.5,
        tools=frozenset(tools),
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


def _authority(envelope: TurnEnvelope | None) -> tuple:
    """What an approval was given to, so what may spend it. Every root of a
    chat may do what any other may, so they share one, told apart only by
    taint and a deadline; no envelope at all is an untainted root. A delegated
    turn's is its tree, its depth, its taint, its deadline and the chat whose
    ceiling it spends. The tools a turn may use are not part of it: a turn
    cannot call a tool its envelope lacks, approved or not."""
    if envelope is None:
        return ("root", False, None)
    if envelope.depth == 0:
        return ("root", envelope.tainted, envelope.deadline_at)
    return (
        envelope.tree_id,
        envelope.depth,
        envelope.tainted,
        envelope.deadline_at,
        envelope.spend_session_id,
    )


def _check(
    seed: int,
    run: Run,
    records: dict[str, str],
    turns: dict[int, TurnEnvelope | None],
    world: _World,
    session: ChatSession,
) -> None:
    """Whoever spent the approval did so under the authority its card was
    opened under, while that authority could still act, never on an approval
    declined, too old, or already reported as spent; a fold, moreover, only
    under one of the call's own records, and never from an unreadable one.
    The same call proposed again under that authority is the call the
    approval was given for, whatever became of its record."""
    step, turn, review_id, in_effect, again, gen = run
    held_under = world.authority[gen]
    where = f"seed {seed}: {review_id[-6:]} approved under {held_under}, run"
    assert _authority(in_effect) == _authority(
        held_under
    ), f"{where} under another authority: {in_effect}"
    assert gen not in world.never, f"{where} on a declined or stale approval"
    assert gen not in world.invited.get(
        review_id, set()
    ), f"{where} on an approval its call was invited again over"
    if again:
        if in_effect is not None:
            _check_may_act(where, step, turn, in_effect, world)
        return
    _check_fold(where, run, records, turns, world, session)


def _check_fold(
    where: str,
    run: Run,
    records: dict[str, str],
    turns: dict[int, TurnEnvelope | None],
    world: _World,
    session: ChatSession,
) -> None:
    step, turn, review_id, in_effect, _, _ = run
    runnable = [r for r in world.written.get(review_id, []) if not r.lost]
    permissions = resolve_session_permissions(session)
    matches = [
        r.envelope
        for r in runnable
        if r.envelope is not None and in_effect == narrowed_by(r.envelope, permissions)
    ]
    if matches:
        _check_may_act(where, step, turn, matches[0], world)
        return
    # Nothing recorded: a root of the user's own chat stands in, the folding
    # turn's or, outside one, its own, as the chat's permissions allow now.
    # A call rebuilt from its card has no rule key; rebuilt from a record that
    # no longer read back, it must be lost, so only one parked afresh counts.
    unrecorded = [r for r in runnable if r.envelope is None]
    if records.get(review_id) == "unreadable":
        unrecorded = [r for r in unrecorded if r.rule_key is not None]
    here = turns.get(turn) if turn is not None else None
    assert unrecorded, f"{where} as {in_effect}, its record unreadable or other"
    assert session.metadata.delegated_by_session_id is None, f"{where} elsewhere"
    assert here is None or here.depth == 0, f"{where} under {here}"
    assert in_effect is not None, f"{where} unbounded"
    root = here or root_envelope(in_effect.tree_id, session_id=session.session_id)
    assert in_effect == narrowed_by(root, permissions), f"{where} as {in_effect}"


def _check_may_act(
    where: str, step: int, turn: int | None, envelope: TurnEnvelope, world: _World
) -> None:
    """Well formed, before its deadline, and, below the root, its tree read
    open and under its ceiling by the same turn before it ran."""
    assert envelope.tree_id and 0 <= envelope.depth <= MAX_DEPTH, f"{where} malformed"
    assert envelope.depth == 0 or envelope.tools is not None, f"{where} malformed"
    deadline = envelope.deadline_at
    assert deadline is None or deadline.tzinfo is not None, f"{where} malformed"
    assert deadline is None or deadline > datetime.now(UTC), f"{where} late"
    if envelope.depth:
        reads = [
            admitted
            for at, by, tree, admitted in world.checks
            if by == turn and tree == envelope.tree_id and at <= step
        ]
        assert reads and reads[-1], f"{where} without its tree read live"
