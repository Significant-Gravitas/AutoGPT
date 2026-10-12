"""Each guard on an approved call, taken on its own.

``held_envelopes_test.py`` checks the invariant the guards hold together over
seeded interleavings. Some of them back each other up, so one removed there
can go unseen while the rest stand. Each test here sets up the case one guard
exists for: the gate, the fold and the held records real, Redis, the review
store and the tools the in-memory stand-ins of that file.
"""

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, TypeVar
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.context import set_execution_context
from backend.copilot.gate import held
from backend.copilot.gate.held_envelopes_test import (
    _LEDGER,
    _PARENT,
    _USER,
    _Tool,
    _World,
    stand_ins,
)
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.tree import TurnEnvelope, root_envelope
from backend.util.encryption import JSONCryptor

_POST = "post_to_chat_platform"
_TREE = "tree-a"
T = TypeVar("T")


@contextmanager
def _world() -> Iterator[tuple[_World, dict[str, _Tool]]]:
    world = _World(random.Random(0))
    with stand_ins(world) as tools:
        yield world, tools


def _chat(*, own: bool = False) -> ChatSession:
    """A chat in Ask First: a delegated expert's thread, or the user's own."""
    session = ChatSession.new(
        _USER, dry_run=False, delegated_by_session_id=None if own else _PARENT
    )
    session.metadata.autopilot_mode = "ask_first"
    return session


def _child() -> TurnEnvelope:
    """A delegated turn of ``_TREE``, allowed to post."""
    return TurnEnvelope(
        tree_id=_TREE, depth=1, tools=frozenset({_POST}), spend_session_id=_PARENT
    )


def _root(session: ChatSession) -> TurnEnvelope:
    return root_envelope(f"turn-{random.random()}", session_id=session.session_id)


def _open_tree(world: _World) -> None:
    world.hashes[_LEDGER + _TREE] = {
        "ceiling": "1000000",
        "spent": "0",
        "max_nodes": "10",
        "nodes": "1",
    }


def _approve(world: _World, review_id: str) -> None:
    row = world.rows[review_id]
    row.status, row.reviewed_at = ReviewStatus.APPROVED, datetime.now(UTC)


async def _in_turn(
    session: ChatSession, envelope: TurnEnvelope, step: Callable[[], Awaitable[T]]
) -> T:
    """``step`` as a turn running under ``envelope`` takes it: a task of its own."""

    async def turn() -> T:
        set_execution_context(_USER, session, envelope=envelope)
        return await step()

    return await asyncio.create_task(turn())


async def _propose(
    tools: dict[str, _Tool], session: ChatSession, envelope: TurnEnvelope
) -> dict[str, Any]:
    """What the model gets back proposing the post in a turn under ``envelope``."""
    out = await _in_turn(
        session,
        envelope,
        lambda: tools[_POST].execute(_USER, session, "call-1", text="hello"),
    )
    return json.loads(out.output) if out.output.startswith("{") else {"ran": True}


async def _fold(session: ChatSession, envelope: TurnEnvelope) -> list[PendingMessage]:
    return await _in_turn(
        session, envelope, lambda: held.resolve_answered(_USER, session)
    )


@pytest.mark.asyncio
async def test_a_held_record_is_never_written_over():
    """The review id binds the call, its arguments and its authority, so a
    record standing under it is the call's own; a second write, whatever it
    carries, leaves it be."""
    with _world() as (world, tools):
        session = _chat()
        _open_tree(world)
        review_id = (await _propose(tools, session, _child()))["review_id"]
        other = held.HeldCall(
            review_id=review_id,
            tool_name=_POST,
            tool_call_id="another",
            args={"text": "hello"},
            envelope=_root(session),
        )

        assert await held.remember(session.session_id, other)

        [record] = (await held._held(session.session_id)).values()
        assert record.envelope == _child() and record.tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_a_card_that_cannot_be_read_refuses_the_call():
    """Unread is not absent: the same call proposed while its card cannot be
    read, under the card's own authority or a root's, is refused, and nothing
    is parked over the card or beside it."""
    with _world() as (world, tools):
        session = _chat()
        _open_tree(world)
        review_id = (await _propose(tools, session, _child()))["review_id"]
        records = dict(world.hashes[held._key(session.session_id)])
        world.failing.add("read")

        same = await _propose(tools, session, _child())
        from_root = await _propose(tools, session, _root(session))

        assert same == from_root == {"type": "error"}
        assert list(world.rows) == [review_id]
        assert world.hashes[held._key(session.session_id)] == records


@pytest.mark.asyncio
async def test_a_record_unlike_its_card_is_refused_with_the_gate_off():
    """With the flag switched off since the hold, the gate lets every call
    through, so the fold's own check is all that keeps a call to the authority
    its card was approved under: a record carrying another, however it came
    to, is refused rather than run."""
    with _world() as (world, tools):
        session = _chat()
        _open_tree(world)
        review_id = (await _propose(tools, session, _child()))["review_id"]
        _approve(world, review_id)
        other = held.HeldCall(
            review_id=review_id,
            tool_name=_POST,
            tool_call_id="call-1",
            args={"text": "hello"},
            envelope=_root(session),
        )
        world.hashes[held._key(session.session_id)][review_id] = JSONCryptor().encrypt(
            other.model_dump(mode="json")
        )

        with patch(
            "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=False)
        ):
            [delivered] = await _fold(session, _root(session))

        assert world.runs == [] and world.unapproved == []
        assert "does not match its approval" in delivered.content


@pytest.mark.asyncio
async def test_a_refusal_whose_approval_cannot_be_spent_invites_nothing():
    """The fold refuses a call whose tree has closed, but its approval cannot
    be deleted: nothing is delivered and the call is kept, rather than invite
    it again over an approval still live. Once the delete goes through, the
    refusal goes out and the approval is spent."""
    with _world() as (world, tools):
        session = _chat()
        _open_tree(world)
        review_id = (await _propose(tools, session, _child()))["review_id"]
        _approve(world, review_id)
        world.hashes.pop(_LEDGER + _TREE)
        world.failing.add("delete")

        assert await _fold(session, _root(session)) == []
        assert world.rows[review_id].status == ReviewStatus.APPROVED
        assert review_id in await held._held(session.session_id)

        world.failing.clear()
        [delivered] = await _fold(session, _root(session))

        assert "Propose it again" in delivered.content
        assert review_id not in world.rows and world.runs == []


@pytest.mark.asyncio
async def test_an_approval_its_turn_may_no_longer_use_is_left_for_the_fold():
    """The delegated turn proposes its call again after its tree closed: the
    gate refuses it and leaves the approval to the fold, which refuses the
    call too, spends the approval and says why. Nothing runs."""
    with _world() as (world, tools):
        session = _chat()
        _open_tree(world)
        review_id = (await _propose(tools, session, _child()))["review_id"]
        _approve(world, review_id)
        world.hashes.pop(_LEDGER + _TREE)

        again = await _propose(tools, session, _child())

        assert again.get("review_id") is None
        assert "cannot be used now" in again.get("reason", "")
        assert world.rows[review_id].status == ReviewStatus.APPROVED
        [delivered] = await _fold(session, _root(session))
        assert "tree has closed" in delivered.content
        assert review_id not in world.rows and world.runs == []


@pytest.mark.asyncio
async def test_a_record_that_no_longer_reads_back_is_rebuilt_never_to_run():
    """In the user's own chat a call whose record is gone is rebuilt from its
    card and runs under a root; one whose record is there but no longer reads
    back may have been bounded more than its card says, so it never runs."""
    with _world() as (world, tools):
        session = _chat(own=True)
        review_id = (await _propose(tools, session, _root(session)))["review_id"]
        world.hashes[held._key(session.session_id)][review_id] = "not a token"
        _approve(world, review_id)
        answered = SimpleNamespace(**vars(world.rows[review_id]))

        await held._restore(_USER, session.session_id, [answered])
        [delivered] = await _fold(session, _root(session))

        assert world.runs == [] and world.unapproved == []
        assert "details were lost" in delivered.content
