"""An answered card's call runs under the envelope it was held under.

One chat can hold cards from different turns. A delegated expert's thread
holds its delegated turn's calls, and those of later turns there, roots among
them: the user typing in it, or another session's message waking it. Whichever
turn folds the answers, each call runs under its own envelope, re-checked as
it runs. One whose envelope has expired, or was never recorded, or no longer
reads back, does not run, says why, and leaves no approval to replay.

Against Postgres and Redis, through the gate and ``BaseTool.execute``; the call
is a stub under ``post_to_chat_platform``'s name that records the envelope it
ran under.
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from backend.copilot.context import get_current_envelope, set_execution_context
from backend.copilot.gate import check_action, held
from backend.copilot.model import (
    ChatSession,
    get_chat_session,
    update_session_autopilot_mode,
    upsert_chat_session,
)
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ResponseType, ToolResponseBase
from backend.copilot.tree import (
    MAX_DEPTH,
    TurnEnvelope,
    get_tree_ledger,
    readmit_turn,
    root_envelope,
)
from backend.copilot.turn_queue import UNRECORDED_WAKE
from backend.data.db_accessors import review_db
from backend.data.redis_client import get_redis_async
from backend.util.encryption import JSONCryptor

_TOOL = "post_to_chat_platform"
_PARENT = "parent-session"
_CEILING = 1_000_000


class _Post(BaseTool):
    def __init__(self) -> None:
        self.runs: list[tuple[str, TurnEnvelope | None]] = []

    @property
    def name(self) -> str:
        return _TOOL

    @property
    def description(self) -> str:
        return "posts"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        self.runs.append((kwargs["text"], get_current_envelope()))
        return ToolResponseBase(
            type=ResponseType.ERROR, message=f"posted {kwargs['text']}"
        )


@pytest.fixture
def gate_on():
    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def post_tool():
    tool = _Post()
    with patch("backend.copilot.tools.get_tool", return_value=tool):
        yield tool


async def _thread(user_id: str, *, delegated: bool = True) -> ChatSession:
    """A delegated expert's thread, or the user's own chat."""
    session = await upsert_chat_session(
        ChatSession.new(
            user_id=user_id,
            dry_run=False,
            delegated_by_session_id=_PARENT if delegated else None,
        )
    )
    await update_session_autopilot_mode(session.session_id, user_id, "auto")
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    return reloaded


async def _delegated_turn(**fields) -> TurnEnvelope:
    """The envelope of a turn ``delegate_to_expert`` started, its tree open:
    a level down, its own tools, tainted by its spawner, spending its chat's."""
    envelope = TurnEnvelope(
        tree_id=f"tree-{uuid.uuid4()}",
        depth=1,
        tainted=True,
        tools=frozenset({_TOOL, "web_fetch"}),
        spend_session_id=_PARENT,
    ).model_copy(update=fields)
    await (await get_tree_ledger()).open(
        envelope.tree_id or "unused", ceiling_microdollars=_CEILING, max_nodes=10
    )
    return envelope


def _root(session: ChatSession) -> TurnEnvelope:
    """A later turn the thread runs as a root: the user typing in it, or
    another session's message waking it."""
    return root_envelope(f"turn-{uuid.uuid4()}", session_id=session.session_id)


async def _hold(
    session: ChatSession, user_id: str, text: str, envelope: TurnEnvelope | None
) -> str:
    async def in_a_turn():
        set_execution_context(user_id, session, envelope=envelope)
        return await check_action(
            _TOOL, {"text": text}, user_id, session, tool_call_id=f"call-{text}"
        )

    decision = await asyncio.create_task(in_a_turn())
    assert not decision.allowed and decision.review_id
    return decision.review_id


async def _answer(review_id: str) -> None:
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": review_id},
        data={"status": ReviewStatus.APPROVED, "reviewedAt": datetime.now(UTC)},
    )


async def _fold(
    session: ChatSession, user_id: str, turn: TurnEnvelope | None
) -> list[PendingMessage]:
    """What a turn running under ``turn`` folds in as it starts."""

    async def in_a_turn():
        set_execution_context(user_id, session, envelope=turn)
        return await held.resolve_answered(user_id, session)

    return await asyncio.create_task(in_a_turn())


def _result(delivered: list[PendingMessage], review_id: str) -> PendingMessage:
    [row] = [
        r
        for r in delivered
        if r.metadata and r.metadata["held_call"]["review_id"] == review_id
    ]
    return row


async def _card(review_id: str, user_id: str):
    rows = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
    return rows.get(review_id)


async def _close(envelope: TurnEnvelope) -> None:
    """Its tree's ledger expires, six hours after the tree opened."""
    ledger = await get_tree_ledger()
    await (await get_redis_async()).delete(ledger.key(envelope.tree_id))


async def _spend_out(envelope: TurnEnvelope) -> None:
    await (await get_tree_ledger()).charge(envelope.tree_id, _CEILING)


def _refused_for(delivered: PendingMessage, reason: str) -> None:
    assert delivered.metadata and delivered.metadata["held_call"]["outcome"] == "closed"
    assert delivered.content.count("Nothing ran") == 1
    assert reason in delivered.content


@pytest.mark.asyncio(loop_scope="session")
async def test_answers_from_two_turns_each_run_under_their_own_envelope(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """A delegated turn's call and a later root's, answered newest first, the
    delegated turn's tree closed by then. The wake starts under the newest; the
    delegated call must not run under it."""
    thread = await _thread(test_user_id)
    delegated = await _delegated_turn()
    older = await _hold(thread, test_user_id, "from the delegated turn", delegated)
    later = _root(thread)
    newer = await _hold(thread, test_user_id, "from the later root", later)
    await _answer(newer)
    await _answer(older)
    await _close(delegated)
    dispatch = AsyncMock()

    with patch("backend.copilot.executor.utils.dispatch_turn", dispatch):
        await held.wake(test_user_id, thread.session_id)
    delivered = await _fold(
        thread, test_user_id, dispatch.await_args.kwargs["envelope"]
    )

    assert dispatch.await_args.kwargs["envelope"] == later
    assert post_tool.runs == [("from the later root", later)]
    _refused_for(_result(delivered, older), "tree has closed")
    assert await _card(older, test_user_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_delegated_call_folded_by_a_later_root_keeps_its_own_limits(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Its tree still open, it runs: a level down, with its own tools and its
    taint, spending its own tree and chat, not as the root that folds it."""
    thread = await _thread(test_user_id)
    delegated = await _delegated_turn()
    review_id = await _hold(thread, test_user_id, "delegated", delegated)
    await _answer(review_id)

    [delivered] = await _fold(thread, test_user_id, _root(thread))

    assert post_tool.runs == [("delegated", delegated)]
    assert delivered.metadata and delivered.metadata["held_call"]["outcome"] == (
        "approved"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stale_approval_replayed_after_a_newer_turn_asks_again(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Answered after its tree closed. The newer turn that folds it does not run
    it, and the same call proposed there opens a fresh card rather than riding
    the old approval."""
    thread = await _thread(test_user_id)
    delegated = await _delegated_turn()
    review_id = await _hold(thread, test_user_id, "stale", delegated)
    await _close(delegated)
    await _answer(review_id)
    newer = _root(thread)

    [delivered] = await _fold(thread, test_user_id, newer)

    async def replay():
        set_execution_context(test_user_id, thread, envelope=newer)
        return await post_tool.execute(test_user_id, thread, "call-again", text="stale")

    replayed = await asyncio.create_task(replay())

    assert post_tool.runs == []
    _refused_for(delivered, "tree has closed")
    assert json.loads(str(replayed.output))["type"] == "approval_required"
    card = await _card(review_id, test_user_id)
    assert card is not None and card.status == ReviewStatus.WAITING


@pytest.mark.asyncio(loop_scope="session")
async def test_racing_wakes_and_folds_run_each_call_once_under_its_own_envelope(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Two answers wake the thread at once, and three turns fold together:
    one turn starts, and each call runs once, under the envelope it was held
    under, whichever turn claims it."""
    thread = await _thread(test_user_id)
    delegated = await _delegated_turn()
    first = await _hold(thread, test_user_id, "delegated", delegated)
    later = _root(thread)
    second = await _hold(thread, test_user_id, "root", later)
    await _answer(first)
    await _answer(second)
    dispatch = AsyncMock()

    with patch("backend.copilot.executor.utils.dispatch_turn", dispatch):
        await asyncio.gather(
            held.wake(test_user_id, thread.session_id),
            held.wake(test_user_id, thread.session_id),
        )
    folds = await asyncio.gather(
        _fold(thread, test_user_id, later),
        _fold(thread, test_user_id, _root(thread)),
        _fold(thread, test_user_id, None),
    )

    assert dispatch.await_count == 1
    assert sorted(post_tool.runs, key=lambda run: run[0]) == [
        ("delegated", delegated),
        ("root", later),
    ]
    assert sum(len(delivered) for delivered in folds) == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_an_envelope_that_expires_after_its_turn_started_stops_the_call(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Each call's wake turn was admitted under its own envelope; one tree then
    closes and the other spends its ceiling before the fold, which checks
    again."""
    thread = await _thread(test_user_id)
    expiries: dict[str, tuple[str, Callable[[TurnEnvelope], Awaitable[None]]]] = {
        "tree closed": ("tree has closed", _close),
        "ceiling spent": ("Budget limit reached", _spend_out),
    }
    reasons: dict[str, str] = {}
    for text, (reason, expire) in expiries.items():
        delegated = await _delegated_turn()
        review_id = await _hold(thread, test_user_id, text, delegated)
        await _answer(review_id)
        # What the wake's dispatch checks before its turn starts.
        await readmit_turn(delegated, user_id=test_user_id)
        await expire(delegated)
        reasons[review_id] = reason

    delivered = await _fold(thread, test_user_id, _root(thread))

    assert post_tool.runs == []
    for review_id, reason in reasons.items():
        _refused_for(_result(delivered, review_id), reason)
        assert await _card(review_id, test_user_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_an_expired_or_malformed_envelope_fails_closed(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Each tree is open, so each would pass the tree's re-check: only reading
    the envelope itself stops it."""
    thread = await _thread(test_user_id)
    broken: dict[str, tuple[dict, str]] = {
        "deadline passed": (
            {"deadline_at": datetime.now(UTC) - timedelta(minutes=1)},
            "deadline has passed",
        ),
        "child with every tool": ({"tools": None}, "could not be read back"),
        "above the root": ({"depth": -1}, "could not be read back"),
        "past the bound": ({"depth": MAX_DEPTH + 1}, "could not be read back"),
        "root of no tree": ({"depth": 0, "tree_id": ""}, "could not be read back"),
    }
    reasons: dict[str, str] = {}
    for text, (fields, reason) in broken.items():
        envelope = await _delegated_turn(**fields)
        review_id = await _hold(thread, test_user_id, text, envelope)
        await _answer(review_id)
        reasons[review_id] = reason

    delivered = await _fold(thread, test_user_id, _root(thread))

    assert post_tool.runs == []
    for review_id, reason in reasons.items():
        _refused_for(_result(delivered, review_id), reason)


@pytest.mark.parametrize(
    "delegated, by_root, runs",
    [(False, True, True), (True, True, False), (False, False, False)],
    ids=[
        "a root of the user's own chat",
        "a root outside the user's own chat",
        "a delegated turn in the user's own chat",
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_call_with_nothing_recorded_runs_only_under_a_root_of_the_users_chat(
    setup_test_user,
    test_user_id,
    gate_on,
    post_tool,
    delegated: bool,
    by_root: bool,
    runs: bool,
):
    """Held before envelopes were kept, or rebuilt from its card. Every turn of
    the user's own chat is a root, so a root there stands in for the one that
    held it; nothing stands in for a delegated turn's limits."""
    thread = await _thread(test_user_id, delegated=delegated)
    review_id = await _hold(thread, test_user_id, "unrecorded", None)
    await _answer(review_id)
    folding = _root(thread) if by_root else await _delegated_turn()

    [delivered] = await _fold(thread, test_user_id, folding)

    if runs:
        assert post_tool.runs == [("unrecorded", folding)]
        return
    assert post_tool.runs == []
    _refused_for(delivered, "was not recorded")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_held_call_whose_envelope_no_longer_reads_back_does_not_run(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Unreadable is gone: the answer rebuilds the call from its card, with
    nothing recorded, so in a delegated thread neither the wake nor the next
    turn runs it."""
    thread = await _thread(test_user_id)
    review_id = await _hold(thread, test_user_id, "garbled", await _delegated_turn())
    redis = await get_redis_async()
    key = held._key(thread.session_id)
    stored = JSONCryptor().decrypt(str(await redis.hget(key, review_id)))
    stored["envelope"]["depth"] = "deep"
    await redis.hset(key, review_id, JSONCryptor().encrypt(stored))
    await _answer(review_id)
    rows = await review_db().get_reviews_by_node_exec_ids([review_id], test_user_id)
    dispatch = AsyncMock()

    with patch("backend.copilot.executor.utils.dispatch_turn", dispatch):
        await held.wake(test_user_id, thread.session_id, rows.values())
    [delivered] = await _fold(thread, test_user_id, _root(thread))

    dispatch.assert_not_awaited()
    reloaded = await get_chat_session(thread.session_id, test_user_id)
    assert reloaded is not None and reloaded.messages[-1].content == UNRECORDED_WAKE
    assert post_tool.runs == []
    _refused_for(delivered, "was not recorded")
