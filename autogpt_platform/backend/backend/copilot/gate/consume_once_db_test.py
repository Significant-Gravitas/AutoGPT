"""One approval runs one call, however the readers of its card interleave.

Identical calls, folds of the held copy and the user's answers run as tasks
that stop wherever the gate reads or writes a card, and a schedule (scripted,
or drawn from a seed) picks which goes next. Every step runs on the real
review store against Postgres. Between steps the test checks that no card
still waiting for the user went missing; at the end, that no more calls ran
than the user approved.
"""

import asyncio
import random
from contextlib import ExitStack
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from backend.copilot.context import set_execution_context
from backend.copilot.gate import check_action, held
from backend.copilot.gate import review as review_store
from backend.copilot.model import (
    ChatSession,
    get_chat_session,
    update_session_autopilot_mode,
    upsert_chat_session,
)
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ResponseType, ToolResponseBase
from backend.copilot.tree import root_envelope
from backend.data.db_accessors import review_db

_TOOL = "post_to_chat_platform"
_ARGS = {"text": "once"}
_SEEDS = range(40)
_ACTOR: ContextVar[str | None] = ContextVar("_actor", default=None)


class _Post(BaseTool):
    """Outward, so Auto asks; records each run and whether its card was still there."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.card_at_run: list[bool] = []
        self.review_id = ""

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
        self.runs.append(kwargs)
        card = await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": self.review_id}
        )
        self.card_at_run.append(card is not None)
        return ToolResponseBase(type=ResponseType.ERROR, message="posted")


class _Interleaving:
    """Runs actors one step at a time. A step ends where the actor next
    reaches the review store, so only one actor touches the card at once."""

    def __init__(self, review_id: str) -> None:
        self.review_id = review_id
        self.lost: list[str] = []
        self._parked: dict[str, tuple[asyncio.Event, str]] = {}
        self._moved = asyncio.Event()

    def stepped(self, label: str, fn: Callable[..., Awaitable[Any]]):
        async def step(*args: Any, **kwargs: Any) -> Any:
            await self.point(label)
            return await fn(*args, **kwargs)

        return step

    async def point(self, label: str) -> None:
        name = _ACTOR.get()
        if name is None:
            return
        go = asyncio.Event()
        self._parked[name] = (go, label)
        self._moved.set()
        await go.wait()

    async def run(
        self,
        actors: dict[str, Callable[[], Awaitable[Any]]],
        pick: Callable[[dict[str, str]], str],
    ) -> dict[str, Any]:
        async def start(name: str, act: Callable[[], Awaitable[Any]]) -> Any:
            _ACTOR.set(name)
            await self.point("start")
            return await act()

        tasks = {n: asyncio.create_task(start(n, a)) for n, a in actors.items()}
        for task in tasks.values():
            task.add_done_callback(lambda _: self._moved.set())
        await self._settle(lambda: all(n in self._parked for n in tasks))
        while self._parked:
            moved = pick({n: label for n, (_, label) in self._parked.items()})
            before = await self._card()
            self._parked.pop(moved)[0].set()
            await self._settle(lambda: moved in self._parked or tasks[moved].done())
            after = await self._card()
            if (
                before is not None
                and before.status == ReviewStatus.WAITING
                and (after is None or after.createdAt != before.createdAt)
            ):
                self.lost.append(f"{moved} removed a waiting card")
        return {n: task.result() for n, task in tasks.items()}

    async def _settle(self, done: Callable[[], bool]) -> None:
        while not done():
            self._moved.clear()
            await asyncio.wait_for(self._moved.wait(), timeout=60)

    async def _card(self) -> PendingHumanReview | None:
        return await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": self.review_id}
        )


class _Scene:
    """A chat with one call parked on a card, and the actors that race over it."""

    def __init__(self, user_id: str, session: ChatSession, tool: _Post) -> None:
        self.user_id = user_id
        self.session = session
        self.tool = tool
        self.approvals = 0
        self.review_id = review_store.review_id_for(
            session.session_id, user_id, _TOOL, _ARGS
        )
        tool.review_id = self.review_id
        self.interleaving = _Interleaving(self.review_id)

    def caller(self, n: int) -> Callable[[], Awaitable[str]]:
        async def call() -> str:
            self._in_a_turn()
            result = await self.tool.execute(
                self.user_id, self.session, f"call-{n}", **_ARGS
            )
            return str(result.output)

        return call

    def fold(self) -> Callable[[], Awaitable[list[PendingMessage]]]:
        async def fold() -> list[PendingMessage]:
            return await _resolved(self.user_id, self.session)

        return fold

    def user(self, verdicts: list[ReviewStatus]) -> Callable[[], Awaitable[None]]:
        async def answer() -> None:
            for verdict in verdicts:
                answered = await PendingHumanReview.prisma().update_many(
                    where={
                        "nodeExecId": self.review_id,
                        "status": ReviewStatus.WAITING,
                    },
                    data={"status": verdict, "reviewedAt": datetime.now(UTC)},
                )
                if verdict == ReviewStatus.APPROVED:
                    self.approvals += answered
                await self.interleaving.point("answer")

        return answer

    async def park(self) -> None:
        async def in_a_turn():
            self._in_a_turn()
            return await check_action(
                _TOOL, _ARGS, self.user_id, self.session, tool_call_id="call-0"
            )

        decision = await asyncio.create_task(in_a_turn())
        assert decision.review_id == self.review_id

    async def approve(self) -> None:
        await self.user([ReviewStatus.APPROVED])()

    def _in_a_turn(self) -> None:
        sid = self.session.session_id
        set_execution_context(
            self.user_id,
            self.session,
            envelope=root_envelope(f"turn-{sid}", session_id=sid),
        )

    def stepping(self) -> ExitStack:
        """Every place the gate reads or writes a card ends a step."""
        stack = ExitStack()
        for name in ("find_review", "consume", "discard", "open_review_row"):
            if hasattr(review_store, name):
                fn = getattr(review_store, name)
                stack.enter_context(
                    patch.object(
                        review_store, name, self.interleaving.stepped(name, fn)
                    )
                )
        stack.enter_context(
            patch.object(
                held, "_claim", self.interleaving.stepped("claim", held._claim)
            )
        )
        return stack


async def _resolved(user_id: str, session: ChatSession) -> list[PendingMessage]:
    delivered: list[PendingMessage] = []
    async for _status in held.resolve_answered(user_id, session, delivered.append):
        pass
    return delivered


async def _scene(user_id: str, tool: _Post) -> _Scene:
    session = await upsert_chat_session(ChatSession.new(user_id=user_id, dry_run=False))
    await update_session_autopilot_mode(session.session_id, user_id, "auto")
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    scene = _Scene(user_id, reloaded, tool)
    await scene.park()
    return scene


def _scripted(steps: list[tuple[str, str | None]]) -> Callable[[dict[str, str]], str]:
    """Move each actor until it stops at the label, or to its end on None;
    then the rest in name order."""
    steps = list(steps)

    def pick(parked: dict[str, str]) -> str:
        while steps:
            name, until = steps[0]
            if name in parked and parked[name] != until:
                return name
            steps.pop(0)
        return sorted(parked)[0]

    return pick


@pytest.fixture
def gate_on():
    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def post_tool():
    tool = _Post()
    with patch("backend.copilot.tools.get_tool", return_value=tool):
        yield tool


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stale_approval_never_takes_the_card_an_identical_call_opened_after(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """A and B read one approval; B spends it and runs; C, the same call
    again, opens a fresh card under the same id. A's spend must miss C's card:
    A refuses, and C's card waits for the user."""
    scene = await _scene(test_user_id, post_tool)
    await scene.approve()
    first = await scene.interleaving._card()
    assert first is not None

    with scene.stepping():
        outputs = await scene.interleaving.run(
            {"A": scene.caller(1), "B": scene.caller(2), "C": scene.caller(3)},
            _scripted([("A", "consume"), ("B", None), ("C", None), ("A", None)]),
        )

    assert post_tool.runs == [_ARGS]
    assert "already used" in outputs["A"]
    assert scene.review_id in outputs["C"]
    card = await scene.interleaving._card()
    assert card is not None and card.status == ReviewStatus.WAITING
    assert card.createdAt > first.createdAt
    assert scene.interleaving.lost == []


@pytest.mark.asyncio(loop_scope="session")
async def test_one_approval_runs_at_most_once_and_loses_no_card_in_any_order(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Three identical calls, two folds and three answers, in seeded orders."""
    ran = 0
    for seed in _SEEDS:
        rng = random.Random(seed)
        post_tool.runs.clear()
        scene = await _scene(test_user_id, post_tool)
        verdicts = [
            rng.choice([ReviewStatus.APPROVED] * 3 + [ReviewStatus.REJECTED])
            for _ in range(3)
        ]
        actors = {f"call-{n}": scene.caller(n) for n in (1, 2, 3)}
        actors |= {"fold-1": scene.fold(), "fold-2": scene.fold()}
        actors["user"] = scene.user(verdicts)

        with scene.stepping():
            await scene.interleaving.run(
                actors, lambda parked: rng.choice(sorted(parked))
            )

        assert scene.interleaving.lost == [], f"seed {seed}"
        assert len(post_tool.runs) <= scene.approvals, f"seed {seed}"
        ran += len(post_tool.runs)
    assert ran, "no seed ran the call at all"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_fold_with_the_gate_off_spends_the_approval_before_it_runs(
    setup_test_user, test_user_id, gate_on, post_tool
):
    scene = await _scene(test_user_id, post_tool)
    await scene.approve()

    with patch(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=False)
    ):
        [delivered] = await _resolved(test_user_id, scene.session)

    assert post_tool.runs == [_ARGS]
    assert post_tool.card_at_run == [False]
    assert delivered.metadata["held_call"]["outcome"] == "approved"


class _DeletesFail:
    """The review store's database, with every delete failing."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        if name.startswith("delete_review"):
            return AsyncMock(side_effect=ConnectionError("database unreachable"))
        return getattr(self._real, name)


def _deletes_fail():
    real = review_db()
    return patch("backend.copilot.gate.review.review_db", lambda: _DeletesFail(real))


@pytest.mark.parametrize("gate", [True, False])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_fold_that_cannot_spend_the_approval_runs_nothing(
    setup_test_user, test_user_id, gate_on, post_tool, gate
):
    """Unspent, the approval would buy another run; so the call waits, held,
    for the next turn, and its card stays approved."""
    scene = await _scene(test_user_id, post_tool)
    await scene.approve()

    with _deletes_fail(), patch(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=gate)
    ):
        await _resolved(test_user_id, scene.session)

    assert post_tool.runs == []
    card = await scene.interleaving._card()
    assert card is not None and card.status == ReviewStatus.APPROVED
    [kept] = await held.answered(test_user_id, scene.session.session_id)
    assert kept.review_id == scene.review_id


@pytest.mark.asyncio(loop_scope="session")
async def test_a_call_whose_approval_cannot_be_spent_runs_nothing(
    setup_test_user, test_user_id, gate_on, post_tool
):
    scene = await _scene(test_user_id, post_tool)
    await scene.approve()

    with _deletes_fail():
        output = await scene.caller(1)()

    assert post_tool.runs == []
    assert "nothing ran" in output.lower() or "already used" in output
    card = await scene.interleaving._card()
    assert card is not None and card.status == ReviewStatus.APPROVED
