"""The harness for ``consume_once_db_test``: actors that race over one
approval card, stepped one at a time against the real review store.

Identical calls, folds of the held copy and the user's answers run as tasks
that stop wherever the gate reads or writes a card, and a schedule (scripted,
or drawn from a seed) picks which goes next. Between steps the harness notes
any card still waiting for the user that went missing.
"""

import asyncio
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Iterator, Literal
from unittest.mock import AsyncMock, patch

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

TOOL = "post_to_chat_platform"
ARGS = {"text": "once"}
_ACTOR: ContextVar[str | None] = ContextVar("_actor", default=None)


class PostStub(BaseTool):
    """Outward, so Auto asks; records each run and whether its card was still there."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.card_at_run: list[bool] = []
        self.review_id = ""
        self.during: Callable[[], Awaitable[None]] | None = None

    @property
    def name(self) -> str:
        return TOOL

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
        if self.during is not None:
            await self.during()
        return ToolResponseBase(type=ResponseType.ERROR, message="posted")


class Interleaving:
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
        try:
            await self._settle(lambda: all(n in self._parked for n in tasks))
            while self._parked:
                await self._step(pick, tasks)
            return {n: task.result() for n, task in tasks.items()}
        finally:
            # A step that timed out or failed leaves no actor behind.
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def _step(
        self,
        pick: Callable[[dict[str, str]], str],
        tasks: dict[str, "asyncio.Task[Any]"],
    ) -> None:
        moved = pick({n: label for n, (_, label) in self._parked.items()})
        before = await self.card()
        self._parked.pop(moved)[0].set()
        await self._settle(lambda: moved in self._parked or tasks[moved].done())
        after = await self.card()
        if (
            before is not None
            and before.status == ReviewStatus.WAITING
            and (after is None or after.createdAt != before.createdAt)
        ):
            self.lost.append(f"{moved} removed a waiting card")

    async def _settle(self, done: Callable[[], bool]) -> None:
        while not done():
            self._moved.clear()
            await asyncio.wait_for(self._moved.wait(), timeout=60)

    async def card(self) -> PendingHumanReview | None:
        return await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": self.review_id}
        )


class Scene:
    """A chat with one call parked on a card, and the actors that race over it."""

    def __init__(self, user_id: str, session: ChatSession, tool: PostStub) -> None:
        self.user_id = user_id
        self.session = session
        self.tool = tool
        self.approvals = 0
        self.review_id = review_store.review_id_for(
            session.session_id, user_id, TOOL, ARGS
        )
        tool.review_id = self.review_id
        self.interleaving = Interleaving(self.review_id)

    def caller(self, n: int) -> Callable[[], Awaitable[str]]:
        async def call() -> str:
            self.enter_turn()
            result = await self.tool.execute(
                self.user_id, self.session, f"call-{n}", **ARGS
            )
            return str(result.output)

        return call

    def fold(self) -> Callable[[], Awaitable[list[PendingMessage]]]:
        async def fold() -> list[PendingMessage]:
            return await resolved(self.user_id, self.session)

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

    async def park(self, tool_call_id: str = "call-0") -> None:
        async def in_a_turn():
            self.enter_turn()
            return await check_action(
                TOOL, ARGS, self.user_id, self.session, tool_call_id=tool_call_id
            )

        decision = await asyncio.create_task(in_a_turn())
        assert decision.review_id == self.review_id

    async def approve(self) -> None:
        await self.user([ReviewStatus.APPROVED])()

    def enter_turn(self) -> None:
        sid = self.session.session_id
        set_execution_context(
            self.user_id,
            self.session,
            envelope=root_envelope(f"turn-{sid}", session_id=sid),
        )

    def stepping(self) -> ExitStack:
        """Every place the gate reads or writes a card ends a step."""
        stack = ExitStack()
        for name, fn in (
            ("find_review", review_store.find_review),
            ("consume", review_store.consume),
            ("discard", review_store.discard),
            ("open_review_row", review_store.open_review_row),
        ):
            stack.enter_context(
                patch.object(review_store, name, self.interleaving.stepped(name, fn))
            )
        stack.enter_context(
            patch.object(
                held, "_claim", self.interleaving.stepped("claim", held._claim)
            )
        )
        return stack


async def resolved(user_id: str, session: ChatSession) -> list[PendingMessage]:
    delivered: list[PendingMessage] = []
    async for _status in held.resolve_answered(user_id, session, delivered.append):
        pass
    return delivered


async def new_scene(user_id: str, tool: PostStub) -> Scene:
    session = await upsert_chat_session(ChatSession.new(user_id=user_id, dry_run=False))
    await update_session_autopilot_mode(session.session_id, user_id, "auto")
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    scene = Scene(user_id, reloaded, tool)
    await scene.park()
    return scene


def scripted(steps: list[tuple[str, str | None]]) -> Callable[[dict[str, str]], str]:
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


Deletes = Literal["ok", "fail", "commit_then_fail"]


class FaultyReviews:
    """The review store's database, with faults: ``deletes`` that fail, or
    that delete the row and then fail (as a delete whose answer was lost),
    and every read failing once ``reads_fail`` is set, which
    ``reads_fail_after_delete`` sets on the first delete."""

    def __init__(self, real: Any, deletes: Deletes, reads_fail_after_delete: bool):
        self.reads_fail = False
        self._real = real
        self._deletes = deletes
        self._reads_fail_after_delete = reads_fail_after_delete

    def __getattr__(self, name: str) -> Any:
        if name.startswith("get_") and self.reads_fail:
            return AsyncMock(side_effect=ConnectionError("database unreachable"))
        found = getattr(self._real, name)
        if not name.startswith("delete_review"):
            return found

        async def delete(*args: Any, **kwargs: Any) -> int:
            try:
                if self._deletes == "fail":
                    raise ConnectionError("database unreachable")
                deleted = await found(*args, **kwargs)
                if self._deletes == "commit_then_fail":
                    raise ConnectionError("database unreachable")
                return deleted
            finally:
                if self._reads_fail_after_delete:
                    self.reads_fail = True

        return delete


@contextmanager
def faulty_reviews(
    deletes: Deletes = "ok", reads_fail_after_delete: bool = False
) -> Iterator[FaultyReviews]:
    database = FaultyReviews(review_db(), deletes, reads_fail_after_delete)
    with ExitStack() as stack:
        for module in ("review", "held"):
            stack.enter_context(
                patch(f"backend.copilot.gate.{module}.review_db", lambda: database)
            )
        yield database


class AnHourLater(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.now(tz) + review_store.APPROVAL_TTL + timedelta(seconds=1)
