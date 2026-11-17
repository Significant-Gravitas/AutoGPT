"""The harness for ``consume_once_db_test``: actors that race over one
approval card, stepped one at a time against the real review store and the
real held-call store.

Identical calls, folds of the held copy, the user's answers and restores from
the card run as tasks that stop wherever the gate reads or writes a card or a
held call, and a schedule (scripted, or drawn from a seed) picks which goes
next. Between steps the harness checks what that one step did: no card still
waiting went missing, no approved card went without its call running, and no
held call was removed except by the fold that listed it.
"""

import asyncio
from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Iterator, Literal, NamedTuple
from unittest.mock import patch

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
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
# What a held-store write is for: a park, a fold's recovery, a restore.
_WRITE: ContextVar[str] = ContextVar("_write", default="park")


class Seen(NamedTuple):
    """The card and the held calls under its id, between two steps."""

    card: tuple[ReviewStatus, datetime] | None
    entries: Counter[str]
    runs: int


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
        # The held calls each step's claims meant to take, by tool call id.
        self.claimed: list[str] = []
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
        seen: Callable[[], Awaitable[Seen]] | None = None,
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
                await self._step(pick, tasks, seen)
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
        seen: Callable[[], Awaitable[Seen]] | None,
    ) -> None:
        parked = {n: label for n, (_, label) in self._parked.items()}
        moved = pick(parked)
        before = await self._seen(seen)
        self.claimed = []
        self._parked.pop(moved)[0].set()
        await self._settle(lambda: moved in self._parked or tasks[moved].done())
        after = await self._seen(seen)
        self.lost += _judged(f"{moved} at {parked[moved]}", before, after, self.claimed)

    async def _seen(self, seen: Callable[[], Awaitable[Seen]] | None) -> Seen:
        if seen is not None:
            return await seen()
        card = await self.card()
        # Runs aren't counted here, so only a lost waiting card is judged.
        return Seen(card and (card.status, card.createdAt), Counter(), -1)

    async def _settle(self, done: Callable[[], bool]) -> None:
        while not done():
            self._moved.clear()
            await asyncio.wait_for(self._moved.wait(), timeout=60)

    async def card(self) -> PendingHumanReview | None:
        return await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": self.review_id}
        )


def _judged(step: str, before: Seen, after: Seen, claimed: list[str]) -> list[str]:
    """What one step did that no actor may do."""
    found = []
    if before.card is not None and (
        after.card is None or after.card[1] != before.card[1]
    ):
        if before.card[0] == ReviewStatus.WAITING:
            found.append(f"{step} removed a waiting card")
        ran = after.runs != before.runs or before.runs < 0
        if before.card[0] == ReviewStatus.APPROVED and not ran:
            found.append(f"{step} removed an approved card and nothing ran")
    for tool_call_id, gone in (before.entries - after.entries).items():
        if gone > claimed.count(tool_call_id):
            found.append(f"{step} removed {tool_call_id}'s held call")
    return found


class Scene:
    """A chat with one call parked on a card, and the actors that race over it."""

    def __init__(
        self,
        user_id: str,
        session: ChatSession,
        tool: PostStub,
        args: dict[str, Any],
    ) -> None:
        self.user_id = user_id
        self.session = session
        self.tool = tool
        self.args = args
        self.approvals = 0
        self.delivered: list[PendingMessage] = []
        self.review_id = review_store.review_id_for(
            session.session_id, user_id, TOOL, args
        )
        tool.review_id = self.review_id
        self.interleaving = Interleaving(self.review_id)
        # The held calls each actor's fold listed, for what its claims meant.
        self._listed: dict[str, list[held.HeldCall]] = {}

    def caller(self, n: int) -> Callable[[], Awaitable[str]]:
        async def call() -> str:
            self.enter_turn()
            result = await self.tool.execute(
                self.user_id, self.session, f"call-{n}", **self.args
            )
            return str(result.output)

        return call

    def fold(self) -> Callable[[], Awaitable[None]]:
        async def fold() -> None:
            async for _status in held.resolve_answered(
                self.user_id, self.session, self.delivered.append
            ):
                pass

        return fold

    def restorer(self) -> Callable[[], Awaitable[None]]:
        """An answer's wake, restoring the held call from its card if gone."""

        async def restore() -> None:
            rows = await review_db().get_reviews_by_node_exec_ids(
                [self.review_id], self.user_id
            )
            answered = [r for r in rows.values() if r.status != ReviewStatus.WAITING]
            await held._restore(self.user_id, self.session.session_id, answered)

        return restore

    async def seen(self) -> Seen:
        card = await self.interleaving.card()
        calls = await held._held(self.session.session_id)
        return Seen(
            card and (card.status, card.createdAt),
            Counter(c.tool_call_id for c in calls.values()),
            len(self.tool.runs),
        )

    async def settle(self) -> None:
        """Later turns, one at a time, fold whatever is still answered."""
        for _ in range(3):
            await resolved(self.user_id, self.session)

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
                TOOL, self.args, self.user_id, self.session, tool_call_id=tool_call_id
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

    def stepping_the_held_store(self, pause_after_listing: bool = False) -> ExitStack:
        """As :meth:`stepping`, with every held-call write a step too, and a
        fold named ``failing…`` failing before it spends anything.
        ``pause_after_listing`` also ends a step once a fold has listed its
        answered calls, before it reads each one's card again."""
        interleaving = self.interleaving
        claim, remember, listed = held._claim, held.remember, held.answered
        stack = self.stepping()
        recover, restore = held._recover, held._restore
        run, run_handler = held._run, held._run_handler

        async def claims(session_id: str, entry: Any) -> bool:
            await interleaving.point("claim")
            interleaving.claimed.append(self._meant(entry))
            return await claim(session_id, entry)

        async def remembers(*args: Any, **kwargs: Any) -> bool:
            await interleaving.point(f"remember({_WRITE.get()})")
            return await remember(*args, **kwargs)

        async def lists(user_id: str, session_id: str) -> list[held.HeldCall]:
            calls = await listed(user_id, session_id)
            self._listed[_ACTOR.get() or ""] = calls
            if pause_after_listing:
                await interleaving.point("listed")
            return calls

        def writing(as_: str, fn: Callable[..., Awaitable[Any]]):
            async def write(*args: Any, **kwargs: Any) -> Any:
                token = _WRITE.set(as_)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _WRITE.reset(token)

            return write

        def failing(fn: Callable[..., Awaitable[Any]]):
            async def runs(*args: Any, **kwargs: Any) -> Any:
                if (_ACTOR.get() or "").startswith("failing"):
                    raise RuntimeError("failed before the spend")
                return await fn(*args, **kwargs)

            return runs

        for name, fn in (
            ("_claim", claims),
            ("remember", remembers),
            ("answered", lists),
            ("_recover", writing("recover", recover)),
            ("_restore", writing("restore", restore)),
            ("_run", failing(run)),
            ("_run_handler", failing(run_handler)),
        ):
            stack.enter_context(patch.object(held, name, fn))
        return stack

    def _meant(self, entry: Any) -> str:
        """The tool call whose held call a claim meant to take: the call
        itself, or the one the fold listed under that review id."""
        if isinstance(entry, held.HeldCall):
            return entry.tool_call_id
        listed = self._listed.get(_ACTOR.get() or "", [])
        return next((c.tool_call_id for c in listed if c.review_id == entry), "")


async def resolved(user_id: str, session: ChatSession) -> list[PendingMessage]:
    delivered: list[PendingMessage] = []
    async for _status in held.resolve_answered(user_id, session, delivered.append):
        pass
    return delivered


async def new_scene(user_id: str, tool: PostStub, args: dict[str, Any] = ARGS) -> Scene:
    session = await upsert_chat_session(ChatSession.new(user_id=user_id, dry_run=False))
    await update_session_autopilot_mode(session.session_id, user_id, "auto")
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    scene = Scene(user_id, reloaded, tool, args)
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
    and card reads failing once ``reads_fail`` is set, which
    ``reads_fail_after_delete`` sets on the first delete. Everything else
    reaches the real store."""

    def __init__(self, real: Any, deletes: Deletes, reads_fail_after_delete: bool):
        self.reads_fail = False
        self._real = real
        self._deletes = deletes
        self._reads_fail_after_delete = reads_fail_after_delete

    async def get_reviews_by_node_exec_ids(
        self, node_exec_ids: list[str], user_id: str
    ) -> dict[str, PendingHumanReviewModel]:
        if self.reads_fail:
            raise ConnectionError("database unreachable")
        return await self._real.get_reviews_by_node_exec_ids(node_exec_ids, user_id)

    async def delete_review_as_read(
        self,
        node_exec_id: str,
        user_id: str,
        status: ReviewStatus,
        created_at: datetime,
    ) -> int:
        return await self._deleting(
            lambda: self._real.delete_review_as_read(
                node_exec_id, user_id, status, created_at
            )
        )

    async def delete_review_by_node_exec_id(
        self, node_exec_id: str, user_id: str
    ) -> int:
        return await self._deleting(
            lambda: self._real.delete_review_by_node_exec_id(node_exec_id, user_id)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def _deleting(self, delete: Callable[[], Awaitable[int]]) -> int:
        try:
            if self._deletes == "fail":
                raise ConnectionError("database unreachable")
            deleted = await delete()
            if self._deletes == "commit_then_fail":
                raise ConnectionError("database unreachable")
            return deleted
        finally:
            if self._reads_fail_after_delete:
                self.reads_fail = True


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
