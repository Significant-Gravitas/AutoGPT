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
import random
from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable, Iterator, Literal
from unittest.mock import AsyncMock, patch

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview
from pydantic import BaseModel

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.context import set_execution_context
from backend.copilot.gate import Decision, check_action, held
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
from backend.data.redis_client import get_redis_async
from backend.util.encryption import JSONCryptor
from backend.util.json import SafeJson

TOOL = "post_to_chat_platform"
ARGS = {"text": "once"}
_ACTOR: ContextVar[str | None] = ContextVar("_actor", default=None)
# What a held-store write is for: a park, a fold's recovery, a restore.
_WRITE: ContextVar[str] = ContextVar("_write", default="park")
# Set while a fold (an actor's, or a later turn's) runs held calls.
_FOLDING: ContextVar[bool] = ContextVar("_folding", default=False)


class Seen(BaseModel):
    """The card and the held calls under its id, between two steps."""

    # Status, creation time, the tool call whose park opened it, and when it
    # was answered.
    card: tuple[ReviewStatus, datetime, str, datetime | None] | None
    entries: Counter[str]
    # Each held call under the id, as stored, by its slot.
    slots: dict[str, str] = {}
    runs: int
    # Of those, the runs a fold made: each must spend an approval.
    fold_runs: int = 0
    # Every result a fold delivered so far, as (tool call, outcome).
    results: list[tuple[str, str]] = []
    gate_on: bool = True
    # The cards a faulted delete was aimed at so far, by creation time (None
    # for a delete by id, which takes whichever card is there).
    faulted: list[datetime | None] = []


class PostStub(BaseTool):
    """Outward, so Auto asks; records each run and whether its card was still there."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.fold_runs = 0
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
        self.fold_runs += _FOLDING.get()
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
        # A task an actor starts inherits its name; two parked under one name
        # would lose one of them.
        assert name not in self._parked, f"{name} is already parked"
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
        return Seen(card=_card_seen(card), entries=Counter(), runs=-1)

    async def _settle(self, done: Callable[[], bool]) -> None:
        while not done():
            self._moved.clear()
            await asyncio.wait_for(self._moved.wait(), timeout=60)

    async def card(self) -> PendingHumanReview | None:
        return await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": self.review_id}
        )


def _card_seen(
    card: PendingHumanReview | None,
) -> tuple[ReviewStatus, datetime, str, datetime | None] | None:
    if card is None:
        return None
    payload = card.payload if isinstance(card.payload, dict) else {}
    owner = str(payload.get("tool_call_id") or "")
    return card.status, card.createdAt, owner, card.reviewedAt


def _judged(step: str, before: Seen, after: Seen, claimed: list[str]) -> list[str]:
    """What one step did that no actor may do."""
    found = []
    counted = before.runs >= 0
    ran = after.runs - before.runs if counted else 0
    fold_ran = after.fold_runs - before.fold_runs if counted else 0
    spent = False
    if before.card is not None and (
        after.card is None or after.card[1] != before.card[1]
    ):
        status, _, owner, answered_at = before.card
        if status == ReviewStatus.WAITING:
            found.append(f"{step} removed a waiting card")
        if status == ReviewStatus.APPROVED:
            spent = True
            lapsed = answered_at is not None and (
                datetime.now(UTC) - answered_at > review_store.APPROVAL_TTL
            )
            # A spend whose answer was lost, or a lapsed card, runs nothing;
            # nor does a restore's copy of the card (it has no cost context),
            # whose fold discards the card and says nothing ran.
            settled_a_copy = any(
                _restored(before.slots[slot])
                for slot in before.slots.keys() - after.slots.keys()
            )
            # Only the card the fault was aimed at is excused.
            aimed = after.faulted[len(before.faulted) :]
            faulted = before.card[1] in aimed or None in aimed
            excused = lapsed or faulted or not counted or settled_a_copy
            if ran == 0 and not excused:
                found.append(f"{step} removed an approved card and nothing ran")
            if ran > 1:
                found.append(f"{step} ran the call {ran} times on one approval")
            for tool_call_id, outcome in after.results[len(before.results) :]:
                if outcome == "approved" and tool_call_id != owner:
                    found.append(f"{step} ran {tool_call_id}'s call on {owner}'s card")
    if ran and before.gate_on and not spent:
        found.append(f"{step} ran the call with the gate on and no approval spent")
    if fold_ran and not spent:
        found.append(f"{step} ran a held call without spending its approval")
    for tool_call_id, gone in (before.entries - after.entries).items():
        if gone > claimed.count(tool_call_id):
            found.append(f"{step} removed {tool_call_id}'s held call")
    for slot in before.slots.keys() & after.slots.keys():
        was, now = before.slots[slot], after.slots[slot]
        # Only a restore's copy of the card, which can never run, may give
        # way, and only to the exact call it stands in for.
        if was != now and not (_restored(was) and not _restored(now)):
            found.append(f"{step} overwrote the held call in {slot}")
    return found


def _restored(slot: str) -> bool:
    """Whether a held call is a restore's copy of its card: it carries no
    cost context, so it never runs."""
    return held.HeldCall.model_validate_json(slot).cost_context is None


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
        # Set by a seeded round: the gate's flag, and the faulty store.
        self.gate: AsyncMock | None = None
        self.database: "FaultyReviews | None" = None
        # The MCP tools a fold replays through, when the tool is one of them.
        self.handlers: dict[str, held.McpHandler] | None = None
        self.review_id = review_store.review_id_for(
            session.session_id, user_id, TOOL, args
        )
        tool.review_id = self.review_id
        self.interleaving = Interleaving(self.review_id)

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
            _FOLDING.set(True)
            async for _status in held.resolve_answered(
                self.user_id,
                self.session,
                self.delivered.append,
                mcp_handlers=self.handlers,
            ):
                pass

        return fold

    def as_mcp_handler(self) -> None:
        """Make the tool an MCP tool outside the registry: a fold replays its
        held call through the handler, whose seam reaches the gate, as
        ``tool_adapter.held_call_handlers`` does."""

        async def handle(args: dict[str, Any]) -> dict[str, Any]:
            result = await self.tool.execute(
                self.user_id, self.session, "replayed", **args
            )
            return {"content": [{"type": "text", "text": str(result.output)}]}

        self.handlers = {TOOL: handle}

    def restorer(self, wakes: int = 1) -> Callable[[], Awaitable[None]]:
        """Answers' wakes, each restoring the held call from its card if gone."""

        async def restore() -> None:
            for wake in range(wakes):
                if wake:
                    await self.interleaving.point("wake")
                rows = await review_db().get_reviews_by_node_exec_ids(
                    [self.review_id], self.user_id
                )
                answered = [
                    r for r in rows.values() if r.status != ReviewStatus.WAITING
                ]
                await held._restore(self.user_id, self.session.session_id, answered)

        return restore

    def flipper(self, flips: int) -> Callable[[], Awaitable[None]]:
        """The gate switched off and on again between other steps."""

        async def flip() -> None:
            assert self.gate is not None
            for _ in range(flips):
                self.gate.return_value = not self.gate.return_value
                await self.interleaving.point("flip")

        return flip

    def faulter(self, faults: list["Deletes"]) -> Callable[[], Awaitable[None]]:
        """The next spend failing, or failing after it committed."""

        async def fault() -> None:
            assert self.database is not None
            for mode in faults:
                self.database.arm(mode)
                await self.interleaving.point("fault")

        return fault

    def ager(self) -> Callable[[], Awaitable[None]]:
        """An hour going by on an approved card."""

        async def age() -> None:
            await PendingHumanReview.prisma().update_many(
                where={"nodeExecId": self.review_id, "status": ReviewStatus.APPROVED},
                data={"reviewedAt": datetime.now(UTC) - timedelta(hours=2)},
            )
            await self.interleaving.point("age")

        return age

    async def make_legacy(self) -> None:
        """Turn the parked call into one held before held calls knew their
        card: its held call keyed by review id alone, its card unstamped."""
        sid = self.session.session_id
        redis = await get_redis_async()
        for field, call in (await held._held(sid)).items():
            if call.review_id != self.review_id:
                continue
            legacy = call.model_copy(update={"generation": None})
            await redis.hdel(held._key(sid), field)
            await redis.hset(
                held._key(sid),
                self.review_id,
                JSONCryptor().encrypt(legacy.model_dump(mode="json")),
            )
        card = await self.interleaving.card()
        assert card is not None
        payload = dict(card.payload) if isinstance(card.payload, dict) else {}
        payload.pop(review_store.HELD_GENERATION, None)
        await PendingHumanReview.prisma().update(
            where={"nodeExecId": self.review_id}, data={"payload": SafeJson(payload)}
        )

    async def seen(self) -> Seen:
        card = await self.interleaving.card()
        calls = await held._held(self.session.session_id)
        mine = {k: c for k, c in calls.items() if c.review_id == self.review_id}
        return Seen(
            card=_card_seen(card),
            entries=Counter(c.tool_call_id for c in mine.values()),
            slots={k: c.model_dump_json() for k, c in mine.items()},
            gate_on=bool(self.gate.return_value) if self.gate else True,
            faulted=list(self.database.faulted) if self.database else [],
            runs=len(self.tool.runs),
            fold_runs=self.tool.fold_runs,
            results=[
                (
                    m.metadata["held_call"]["tool_call_id"],
                    m.metadata["held_call"]["outcome"],
                )
                for m in self.delivered
            ],
        )

    async def settle(self, propose_again: bool = False) -> None:
        """Later turns, one at a time, fold whatever is still answered; each
        is judged as a step is. ``propose_again`` then makes the call once
        more, as the chat is told to when a spend could not be confirmed, so
        an approval that still stands is used."""
        claim = held._claim
        for turn in range(3):
            claimed: list[str] = []

            async def claims(session_id: str, entry: held.HeldCall) -> bool:
                claimed.append(entry.tool_call_id)
                return await claim(session_id, entry)

            before = await self.seen()
            folding = _FOLDING.set(True)
            try:
                with patch.object(held, "_claim", claims):
                    async for _status in held.resolve_answered(
                        self.user_id,
                        self.session,
                        self.delivered.append,
                        mcp_handlers=self.handlers,
                    ):
                        pass
            finally:
                _FOLDING.reset(folding)
            after = await self.seen()
            self.interleaving.lost += _judged(
                f"later turn {turn + 1}", before, after, claimed
            )
        card = await self.interleaving.card()
        if propose_again and card is not None and card.status == ReviewStatus.APPROVED:
            before = await self.seen()
            await self.caller(9)()
            after = await self.seen()
            self.interleaving.lost += _judged("proposed again", before, after, [])

    def user(self, verdicts: list[ReviewStatus]) -> Callable[[], Awaitable[None]]:
        """The user's answers, each followed by its wake, which restores the
        held call from the card as answered if it is gone by then (a fold may
        have claimed it in between, as a running turn's end can)."""

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
                rows = await review_db().get_reviews_by_node_exec_ids(
                    [self.review_id], self.user_id
                )
                await self.interleaving.point("answer")
                if answered:
                    await held._restore(
                        self.user_id, self.session.session_id, rows.values()
                    )

        return answer

    async def park(self, tool_call_id: str = "call-0") -> Decision:
        async def in_a_turn():
            self.enter_turn()
            return await check_action(
                TOOL, self.args, self.user_id, self.session, tool_call_id=tool_call_id
            )

        decision = await asyncio.create_task(in_a_turn())
        assert decision.review_id == self.review_id
        return decision

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
        claim, listed = held._claim, held.answered
        remember, rehold = held.remember, held._rehold
        stack = self.stepping()
        recover, restore = held._recover, held._restore
        run, run_handler = held._run, held._run_handler

        async def claims(session_id: str, entry: held.HeldCall) -> bool:
            await interleaving.point("claim")
            interleaving.claimed.append(entry.tool_call_id)
            return await claim(session_id, entry)

        def a_step(fn: Callable[..., Awaitable[bool]]):
            async def write(*args: Any, **kwargs: Any) -> bool:
                await interleaving.point(f"remember({_WRITE.get()})")
                return await fn(*args, **kwargs)

            return write

        async def lists(user_id: str, session_id: str) -> list[held.HeldCall]:
            calls = await listed(user_id, session_id)
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
            ("remember", a_step(remember)),
            ("_rehold", a_step(rehold)),
            ("answered", lists),
            ("_recover", writing("recover", recover)),
            ("_restore", writing("restore", restore)),
            ("_run", failing(run)),
            ("_run_handler", failing(run_handler)),
        ):
            stack.enter_context(patch.object(held, name, fn))
        return stack


async def resolved(user_id: str, session: ChatSession) -> list[PendingMessage]:
    delivered: list[PendingMessage] = []
    async for _status in held.resolve_answered(user_id, session, delivered.append):
        pass
    return delivered


async def new_scene(
    user_id: str, tool: PostStub, args: dict[str, Any] = ARGS, parked: bool = True
) -> Scene:
    session = await upsert_chat_session(ChatSession.new(user_id=user_id, dry_run=False))
    await update_session_autopilot_mode(session.session_id, user_id, "auto")
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    scene = Scene(user_id, reloaded, tool, args)
    if parked:
        await scene.park()
    return scene


class SameMillisecond(datetime):
    """Every park in one instant: the gate's clock, stopped."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 17, 12, 0, 0, 123000, tzinfo=tz)


async def seeded_round(user_id: str, tool: PostStub, seed: int) -> list[str]:
    """One seeded schedule over one review id: identical calls (each a
    replacement once the card before it is spent), two folds, a fold that
    fails before it spends and so recovers, a restore from the card and the
    user's answers. Drawn per seed: a cold start (no call parked first, so
    the first parks race), a call held before held calls knew their card,
    the gate's clock stopped (every park in one instant), a timestamp among
    the arguments, the tool reached as an MCP handler (so folds replay
    through the handler's seam), the gate flipping off and on, spends failing
    (before or after they commit), and an approval lapsing. Then later turns settle
    what is left, proposing the call again if an approval still stands.
    What any step did that no actor may do, and what the settle left."""
    rng = random.Random(seed)
    tool.runs.clear()
    tool.fold_runs = 0
    cold = rng.random() < 0.3
    legacy = not cold and rng.random() < 0.3
    same_instant = rng.random() < 0.3
    stamped = rng.random() < 0.3
    through_a_handler = rng.random() < 0.3
    args = (
        {**ARGS, "when": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)} if stamped else ARGS
    )
    verdicts = [
        rng.choice([ReviewStatus.APPROVED] * 3 + [ReviewStatus.REJECTED])
        for _ in range(3)
    ]
    gate = AsyncMock(return_value=True)
    with ExitStack() as stack:
        stack.enter_context(patch("backend.copilot.gate.is_feature_enabled", gate))
        if same_instant:
            stack.enter_context(patch.object(held, "datetime", SameMillisecond))
        if through_a_handler:
            stack.enter_context(
                patch("backend.copilot.tools.get_tool", return_value=None)
            )
        database = stack.enter_context(faulty_reviews())
        scene = await new_scene(user_id, tool, args, parked=not cold)
        scene.gate, scene.database = gate, database
        if through_a_handler:
            scene.as_mcp_handler()
        if legacy:
            await scene.make_legacy()
        # A cold start or a stopped clock gets a fourth call, so two parks
        # can follow a spent card.
        callers = (1, 2, 3, 4) if cold or same_instant else (1, 2, 3)
        actors = {f"call-{n}": scene.caller(n) for n in callers}
        actors |= {"fold-1": scene.fold(), "fold-2": scene.fold()}
        actors |= {
            "failing-fold": scene.fold(),
            "restorer": scene.restorer(rng.choice([1, 2, 3])),
        }
        actors["user"] = scene.user(verdicts)
        if rng.random() < 0.4:
            actors["flipper"] = scene.flipper(rng.choice([1, 2, 3]))
        if rng.random() < 0.4:
            modes: list[Deletes] = ["fail", "commit_then_fail"]
            actors["faulter"] = scene.faulter([rng.choice(modes) for _ in range(2)])
        if rng.random() < 0.25:
            actors["ager"] = scene.ager()
        with scene.stepping_the_held_store():
            await scene.interleaving.run(
                actors, lambda parked: rng.choice(sorted(parked)), scene.seen
            )
        gate.return_value = True
        database.arm(None)
        await scene.settle(propose_again=True)
    found = list(scene.interleaving.lost)
    card = await scene.interleaving.card()
    if card is not None and card.status == ReviewStatus.APPROVED:
        found.append("an approved call was never run")
    return found


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
        # The card each fault that fired was aimed at (None: a delete by id),
        # and the fault the next delete takes, if armed.
        self.faulted: list[datetime | None] = []
        self._armed: Deletes | None = None
        self._real = real
        self._deletes = deletes
        self._reads_fail_after_delete = reads_fail_after_delete

    def arm(self, mode: Deletes | None) -> None:
        self._armed = mode

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
            ),
            created_at,
        )

    async def delete_review_by_node_exec_id(
        self, node_exec_id: str, user_id: str
    ) -> int:
        return await self._deleting(
            lambda: self._real.delete_review_by_node_exec_id(node_exec_id, user_id),
            None,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def _deleting(
        self, delete: Callable[[], Awaitable[int]], card: datetime | None
    ) -> int:
        mode, self._armed = self._armed or self._deletes, None
        if mode != "ok":
            self.faulted.append(card)
        try:
            if mode == "fail":
                raise ConnectionError("database unreachable")
            deleted = await delete()
            if mode == "commit_then_fail":
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
