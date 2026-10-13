"""A turn the platform relaunches after a transient provider failure.

The relaunch hides a 5xx or an overload the user never saw, so it must leave
no trace of the failed launch: no charge for what it ran, on the live stream
or in a subagent's file, and no session id the CLI would refuse to start
again. A launch that already streamed a tool call cannot be hidden: the call
ran and the user watched it, and a relaunch would resend the query that made
it. That turn ends in the error instead, with the call kept.

Each turn below runs ``stream_chat_completion_sdk`` for real against a
scripted CLI. On an OpenRouter turn the cost reconcile runs for real too,
against a stand-in for OpenRouter's ``/generation`` endpoint.
"""

import asyncio
import contextlib
import inspect
import json
import os
import random
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from pydantic import BaseModel

from backend.copilot.constants import (
    COMPACTION_TOOL_NAME,
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamStatus,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.sdk import service as sdk_service
from backend.copilot.transcript import cli_session_path

from .conftest import build_test_transcript
from .retry_scenarios_test import _make_sdk_patches
from .turn_cost_test import _SDK_CWD, _SESSION_ID, _SVC, _result

_MODEL = "anthropic/claude-sonnet-4-6"
# What OpenRouter billed for each generation, by ID.
_COSTS = {
    "gen-overflow": 0.011,
    "gen-overflow-late": 0.007,
    "gen-failed": 0.02,
    "gen-failed-late": 0.009,
    "gen-failed-compact": 0.004,
    "gen-ok": 0.03,
    "gen-ok-compact": 0.005,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overflowed_first",
    [
        pytest.param(False, id="first-launch"),
        # A context retry keeps the overflowing launch billed, as the
        # CLI-priced path does; only the launch the 529 ends is absorbed.
        pytest.param(True, id="after-a-context-retry"),
    ],
)
async def test_a_relaunch_bills_nothing_the_failed_launch_ran(
    overflowed_first, tmp_path, monkeypatch
):
    """The failed launch ran a generation OpenRouter billed, then its next
    call was refused with a 529. Only the launch that succeeded is charged."""
    turn = _Turn(tmp_path, monkeypatch)
    overflow = [[_generation("gen-overflow"), _prompt_too_long()]]
    launches = [
        *(overflow if overflowed_first else []),
        [_generation("gen-failed"), _overloaded()],
        _reply("gen-ok"),
    ]

    await turn.run(launches)

    assert len(turn.launched) == len(launches)
    assert not turn.errors()
    billed = ["gen-overflow", "gen-ok"] if overflowed_first else ["gen-ok"]
    assert turn.reconciled_cost() == pytest.approx(sum(_COSTS[g] for g in billed))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overflowed_first",
    [
        pytest.param(False, id="failed-launch"),
        pytest.param(True, id="context-launch-then-failed-launch"),
    ],
)
async def test_a_subagent_writing_as_its_cli_exits_is_billed_with_its_own_launch(
    overflowed_first, tmp_path, monkeypatch
):
    """A subagent can write its last generation while its CLI shuts down.

    That write is still its launch's. The retry absorbs the failed launch
    only after the failed CLI has exited, so its late write is absorbed with
    it. The next launch marks what it inherits only after the launch before
    it has exited, so a context launch's late write stays billed. Reorder
    either step and one of these writes lands with the wrong launch.
    """
    turn = _Turn(tmp_path, monkeypatch)
    overflow = [
        _generation("gen-overflow"),
        _prompt_too_long(),
        _AtExit(run=turn.subagent("agent-overflow", "gen-overflow-late")),
    ]
    failed = [
        _generation("gen-failed"),
        _overloaded(),
        _AtExit(run=turn.subagent("agent-failed", "gen-failed-late")),
    ]
    launches = [*([overflow] if overflowed_first else []), failed, _reply("gen-ok")]

    await turn.run(launches)

    assert len(turn.launched) == len(launches)
    assert not turn.errors()
    billed = ["gen-ok"]
    if overflowed_first:
        billed += ["gen-overflow", "gen-overflow-late"]
    assert turn.reconciled_cost() == pytest.approx(sum(_COSTS[g] for g in billed))


@pytest.mark.asyncio
async def test_a_turn_left_during_the_backoff_bills_nothing_of_the_failed_launch(
    tmp_path, monkeypatch
):
    """The client goes away on the "retrying" notice, before the relaunch.

    The failed launch was ours to absorb the moment the retry was decided, so
    the turn, which ends there, is charged nothing for it.
    """
    turn = _Turn(tmp_path, monkeypatch)

    await turn.run(
        [[_generation("gen-failed"), _overloaded()], _reply("gen-ok")],
        leave_when=lambda event: isinstance(event, StreamStatus)
        and "retrying" in event.message,
    )

    assert len(turn.launched) == 1
    turn.reconcile_mock().assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failure_before_any_output_is_retried_and_billed_once(
    tmp_path, monkeypatch
):
    """The CLI compacted the session, then the query's own call was refused.

    Nothing reached the user, so the turn is relaunched, and the
    compaction's subagent generation goes with the failed launch: the turn
    is charged the launch that succeeded, its own compaction included.
    """
    turn = _Turn(tmp_path, monkeypatch)

    await turn.run(
        [
            [turn.subagent("agent-acompact-1", "gen-failed-compact"), _overloaded()],
            [turn.subagent("agent-acompact-2", "gen-ok-compact"), *_reply("gen-ok")],
        ]
    )

    assert len(turn.launched) == 2
    assert not turn.errors()
    assert [m.content for m in turn.persisted_messages()[3:]] == ["ok"]
    assert turn.reconciled_cost() == pytest.approx(
        _COSTS["gen-ok"] + _COSTS["gen-ok-compact"]
    )


@pytest.mark.asyncio
async def test_a_first_turn_failure_before_any_output_is_retried(tmp_path, monkeypatch):
    """A chat's first turn starts a fresh CLI session, and the CLI writes its
    file before the first call. Relaunched under the same session id the CLI
    refuses to start ("Session ID ... is already in use"), so the retry must
    start the session over instead of failing the turn."""
    turn = _Turn(tmp_path, monkeypatch, first_turn=True)

    await turn.run([[_overloaded()], _reply("gen-ok")])

    assert [options.resume for options in turn.launched] == [None, None]
    assert not turn.errors()
    assert [m.content for m in turn.persisted_messages()[1:]] == ["ok"]
    assert turn.reconciled_cost() == pytest.approx(_COSTS["gen-ok"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "code"),
    [
        pytest.param("overloaded", "transient_api_error", id="overloaded"),
        pytest.param("dropped", "sdk_stream_error", id="connection-dropped"),
    ],
)
async def test_a_launch_that_ran_a_tool_is_not_relaunched(
    failure, code, tmp_path, monkeypatch
):
    """The model sent an email, then its next call failed transiently.

    A relaunch would resend the query, and a model that sends the email
    again sends it twice; it would also roll the call the user watched out
    of the chat. The turn ends in the error, and the chat keeps what ran.
    """
    turn = _Turn(tmp_path, monkeypatch)
    sent: list[str] = []
    fail = _overloaded() if failure == "overloaded" else _dropped

    await turn.run(
        [
            [*_sends_email("toolu_1", sent), fail],
            # Relaunched, the model repeats the call: a second email.
            [*_sends_email("toolu_2", sent), *_reply(None)],
        ]
    )

    assert len(turn.launched) == 1
    assert sent == ["toolu_1"]
    assert [error.code for error in turn.errors()] == [code]
    assert turn.streamed_tool_rows() == turn.persisted_tool_rows()
    assert turn.streamed_tool_rows() == ({"toolu_1"}, {"toolu_1"})
    persisted = turn.persisted_messages()
    assert (persisted[-1].content or "").startswith(
        (COPILOT_ERROR_PREFIX, COPILOT_RETRYABLE_ERROR_PREFIX)
    )


@pytest.mark.asyncio
async def test_random_turns_relaunch_only_what_streamed_nothing(tmp_path, monkeypatch):
    """Seeded random turns: every launch does some unseen work (a generation,
    a subagent), maybe shows text or runs a tool, then succeeds, overflows
    the context, or fails transiently with a 529 or a dropped connection.

    However the steps fall, the loop relaunches exactly when the retry rules
    allow it and nothing was streamed, so no tool runs twice and the chat keeps
    every row it showed; and OpenRouter bills exactly the launches no transient
    retry absorbed.
    """
    for seed in range(_SEEDS):
        rng = random.Random(seed)
        root = tmp_path / str(seed)
        root.mkdir()
        turn = _Turn(root, monkeypatch, first_turn=rng.random() < 0.3)
        sent: list[str] = []
        plans = [
            _random_launch(rng, f"{seed}-{n}", turn, sent, monkeypatch)
            for n in range(_MAX_LAUNCHES)
        ]

        await turn.run([plan.steps for plan in plans])

        ran = plans[: len(turn.launched)]
        context = f"seed {seed}: {[(p.shown, p.end) for p in ran]}"
        assert len(ran) == _launches_allowed(plans), context
        assert sent == [p.tool for p in ran if p.tool], context
        assert turn.streamed_tool_rows() == turn.persisted_tool_rows(), context
        kept = [p for i, p in enumerate(ran) if i == len(ran) - 1 or not p.transient]
        billed = [g for p in kept for g in p.live + p.subagent]
        reconcile = turn.reconcile_mock()
        if any(p.live for p in kept):
            reconcile.assert_awaited_once()
            assert reconcile.await_args.kwargs["cost_usd"] == pytest.approx(
                sum(_COSTS[g] for g in billed)
            ), context
        else:
            reconcile.assert_not_awaited()


class _AtExit(BaseModel):
    """A step the CLI runs while it shuts down, whatever came before it in
    the script: a subagent writing its last generation as the process exits."""

    run: Callable[[], None]


# A step of a scripted launch: a message the CLI streams, something done at
# that point (a subagent writing its transcript, a tool doing its work, an
# awaited hook), or something done as the CLI exits.
_Step = (
    AssistantMessage
    | UserMessage
    | ResultMessage
    | Callable[[], None]
    | Callable[[], Awaitable[None]]
    | _AtExit
)

_SEEDS = 200
# Three context attempts, each with one transient retry.
_MAX_LAUNCHES = 6
_TRANSIENT_ENDS = ("overloaded", "dropped")


class _Launch(BaseModel):
    """One scripted launch of a random turn, and what it does."""

    steps: list[Any] = []
    live: list[str] = []
    subagent: list[str] = []
    shown: str = "nothing"
    tool: str | None = None
    end: str = "ok"

    @property
    def streamed(self) -> bool:
        return self.shown != "nothing" or self.end == "ok"

    @property
    def transient(self) -> bool:
        return self.end in _TRANSIENT_ENDS


def _random_launch(
    rng: random.Random,
    key: str,
    turn: "_Turn",
    sent: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> _Launch:
    launch = _Launch(shown=rng.choice(["nothing", "text", "tool"]))
    for n in range(rng.randint(0, 2)):
        generation = f"gen-{key}-{n}"
        monkeypatch.setitem(_COSTS, generation, rng.randint(1, 99) / 1000)
        if rng.random() < 0.5:
            launch.steps.append(_generation(generation))
            launch.live.append(generation)
        else:
            launch.steps.append(turn.subagent(f"agent-{key}-{n}", generation))
            launch.subagent.append(generation)
    shown = f"gen-{key}-shown"
    monkeypatch.setitem(_COSTS, shown, rng.randint(1, 99) / 1000)
    if launch.shown == "text":
        launch.steps.append(
            AssistantMessage(
                content=[TextBlock(text="partial")], model=_MODEL, message_id=shown
            )
        )
        launch.live.append(shown)
    elif launch.shown == "tool":
        launch.tool = f"toolu_{key}"
        launch.steps += _sends_email(launch.tool, sent, generation_id=shown)
        launch.live.append(shown)
    launch.end = rng.choice(["ok", "overflow", *_TRANSIENT_ENDS])
    if launch.end == "ok":
        reply = f"gen-{key}-reply"
        monkeypatch.setitem(_COSTS, reply, rng.randint(1, 99) / 1000)
        launch.steps += _reply(reply)
        launch.live.append(reply)
    elif launch.end == "overflow":
        launch.steps.append(_prompt_too_long())
    elif launch.end == "overloaded":
        launch.steps.append(_overloaded())
    else:
        launch.steps.append(_dropped)
    return launch


def _launches_allowed(plans: list[_Launch]) -> int:
    """How many launches the retry rules make of *plans*: a launch is retried
    only if it streamed nothing, a transient failure once per context
    attempt, a context overflow until the third attempt."""
    attempt = transient = 0
    for n, plan in enumerate(plans, start=1):
        if plan.streamed:
            return n
        if plan.end == "overflow":
            attempt, transient = attempt + 1, 0
            if attempt >= 3:
                return n
        else:
            transient += 1
            if transient > 1:
                return n
    return len(plans)


class _Turn:
    """A user turn of a chat: its second by default, so the CLI resumes its
    session; on ``first_turn`` the CLI starts one."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        first_turn: bool = False,
    ) -> None:
        # The CLI keeps its sessions, and its subagents their transcripts,
        # under the home dir; the reconcile reads them back from there.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
        session_file = Path(cli_session_path(_SDK_CWD, _SESSION_ID))
        self._subagents = session_file.parent / _SESSION_ID / "subagents"
        self._subagents.mkdir(parents=True)
        self.launched: list[Any] = []
        self.events: list[StreamBaseResponse] = []
        prior = [
            ChatMessage(role="user", content="prior question"),
            ChatMessage(role="assistant", content="prior answer"),
        ]
        self.session = ChatSession(
            session_id=_SESSION_ID,
            user_id="test-user",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            messages=[
                *([] if first_turn else prior),
                ChatMessage(role="user", content="hello"),
            ],
            # A titled chat skips title generation, a model call of its own.
            title="Retries",
        )
        self._launches: list[list[_Step]] = []
        self._mocks: dict[str, MagicMock] = {}
        self._compaction_opened = asyncio.Event()

    def precompact(self) -> Callable[[], Awaitable[None]]:
        """A step: the CLI starts compacting. Its PreCompact hook fires, and
        the CLI goes on once the loop has opened the compaction row."""

        async def start() -> None:
            hooks = self._mocks[f"{_SVC}.create_security_hooks"]
            hooks.call_args.kwargs["on_compact"](str(self._subagents.parent))
            await asyncio.wait_for(self._compaction_opened.wait(), timeout=10)

        return start

    def subagent(self, agent: str, generation_id: str) -> Callable[[], None]:
        """A step: a subagent of the CLI's writes its transcript."""

        def write() -> None:
            path = self._subagents / f"{agent}.jsonl"
            entry = {"type": "assistant", "message": {"id": generation_id}}
            path.write_text(json.dumps(entry) + "\n")
            # During the turn, whatever the filesystem's clock says.
            now = time.time()
            os.utime(path, (now, now))

        return write

    async def run(
        self,
        launches: list[list[_Step]],
        *,
        leave_when: Callable[[StreamBaseResponse], bool] | None = None,
    ) -> None:
        """Play the turn; ``leave_when`` closes the stream at the first event
        it matches, the way a client that goes away does."""
        self._launches = launches
        patches = _make_sdk_patches(
            self.session,
            original_transcript=build_test_transcript(
                [("user", "prior question"), ("assistant", "prior answer")]
            ),
            compacted_transcript=build_test_transcript(
                [("user", "summary"), ("assistant", "summarized answer")]
            ),
            client_side_effect=self._launch,
        )
        for target, kwargs in patches:
            if target == f"{_SVC}.config":
                kwargs.update(
                    openrouter_active=True,
                    sdk_reconcile_openrouter_cost=True,
                    thinking_standard_model=_MODEL,
                )
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                self._mocks[target] = stack.enter_context(patch(target, **kwargs))
            # A resumed turn diffs the user's skills, a first turn loads the
            # user's context, and every turn folds in answered cards and
            # follow-ups: DB, scheduler and Redis round trips, none of them
            # under test.
            for name, value in (
                ("build_skills_update_notice", None),
                ("build_session_context", ""),
                ("inject_user_context", None),
                ("resolve_answered", []),
                ("drain_pending_for_persist", []),
            ):
                stack.enter_context(
                    patch(f"{_SVC}.{name}", new_callable=AsyncMock, return_value=value)
                )
            stack.enter_context(
                patch(f"{_SVC}._compute_transient_backoff", return_value=0)
            )
            self._mocks["persist"] = stack.enter_context(
                patch(f"{_SVC}.persist_and_record_usage", new_callable=AsyncMock)
            )
            self._mocks["reconcile"] = stack.enter_context(
                patch(
                    "backend.copilot.sdk.openrouter_cost.persist_and_record_usage",
                    new_callable=AsyncMock,
                )
            )
            stack.enter_context(patch("httpx.AsyncClient.get", new=_generation_lookup))
            stream = sdk_service.stream_chat_completion_sdk(
                session_id=_SESSION_ID,
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=self.session,
            )
            async for event in stream:
                self.events.append(event)
                if (
                    isinstance(event, StreamToolInputStart)
                    and event.toolName == COMPACTION_TOOL_NAME
                ):
                    self._compaction_opened.set()
                if leave_when is not None and leave_when(event):
                    break
            await stream.aclose()
            # The reconcile charges from a task the turn leaves behind.
            await asyncio.gather(*_reconcile_tasks())

    def errors(self) -> list[StreamError]:
        return [e for e in self.events if isinstance(e, StreamError)]

    def reconcile_mock(self) -> MagicMock:
        """What the OpenRouter reconcile charged, if it ran."""
        return self._mocks["reconcile"]

    def reconciled_cost(self) -> float:
        self._mocks["persist"].assert_not_awaited()
        reconcile = self._mocks["reconcile"]
        reconcile.assert_awaited_once()
        return reconcile.await_args.kwargs["cost_usd"]

    def persisted_messages(self) -> list[ChatMessage]:
        """The chat as the turn's last save wrote it."""
        upsert = self._mocks[f"{_SVC}.upsert_chat_session"]
        return list(upsert.await_args.args[0].messages)

    def streamed_tool_rows(self) -> tuple[set[str], set[str]]:
        """The model's tool calls and their results the turn streamed, by ID.

        Compaction rows are left out: one the loop opens and then finds
        nothing to compact for is retired on the wire, never persisted.
        """
        calls = {
            e.toolCallId
            for e in self.events
            if isinstance(e, StreamToolInputAvailable)
            and e.toolName != COMPACTION_TOOL_NAME
        }
        results = {
            e.toolCallId
            for e in self.events
            if isinstance(e, StreamToolOutputAvailable)
            and e.toolName != COMPACTION_TOOL_NAME
        }
        return calls, results

    def persisted_tool_rows(self) -> tuple[set[str], set[str]]:
        """The model's tool calls and their results the chat kept, by ID."""
        messages = self.persisted_messages()
        tool_calls = [tc for m in messages for tc in m.tool_calls or []]
        compactions = {
            tc["id"]
            for tc in tool_calls
            if tc["function"]["name"] == COMPACTION_TOOL_NAME
        }
        calls = {tc["id"] for tc in tool_calls} - compactions
        results = {
            m.tool_call_id
            for m in messages
            if m.role == "tool" and m.tool_call_id and m.tool_call_id not in compactions
        }
        return calls, results

    def _launch(self, *args: Any, **kwargs: Any) -> AsyncMock:
        options = kwargs["options"]
        script = self._launches[len(self.launched)]
        self.launched.append(options)
        refusal = _start_session(options)
        at_exit = [step.run for step in script if isinstance(step, _AtExit)]

        async def receive():
            if refusal is not None:
                raise refusal
            for step in script:
                if isinstance(step, _AtExit):
                    continue
                if callable(step):
                    if inspect.isawaitable(done := step()):
                        await done
                else:
                    yield step

        async def shut_down(*_: Any) -> None:
            for run in at_exit:
                run()

        client = MagicMock()
        client.query = AsyncMock()
        client.receive_response = receive
        context = AsyncMock()
        context.__aenter__.return_value = client
        context.__aexit__.side_effect = shut_down
        return context


def _start_session(options: Any) -> Exception | None:
    """What the CLI does with ``--session-id``: write the session's file before
    its first call, and refuse an id whose file is already there."""
    if options.resume is not None or not options.session_id:
        return None
    path = Path(cli_session_path(_SDK_CWD, options.session_id))
    if path.exists():
        return RuntimeError(
            "Command failed with exit code 1: "
            f"Error: Session ID {options.session_id} is already in use."
        )
    path.write_text("")
    return None


def _overloaded() -> AssistantMessage:
    """How the bundled CLI (2.1.284) reports a 529 once its retries are spent."""
    return AssistantMessage(
        content=[
            TextBlock(
                text="API Error: 529 Overloaded. This is a server-side issue, "
                "usually temporary — try again in a moment."
            )
        ],
        model=_MODEL,
        error="server_error",
    )


def _dropped() -> None:
    raise ConnectionError("ECONNRESET: connection reset by peer")


def _generation(generation_id: str) -> AssistantMessage:
    """A billed generation that streamed nothing the user could see."""
    return AssistantMessage(content=[], model=_MODEL, message_id=generation_id)


def _reply(generation_id: str | None) -> list[_Step]:
    return [
        AssistantMessage(
            content=[TextBlock(text="ok")], model=_MODEL, message_id=generation_id
        ),
        _result(0.03),
    ]


def _prompt_too_long() -> ResultMessage:
    result = _result(0.011)
    result.subtype = "error_during_execution"
    result.is_error = True
    result.result = "Prompt is too long"
    return result


def _sends_email(
    tool_use_id: str, sent: list[str], generation_id: str | None = None
) -> list[_Step]:
    return [
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id=tool_use_id,
                    name="send_email",
                    input={"to": "boss@example.com", "subject": "Report"},
                )
            ],
            model=_MODEL,
            message_id=generation_id,
        ),
        lambda: sent.append(tool_use_id),
        UserMessage(content=[ToolResultBlock(tool_use_id=tool_use_id, content="sent")]),
    ]


async def _generation_lookup(
    _client: httpx.AsyncClient, _url: str, *, params: dict[str, str], **_: Any
) -> httpx.Response:
    cost = _COSTS.get(params["id"])
    if cost is None:
        # Permanent, so the reconcile does not back off and retry it.
        return httpx.Response(403)
    return httpx.Response(200, json={"data": {"total_cost": cost}})


def _reconcile_tasks() -> list[asyncio.Task]:
    loop = asyncio.get_running_loop()
    return [
        task
        for task in sdk_service._background_tasks
        if task.get_loop() is loop and not task.done()
    ]
