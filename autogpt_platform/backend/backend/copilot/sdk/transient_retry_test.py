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
import json
import os
import time
from collections.abc import Callable
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

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamToolInputAvailable,
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
    "gen-failed": 0.02,
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
    streamed_calls = {
        e.toolCallId for e in turn.events if isinstance(e, StreamToolInputAvailable)
    }
    streamed_results = {
        e.toolCallId for e in turn.events if isinstance(e, StreamToolOutputAvailable)
    }
    persisted = turn.persisted_messages()
    persisted_calls = {tc["id"] for m in persisted for tc in m.tool_calls or []}
    persisted_results = {m.tool_call_id for m in persisted if m.role == "tool"}
    assert streamed_calls == persisted_calls == {"toolu_1"}
    assert streamed_results == persisted_results == {"toolu_1"}
    assert (persisted[-1].content or "").startswith(
        (COPILOT_ERROR_PREFIX, COPILOT_RETRYABLE_ERROR_PREFIX)
    )


# A step of a scripted launch: a message the CLI streams, or something done
# at that point (a subagent writing its transcript, a tool doing its work).
_Step = AssistantMessage | UserMessage | ResultMessage | Callable[[], None]


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

    async def run(self, launches: list[list[_Step]]) -> None:
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
            # A resumed turn diffs the user's skills, and a first turn loads
            # the user's context, both over DB and scheduler RPCs.
            for name, value in (
                ("build_skills_update_notice", None),
                ("build_session_context", ""),
                ("inject_user_context", None),
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
            async for event in sdk_service.stream_chat_completion_sdk(
                session_id=_SESSION_ID,
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=self.session,
            ):
                self.events.append(event)
            # The reconcile charges from a task the turn leaves behind.
            await asyncio.gather(*_reconcile_tasks())

    def errors(self) -> list[StreamError]:
        return [e for e in self.events if isinstance(e, StreamError)]

    def reconciled_cost(self) -> float:
        self._mocks["persist"].assert_not_awaited()
        reconcile = self._mocks["reconcile"]
        reconcile.assert_awaited_once()
        return reconcile.await_args.kwargs["cost_usd"]

    def persisted_messages(self) -> list[ChatMessage]:
        """The chat as the turn's last save wrote it."""
        upsert = self._mocks[f"{_SVC}.upsert_chat_session"]
        return list(upsert.await_args.args[0].messages)

    def _launch(self, *args: Any, **kwargs: Any) -> AsyncMock:
        options = kwargs["options"]
        script = self._launches[len(self.launched)]
        self.launched.append(options)
        refusal = _start_session(options)

        async def receive():
            if refusal is not None:
                raise refusal
            for step in script:
                if callable(step):
                    step()
                else:
                    yield step

        client = MagicMock()
        client.query = AsyncMock()
        client.receive_response = receive
        context = AsyncMock()
        context.__aenter__.return_value = client
        context.__aexit__.return_value = None
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


def _sends_email(tool_use_id: str, sent: list[str]) -> list[_Step]:
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
