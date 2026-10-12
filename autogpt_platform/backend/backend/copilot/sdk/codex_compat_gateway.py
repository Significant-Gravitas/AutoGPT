from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager, suppress
from typing import Protocol, cast
from uuid import uuid4

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field

from backend.copilot.provider_failure import ProviderFailure
from backend.copilot.provider_failure import classify as classify_provider_failure
from backend.copilot.provider_failure import status_for
from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexReasoningEffort,
    CodexStreamEvent,
    CodexTokenUsage,
)
from backend.integrations.codex.transport import (
    CodexAgentSession,
    CodexTransport,
    get_codex_transport,
)
from backend.integrations.credential_lease import CredentialLease

_MAX_REQUEST_BYTES = 16 * 1024 * 1024
# Bounds on the per-conversation replay cache. A retry arrives seconds to
# minutes after the original, so no entry has to outlive its own turn: over 30
# days of Dev logs the longest Codex turn ran 1,406 s and resolved 67 tool calls
# (2026-09-18), one boundary each. An entry holds only the response body -- a
# tool_use boundary or the final text -- never the 0.97M-5.8M token request that
# provoked the retry, and the byte cap is half the request limit above.
_MAX_REPLAY_ENTRIES = 128
_MAX_REPLAY_BYTES = 8 * 1024 * 1024
_REPLAY_TTL_SECONDS = 1800.0
_MAX_LOGGED_ERROR_MESSAGE_CHARS = 240
_REDACTED = "[REDACTED]"
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_JWT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"
    r"\.[A-Za-z0-9_-]{6,}(?![A-Za-z0-9_-])"
)
_DEVICE_CODE_PATTERN = re.compile(r"\b[A-Z0-9]{4}(?:-[A-Z0-9]{4}){1,3}\b")
_PROVIDER_STATE_PATTERN = re.compile(r"(?is)[\"']?provider[_ -]?state[\"']?\s*[:=].*$")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?ix)"
    r"(?P<label>\b(?:access[_ -]?token|refresh[_ -]?token|id[_ -]?token|"
    r"device[_ -]?code|user[_ -]?code|authorization|anthropic[_ -]?auth[_ -]?"
    r"token|api[_ -]?key)\b)"
    r"\s*[\"']?\s*[:=]\s*(?:Bearer\s+)?"
    r"(?:\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^\s,;]+)"
)

logger = logging.getLogger(__name__)


class _AgentTransport(Protocol):
    def agent_session(
        self,
        lease: CredentialLease,
    ) -> AbstractAsyncContextManager[CodexAgentSession]: ...


class _AgentSession(Protocol):
    async def invoke(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[CodexStreamEvent], Awaitable[None]] | None = None,
    ) -> CodexInvocationResult: ...


class _SseSink(Protocol):
    async def write(self, data: bytes) -> None: ...


class _GatewayState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class _TextDelta(_GatewayState):
    text: str


class _ToolUse(_GatewayState):
    call_id: str
    name: str
    arguments: object


class _Completed(_GatewayState):
    result: CodexInvocationResult


class _Failed(_GatewayState):
    error: BaseException


_ConversationEvent = _TextDelta | _ToolUse | _Completed | _Failed


class _Replay(_GatewayState):
    """A response already delivered, kept to answer a byte-identical retry."""

    status: int
    headers: dict[str, str]
    body: bytes
    streamed: bool
    expires_at: float


class _Conversation(_GatewayState):
    id: str
    queue: asyncio.Queue[_ConversationEvent] = Field(default_factory=asyncio.Queue)
    pending: dict[str, _ToolCallRecord] = Field(default_factory=dict)
    response_lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = None
    result: CodexInvocationResult | None = None
    replays: dict[str, _Replay] = Field(default_factory=dict)


class _Continuation(_GatewayState):
    conversation: _Conversation
    replay_key: str


class _DuplicateSubmission(_GatewayState):
    conversation: _Conversation
    replay_key: str


class _ToolCallRecord(_GatewayState):
    gateway_call_id: str
    raw_call_id: str
    conversation: _Conversation
    future: asyncio.Future[CodexDynamicToolResult]
    result: CodexDynamicToolResult | None = None
    claim_fingerprint: str | None = None
    closed: bool = False


_Conversation.model_rebuild()
_ToolCallRecord.model_rebuild()
_Continuation.model_rebuild()
_DuplicateSubmission.model_rebuild()


class _DuplicateToolResultError(ValueError):
    pass


class CodexAnthropicGateway:
    def __init__(
        self,
        *,
        credential_lease: CredentialLease | None = None,
        agent_session: _AgentSession | None = None,
        model: str,
        effort: CodexReasoningEffort | None = None,
        transport: CodexTransport | _AgentTransport | None = None,
    ) -> None:
        if credential_lease is None and agent_session is None:
            raise ValueError(
                "Codex gateway requires a credential lease or agent session"
            )
        self._credential_lease = credential_lease
        # The last failure this gateway named, kept so the service layer can
        # report *what* went wrong. By the time a failure reaches the CLI it
        # is an HTTP status and a sentence; the typed exception only exists
        # in here.
        self.last_failure: ProviderFailure | None = None
        self.model = model
        self.effort: CodexReasoningEffort | None = effort
        self._transport = cast(
            _AgentTransport,
            transport if transport is not None else get_codex_transport(),
        )
        self._agent_context: AbstractAsyncContextManager[CodexAgentSession] | None = (
            None
        )
        self._agent_session: _AgentSession | None = agent_session
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._base_url: str | None = None
        self._auth_token = secrets.token_urlsafe(32)
        self._conversations: dict[str, _Conversation] = {}
        self._tool_calls: dict[str, _ToolCallRecord] = {}
        self._results: list[CodexInvocationResult] = []
        # Max per-request estimate reported to the CLI this turn. Tracked
        # at the boundary (not derived from _results) so it survives turns
        # whose conversations raised before recording anything.
        self._peak_boundary_estimate: int = 0
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("Codex Anthropic gateway is not running")
        return self._base_url

    @property
    def auth_token(self) -> str:
        return self._auth_token

    @property
    def result(self) -> CodexInvocationResult | None:
        if not self._results:
            return None
        latest = self._results[-1]
        usage = _sum_usage(result.usage for result in self._results)
        durations = [
            result.duration_ms
            for result in self._results
            if result.duration_ms is not None
        ]
        return latest.model_copy(
            update={
                "duration_ms": sum(durations) if durations else None,
                "usage": usage,
            }
        )

    @property
    def results(self) -> tuple[CodexInvocationResult, ...]:
        return tuple(self._results)

    @property
    def peak_boundary_estimate(self) -> int:
        """Max per-request input estimate reported this turn (0 = none yet)."""
        return self._peak_boundary_estimate

    def _record_boundary_estimate(self, estimate: int) -> None:
        """Record one reported boundary estimate and its running peak.

        Logged per request: this series is the CLI's trigger input on the
        Codex route, and the peak is the turn summary that survives
        conversations which raise before recording a result.
        """
        self._peak_boundary_estimate = max(self._peak_boundary_estimate, estimate)
        logger.info(
            f"codex boundary: estimate={estimate} "
            f"peak={self._peak_boundary_estimate} model={self.model}"
        )

    async def start(self) -> None:
        await self.__aenter__()

    async def __aenter__(self) -> "CodexAnthropicGateway":
        if self._runner is not None:
            return self
        if self._agent_session is None:
            credential_lease = self._credential_lease
            if credential_lease is None:
                raise RuntimeError("Codex gateway credential lease is unavailable")
            self._agent_context = self._transport.agent_session(credential_lease)
            self._agent_session = await self._agent_context.__aenter__()
        try:
            application = web.Application(client_max_size=_MAX_REQUEST_BYTES)
            application.router.add_post("/v1/messages", self._handle_messages)
            application.router.add_post(
                "/v1/messages/count_tokens",
                self._handle_count_tokens,
            )
            application.router.add_get("/healthz", self._handle_health)
            runner = web.AppRunner(
                application,
                access_log=None,
                shutdown_timeout=2,
            )
            self._runner = runner
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            self._site = site
            await site.start()
            server = cast(asyncio.Server | None, site._server)
            if server is None or not server.sockets:
                raise RuntimeError("Codex Anthropic gateway failed to bind")
            port = int(server.sockets[0].getsockname()[1])
            self._base_url = f"http://127.0.0.1:{port}"
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        await self.close()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            runner, self._runner = self._runner, None
            self._site = None
            self._base_url = None
            for record in tuple(self._tool_calls.values()):
                if not record.future.done():
                    record.future.cancel()
            self._tool_calls.clear()
            for conversation in self._conversations.values():
                conversation.queue.put_nowait(
                    _Failed(error=RuntimeError("Codex Anthropic gateway is closing"))
                )
            tasks = [
                conversation.task
                for conversation in self._conversations.values()
                if conversation.task is not None and not conversation.task.done()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                with suppress(BaseException):
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=5,
                    )

            cleanup_error: BaseException | None = None
            if runner is not None:
                try:
                    await asyncio.wait_for(runner.cleanup(), timeout=5)
                except BaseException as exc:
                    cleanup_error = exc

            agent_context, self._agent_context = self._agent_context, None
            self._agent_session = None
            if agent_context is not None:
                try:
                    await agent_context.__aexit__(None, None, None)
                except BaseException as exc:
                    cleanup_error = exc
            if cleanup_error is not None:
                raise cleanup_error

    async def _handle_health(self, request: web.Request) -> web.Response:
        if not self._is_authorized(request):
            return _anthropic_error(401, "authentication_error", "Unauthorized")
        return web.json_response({"status": "ok"})

    async def _handle_count_tokens(self, request: web.Request) -> web.Response:
        if not self._is_authorized(request):
            return _anthropic_error(401, "authentication_error", "Unauthorized")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _anthropic_error(
                400,
                "invalid_request_error",
                "Request body must be JSON",
            )
        return web.json_response({"input_tokens": _estimate_input_tokens(payload)})

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        if not self._is_authorized(request):
            return _anthropic_error(401, "authentication_error", "Unauthorized")
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _anthropic_error(
                400,
                "invalid_request_error",
                "Request body must be JSON",
            )
        if not isinstance(payload, dict):
            return _anthropic_error(
                400,
                "invalid_request_error",
                "Request body must be an object",
            )

        try:
            continuation = self._continue_conversation(payload)
        except _DuplicateToolResultError as exc:
            return _anthropic_error(
                409,
                "invalid_request_error",
                str(exc),
            )
        except (TypeError, ValueError) as exc:
            return _anthropic_error(
                400,
                "invalid_request_error",
                str(exc),
            )

        streamed = payload.get("stream") is True
        if isinstance(continuation, _DuplicateSubmission):
            return await self._duplicate_response(request, continuation, streamed)
        if continuation is None:
            conversation, replay_key = self._start_conversation(payload), None
        else:
            conversation = continuation.conversation
            replay_key = continuation.replay_key

        input_tokens = _estimate_input_tokens(payload)
        self._record_boundary_estimate(input_tokens)
        if streamed:
            return await self._streaming_response(
                request,
                conversation,
                input_tokens,
                replay_key,
            )
        return await self._nonstreaming_response(
            conversation,
            input_tokens,
            replay_key,
        )

    async def _duplicate_response(
        self,
        request: web.Request,
        duplicate: _DuplicateSubmission,
        streamed: bool,
    ) -> web.StreamResponse:
        # The lock is held while the accepted request is still writing, so
        # waiting on it is what gives a concurrent duplicate the same answer.
        async with duplicate.conversation.response_lock:
            replay = self._take_replay(
                duplicate.conversation,
                duplicate.replay_key,
                streamed=streamed,
            )
        if replay is None:
            return _anthropic_error(
                409,
                "invalid_request_error",
                "This tool-result request was already accepted",
            )
        if not replay.streamed:
            return web.Response(
                status=replay.status,
                body=replay.body,
                headers=replay.headers,
            )
        response = web.StreamResponse(status=replay.status, headers=replay.headers)
        await response.prepare(request)
        await response.write(replay.body)
        await response.write_eof()
        return response

    def _continue_conversation(
        self,
        payload: dict[str, object],
    ) -> _Continuation | _DuplicateSubmission | None:
        tool_results = _extract_tool_results(payload.get("messages"))
        known = [
            (self._tool_calls[call_id], result)
            for call_id, result in tool_results.items()
            if call_id in self._tool_calls
        ]
        if not known:
            return None

        for record, result in known:
            if record.result is not None and record.result != result:
                raise _DuplicateToolResultError(
                    f"Conflicting result for tool_use_id {record.gateway_call_id!r}"
                )

        if _delivers_results_with_new_ask(payload.get("messages")):
            # A delivery that also says something new is a new ask, and the
            # whole request has to reach the model — a continuation forwards
            # the tool output alone and drops the text.
            #
            # The CLI's auto-compaction request is the case that matters.
            # Captured from claude 2.1.274 mid-turn: the CLI compacts before
            # running the tool it was just handed, drops that unanswered
            # ``tool_use``, and appends the summarisation instruction to the
            # last user message — whose ``tool_result`` was delivered one
            # request earlier and is therefore *settled*.  Every branch below
            # this point either replays or raises for a settled result, so
            # without this check the request is a 409 the CLI retries with
            # backoff, no summary is written, and the turn refires until it
            # dies — the ``after_source=no_summary_line`` cycle in the dev
            # traces, minutes apart per firing.
            #
            # A still-pending result in the same shape has not been observed
            # but is handled the same way, defensively: claiming it would
            # resume the *task* upstream and the model would answer with its
            # next tool call, not a summary.  Cancelling the future unwinds
            # the upstream invoke without another model call — CancelledError
            # escapes the session's tool-error handling, whereas an unanswered
            # future would time out, be reported to the model as a failed
            # tool, and resume the task anyway.  Requests carry
            # ``store: False``, so nothing upstream is lost.  The result stays
            # on the record so a re-sent copy reads as settled rather than
            # conflicting.
            for record, result in known:
                if record.result is None:
                    record.result = result
                record.closed = True
                if not record.future.done():
                    record.future.cancel()
            return None

        claimable = [
            (record, result)
            for record, result in known
            if not record.closed and record.result is None
        ]
        fingerprint = _tool_result_request_fingerprint(payload)
        if not claimable:
            claimed_by = [
                record for record, _ in known if record.claim_fingerprint == fingerprint
            ]
            if claimed_by:
                # The CLI re-sends this verbatim when its first attempt times
                # out, so the response that request produced is the answer.
                return _DuplicateSubmission(
                    conversation=claimed_by[0].conversation,
                    replay_key=fingerprint,
                )
            if any(record.result is None for record, _ in known):
                raise _DuplicateToolResultError(
                    "This tool-result request refers to a closed model call"
                )
            # A pure delivery of results that were already answered, under a
            # new fingerprint: a re-framed replay.  There is no response of
            # its own to serve and no new ask to forward (that case returned
            # above), so one tool result still buys one upstream call.
            raise _DuplicateToolResultError(
                "This tool-result request refers to a completed model call"
            )

        conversation = claimable[0][0].conversation
        if any(record.conversation is not conversation for record, _ in claimable):
            raise ValueError("Tool results span multiple model conversations")

        for record, result in claimable:
            record.result = result
            record.claim_fingerprint = fingerprint
        for record, result in claimable:
            if not record.future.done():
                record.future.set_result(result)
        return _Continuation(conversation=conversation, replay_key=fingerprint)

    def _start_conversation(self, payload: dict[str, object]) -> _Conversation:
        agent_session = self._agent_session
        if agent_session is None:
            raise RuntimeError("Codex Anthropic gateway is not running")
        tools, original_names = _parse_tools(payload.get("tools"))
        conversation = _Conversation(id=uuid4().hex)
        self._conversations[conversation.id] = conversation
        invocation = CodexInvocationRequest(
            prompt=_serialize_messages(payload.get("messages")),
            instructions=_serialize_system(payload.get("system")),
            model=self.model,
            effort=self.effort,
        )
        conversation.task = asyncio.create_task(
            self._run_conversation(
                conversation,
                agent_session,
                invocation,
                tools,
                original_names,
            )
        )
        return conversation

    async def _run_conversation(
        self,
        conversation: _Conversation,
        agent_session: _AgentSession,
        invocation: CodexInvocationRequest,
        tools: list[CodexDynamicToolSpec],
        original_names: dict[str, str],
    ) -> None:
        async def handle_event(event: CodexStreamEvent) -> None:
            if event.type == "text_delta" and event.delta:
                await conversation.queue.put(_TextDelta(text=event.delta))

        async def handle_tool(call: CodexDynamicToolCall) -> CodexDynamicToolResult:
            future: asyncio.Future[CodexDynamicToolResult] = (
                asyncio.get_running_loop().create_future()
            )
            gateway_call_id = f"toolu_codex_{uuid4().hex}"
            record = _ToolCallRecord(
                gateway_call_id=gateway_call_id,
                raw_call_id=call.call_id,
                conversation=conversation,
                future=future,
            )
            conversation.pending[gateway_call_id] = record
            self._tool_calls[gateway_call_id] = record
            await conversation.queue.put(
                _ToolUse(
                    call_id=gateway_call_id,
                    name=original_names.get(call.tool, call.tool),
                    arguments=call.arguments,
                )
            )
            try:
                return await future
            finally:
                record.closed = True
                conversation.pending.pop(gateway_call_id, None)

        try:
            result = await agent_session.invoke(
                invocation,
                tools,
                handle_tool,
                handle_event,
            )
            conversation.result = result
            self._results.append(result)
            await conversation.queue.put(_Completed(result=result))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.error(
                "Codex gateway conversation failed: exception_type=%s error=%s",
                type(exc).__name__,
                _safe_exception_message(exc, secrets_to_redact=(self._auth_token,)),
            )
            await conversation.queue.put(_Failed(error=exc))

    def _failure_response(self, failed: "_Failed") -> web.Response:
        """Answer a failed conversation with a status the CLI can act on.

        Every failure used to become ``502 api_error``, which reads as a
        server fault and invites the CLI to retry something that cannot
        succeed -- an expired credential retried three times is still
        expired. Naming the failure lets the status say "stop asking".

        Unrecognised failures keep the old 502: a wrong specific status
        would be worse than an honest generic one.
        """
        failure = classify_provider_failure(
            failed.error,
            auth_provider="codex",
            credential_id=(
                self._credential_lease.credentials.id
                if self._credential_lease is not None
                else None
            ),
        )
        self.last_failure = failure
        if failure is None:
            return _anthropic_error(502, "api_error", "Codex model transport failed")
        return _anthropic_error(
            status_for(failure),
            failure.kind.value,
            failure.message,
        )

    async def _streaming_response(
        self,
        request: web.Request,
        conversation: _Conversation,
        input_tokens: int,
        replay_key: str | None = None,
    ) -> web.StreamResponse:
        async with conversation.response_lock:
            return await self._streaming_response_locked(
                request,
                conversation,
                input_tokens,
                replay_key,
            )

    async def _streaming_response_locked(
        self,
        request: web.Request,
        conversation: _Conversation,
        input_tokens: int,
        replay_key: str | None,
    ) -> web.StreamResponse:
        first = await conversation.queue.get()
        if isinstance(first, _Failed):
            return self._failure_response(first)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        response = web.StreamResponse(status=200, headers=headers)
        await response.prepare(request)
        sink = _RecordingSink(response)
        message_id = f"msg_codex_{uuid4().hex}"
        try:
            await _write_sse(
                sink,
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 0,
                        },
                    },
                },
            )
            await self._write_boundary(sink, conversation, first)
            await response.write_eof()
        except (ConnectionError, ConnectionResetError, asyncio.CancelledError):
            if conversation.task is not None:
                conversation.task.cancel()
            raise
        body = sink.body
        if body is not None:
            self._record_replay(
                conversation,
                replay_key,
                _Replay(
                    status=200,
                    headers=headers,
                    body=body,
                    streamed=True,
                    expires_at=_now() + _REPLAY_TTL_SECONDS,
                ),
            )
        return response

    async def _write_boundary(
        self,
        sink: _SseSink,
        conversation: _Conversation,
        first: _ConversationEvent,
    ) -> None:
        index = 0
        text_open = False
        emitted_text = False
        event = first
        while True:
            if isinstance(event, _TextDelta):
                emitted_text = True
                if not text_open:
                    await _write_sse(
                        sink,
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    text_open = True
                await _write_sse(
                    sink,
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": event.text},
                    },
                )
            elif isinstance(event, _ToolUse):
                if text_open:
                    await _write_sse(
                        sink,
                        {"type": "content_block_stop", "index": index},
                    )
                    index += 1
                    text_open = False
                await _write_tool_use(sink, index, event)
                await _write_message_end(sink, "tool_use", 0)
                return
            elif isinstance(event, _Completed):
                if not emitted_text and event.result.final_response:
                    if not text_open:
                        await _write_sse(
                            sink,
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                        text_open = True
                    await _write_sse(
                        sink,
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "text_delta",
                                "text": event.result.final_response,
                            },
                        },
                    )
                if text_open:
                    await _write_sse(
                        sink,
                        {"type": "content_block_stop", "index": index},
                    )
                output_tokens = (
                    event.result.usage.output_tokens if event.result.usage else 0
                )
                await _write_message_end(sink, "end_turn", output_tokens)
                return
            elif isinstance(event, _Failed):
                # The HTTP status can no longer change once streaming starts,
                # but classification is still required: the service reads
                # ``last_failure`` after the gateway closes to persist the
                # provider-specific reason on the turn.
                self._failure_response(event)
                failure = self.last_failure
                await _write_sse(
                    sink,
                    {
                        "type": "error",
                        "error": {
                            "type": failure.kind.value if failure else "api_error",
                            "message": (
                                failure.message
                                if failure
                                else "Codex model transport failed"
                            ),
                        },
                    },
                )
                return
            else:
                await _write_sse(
                    sink,
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "Codex model transport failed",
                        },
                    },
                )
                return
            event = await conversation.queue.get()

    async def _nonstreaming_response(
        self,
        conversation: _Conversation,
        input_tokens: int,
        replay_key: str | None = None,
    ) -> web.Response:
        async with conversation.response_lock:
            return await self._nonstreaming_response_locked(
                conversation,
                input_tokens,
                replay_key,
            )

    async def _nonstreaming_response_locked(
        self,
        conversation: _Conversation,
        input_tokens: int,
        replay_key: str | None,
    ) -> web.Response:
        content: list[dict[str, object]] = []
        text_parts: list[str] = []
        output_tokens = 0
        stop_reason = "end_turn"
        while True:
            event = await conversation.queue.get()
            if isinstance(event, _TextDelta):
                text_parts.append(event.text)
                continue
            if text_parts:
                content.append({"type": "text", "text": "".join(text_parts)})
                text_parts = []
            if isinstance(event, _ToolUse):
                content.append(
                    {
                        "type": "tool_use",
                        "id": event.call_id,
                        "name": event.name,
                        "input": event.arguments,
                    }
                )
                stop_reason = "tool_use"
                break
            if isinstance(event, _Completed):
                if not content and event.result.final_response:
                    content.append(
                        {"type": "text", "text": event.result.final_response}
                    )
                output_tokens = (
                    event.result.usage.output_tokens if event.result.usage else 0
                )
                break
            return _anthropic_error(
                502,
                "api_error",
                "Codex model transport failed",
            )
        response = web.json_response(
            {
                "id": f"msg_codex_{uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": content,
                "model": self.model,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            }
        )
        self._record_replay(
            conversation,
            replay_key,
            _Replay(
                status=response.status,
                headers={"Content-Type": response.headers["Content-Type"]},
                body=cast(bytes, response.body),
                streamed=False,
                expires_at=_now() + _REPLAY_TTL_SECONDS,
            ),
        )
        return response

    def _record_replay(
        self,
        conversation: _Conversation,
        replay_key: str | None,
        replay: _Replay,
    ) -> None:
        if replay_key is None or len(replay.body) > _MAX_REPLAY_BYTES:
            return
        replays = conversation.replays
        replays.pop(replay_key, None)
        replays[replay_key] = replay
        stored = sum(len(entry.body) for entry in replays.values())
        while len(replays) > _MAX_REPLAY_ENTRIES or stored > _MAX_REPLAY_BYTES:
            stored -= len(replays.pop(next(iter(replays))).body)

    def _take_replay(
        self,
        conversation: _Conversation,
        replay_key: str,
        *,
        streamed: bool,
    ) -> _Replay | None:
        replay = conversation.replays.get(replay_key)
        if replay is None or replay.streamed != streamed:
            return None
        if replay.expires_at <= _now():
            conversation.replays.pop(replay_key, None)
            return None
        return replay

    def _is_authorized(self, request: web.Request) -> bool:
        authorization = request.headers.get("Authorization", "")
        bearer = authorization[7:] if authorization.startswith("Bearer ") else ""
        api_key = request.headers.get("x-api-key", "")
        return secrets.compare_digest(
            bearer, self._auth_token
        ) or secrets.compare_digest(
            api_key,
            self._auth_token,
        )


class _RecordingSink:
    """Mirrors the stream into a buffer so a retry can be answered verbatim."""

    def __init__(self, response: web.StreamResponse) -> None:
        self._response = response
        self._chunks: list[bytes] = []
        self._buffered = 0
        self._recording = True

    async def write(self, data: bytes) -> None:
        if self._recording:
            self._buffered += len(data)
            # A body past the cap can never be stored, so holding on to it
            # would buffer an unbounded stream for a replay nobody can keep.
            if self._buffered > _MAX_REPLAY_BYTES:
                self._chunks.clear()
                self._recording = False
            else:
                self._chunks.append(data)
        await self._response.write(data)

    @property
    def body(self) -> bytes | None:
        """None once the stream outgrew the cap; a partial body must not be cached."""
        return b"".join(self._chunks) if self._recording else None


def _now() -> float:
    return asyncio.get_running_loop().time()


def _parse_tools(
    value: object,
) -> tuple[list[CodexDynamicToolSpec], dict[str, str]]:
    if value is None:
        return [], {}
    if not isinstance(value, list):
        raise TypeError("tools must be an array")
    tools: list[CodexDynamicToolSpec] = []
    names: dict[str, str] = {}
    used: set[str] = set()
    for raw_tool in value:
        if not isinstance(raw_tool, dict) or not isinstance(raw_tool.get("name"), str):
            continue
        original = raw_tool["name"]
        safe = _safe_tool_name(original, used)
        used.add(safe)
        schema = raw_tool.get("input_schema", {"type": "object"})
        if not isinstance(schema, dict):
            raise TypeError(f"Tool {original!r} input_schema must be an object")
        tools.append(
            CodexDynamicToolSpec(
                name=safe,
                description=str(raw_tool.get("description") or ""),
                input_schema=cast(dict[str, object], schema),
            )
        )
        names[safe] = original
    return tools, names


def _safe_tool_name(original: str, used: set[str]) -> str:
    if original.startswith("mcp__"):
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", original).strip("_") or "tool"
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
        candidate = f"tool_{normalized[:112]}_{digest}"
        counter = 1
        while candidate in used:
            suffix = f"_{counter}"
            candidate = f"tool_{normalized[: 112 - len(suffix)]}_{digest}{suffix}"
            counter += 1
        return candidate
    if _TOOL_NAME_PATTERN.fullmatch(original) and original not in used:
        return original
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", original).strip("_") or "tool"
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    candidate = f"{normalized[:116]}_{digest}"
    counter = 1
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{normalized[: 116 - len(suffix)]}_{digest}{suffix}"
        counter += 1
    return candidate


def _serialize_system(value: object) -> str | None:
    text = _content_text(value)
    return text or None


def _serialize_messages(value: object) -> str:
    if not isinstance(value, list):
        raise TypeError("messages must be an array")
    normalized: list[dict[str, object]] = []
    for message in value:
        if not isinstance(message, dict):
            raise TypeError("Each message must be an object")
        role = message.get("role")
        # Anthropic accepts `system` turns inline in `messages`, not only in
        # the top-level `system` field; the CLI uses this for mid-conversation
        # reminders, so the role allowlist must include it.
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Message roles must be user, assistant, or system")
        normalized.append(
            {
                "role": role,
                "content": _normalize_content(message.get("content")),
            }
        )
    return (
        "Continue the following conversation as the assistant. Use the supplied "
        "tools when needed. The transcript is JSON:\n"
        + json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    )


def _normalize_content(value: object) -> object:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    normalized: list[object] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            continue
        if block_type == "image":
            normalized.append({"type": "text", "text": "[image input omitted]"})
            continue
        normalized.append(
            {
                key: item
                for key, item in block.items()
                if key not in {"cache_control", "signature"}
            }
        )
    return normalized


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(
        str(block.get("text"))
        for block in value
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
    )


def _delivers_results_with_new_ask(value: object) -> bool:
    """Does the final user message hand back tool results *and* ask for more?

    A continuation answers an outstanding ``tool_use`` and nothing else: its
    last user message is ``tool_result`` blocks, plus at most the
    ``<system-reminder>`` text the CLI injects beside them.  A message that
    also carries free text is a new ask that happens to include results.

    The CLI's auto-compaction request is exactly that shape (captured from
    claude 2.1.274): the whole conversation replayed, the latest
    ``tool_result`` still undelivered, and the summarisation instruction as a
    text block in that same user message.  Trailing ``system`` messages are
    skipped — the CLI appends one to every request.
    """
    if not isinstance(value, list):
        return False
    for message in reversed(value):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return False
        has_result = False
        has_ask = False
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                has_result = True
            elif block.get("type") == "text":
                text = str(block.get("text") or "").lstrip()
                if text and not text.startswith("<system-reminder>"):
                    has_ask = True
        return has_result and has_ask
    return False


def _extract_tool_results(value: object) -> dict[str, CodexDynamicToolResult]:
    results: dict[str, CodexDynamicToolResult] = {}
    if not isinstance(value, list):
        return results
    for message in value:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str):
                continue
            results[call_id] = CodexDynamicToolResult(
                content=_content_text(block.get("content"))
                or json.dumps(block.get("content"), ensure_ascii=False, default=str),
                success=not bool(block.get("is_error")),
            )
    return results


def _tool_result_request_fingerprint(payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload.get("messages"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _estimate_input_tokens(payload: object) -> int:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    return max(1, (len(serialized) + 3) // 4)


def _safe_exception_message(
    exc: BaseException,
    *,
    secrets_to_redact: Iterable[str] = (),
) -> str:
    try:
        message = str(exc)
    except BaseException:
        message = "<unprintable exception message>"
    for secret in sorted(
        (value for value in secrets_to_redact if value),
        key=len,
        reverse=True,
    ):
        message = message.replace(secret, _REDACTED)
    message = _PROVIDER_STATE_PATTERN.sub(f"provider_state={_REDACTED}", message)
    message = _BEARER_PATTERN.sub(f"Bearer {_REDACTED}", message)
    message = _SECRET_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group('label')}={_REDACTED}",
        message,
    )
    message = _JWT_PATTERN.sub(_REDACTED, message)
    message = _DEVICE_CODE_PATTERN.sub(_REDACTED, message)
    message = " ".join(message.split()) or "<empty exception message>"
    if len(message) > _MAX_LOGGED_ERROR_MESSAGE_CHARS:
        return message[: _MAX_LOGGED_ERROR_MESSAGE_CHARS - 3] + "..."
    return message


async def _write_sse(sink: _SseSink, event: dict[str, object]) -> None:
    event_type = str(event["type"])
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    await sink.write(f"event: {event_type}\ndata: {payload}\n\n".encode())


async def _write_tool_use(
    sink: _SseSink,
    index: int,
    event: _ToolUse,
) -> None:
    await _write_sse(
        sink,
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": event.call_id,
                "name": event.name,
                "input": {},
            },
        },
    )
    await _write_sse(
        sink,
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(event.arguments, ensure_ascii=False),
            },
        },
    )
    await _write_sse(
        sink,
        {"type": "content_block_stop", "index": index},
    )


async def _write_message_end(
    sink: _SseSink,
    stop_reason: str,
    output_tokens: int,
) -> None:
    await _write_sse(
        sink,
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    await _write_sse(sink, {"type": "message_stop"})


def _anthropic_error(status: int, error_type: str, message: str) -> web.Response:
    return web.json_response(
        {
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
        status=status,
    )


def _sum_usage(
    usages: Iterable[CodexTokenUsage | None],
) -> CodexTokenUsage | None:
    values = [usage for usage in usages if usage is not None]
    if not values:
        return None
    return CodexTokenUsage(
        input_tokens=sum(value.input_tokens for value in values),
        cached_input_tokens=sum(value.cached_input_tokens for value in values),
        output_tokens=sum(value.output_tokens for value in values),
        reasoning_output_tokens=sum(value.reasoning_output_tokens for value in values),
        total_tokens=sum(value.total_tokens for value in values),
    )
