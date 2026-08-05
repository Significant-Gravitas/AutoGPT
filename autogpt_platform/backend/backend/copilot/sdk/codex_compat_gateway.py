from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager, suppress
from typing import Protocol, cast
from uuid import uuid4

from aiohttp import web
from openai_codex.generated.v2_all import AgentMessageDeltaNotification
from openai_codex.models import Notification
from pydantic import BaseModel, ConfigDict, Field

from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexReasoningEffort,
    CodexTokenUsage,
)
from backend.integrations.codex.transport import (
    CodexAgentSession,
    CodexTransport,
    get_codex_transport,
)
from backend.integrations.credential_lease import CredentialLease

_MAX_REQUEST_BYTES = 16 * 1024 * 1024
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


class _Conversation(_GatewayState):
    id: str
    queue: asyncio.Queue[_ConversationEvent] = Field(default_factory=asyncio.Queue)
    pending: dict[str, _ToolCallRecord] = Field(default_factory=dict)
    response_lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = None
    result: CodexInvocationResult | None = None


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


class _DuplicateToolResultError(ValueError):
    pass


class CodexAnthropicGateway:
    def __init__(
        self,
        *,
        credential_lease: CredentialLease,
        model: str,
        effort: CodexReasoningEffort | None = None,
        transport: CodexTransport | _AgentTransport | None = None,
    ) -> None:
        self._credential_lease = credential_lease
        self.model = model
        self.effort: CodexReasoningEffort | None = effort
        self._transport = cast(
            _AgentTransport,
            transport if transport is not None else get_codex_transport(),
        )
        self._agent_context: AbstractAsyncContextManager[CodexAgentSession] | None = (
            None
        )
        self._agent_session: CodexAgentSession | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._base_url: str | None = None
        self._auth_token = secrets.token_urlsafe(32)
        self._conversations: dict[str, _Conversation] = {}
        self._tool_calls: dict[str, _ToolCallRecord] = {}
        self._results: list[CodexInvocationResult] = []
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

    async def start(self) -> None:
        await self.__aenter__()

    async def __aenter__(self) -> "CodexAnthropicGateway":
        if self._runner is not None:
            return self
        self._agent_context = self._transport.agent_session(self._credential_lease)
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
            conversation = self._continue_conversation(payload)
            if conversation is None:
                conversation = self._start_conversation(payload)
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

        input_tokens = _estimate_input_tokens(payload)
        if payload.get("stream") is True:
            return await self._streaming_response(
                request,
                conversation,
                input_tokens,
            )
        return await self._nonstreaming_response(conversation, input_tokens)

    def _continue_conversation(
        self,
        payload: dict[str, object],
    ) -> _Conversation | None:
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

        claimable = [
            (record, result)
            for record, result in known
            if not record.closed and record.result is None
        ]
        fingerprint = _tool_result_request_fingerprint(payload)
        if not claimable:
            if any(record.claim_fingerprint == fingerprint for record, _ in known):
                raise _DuplicateToolResultError(
                    "This tool-result request was already accepted"
                )
            if any(record.result is None for record, _ in known):
                raise _DuplicateToolResultError(
                    "This tool-result request refers to a closed model call"
                )
            return None

        conversation = claimable[0][0].conversation
        if any(record.conversation is not conversation for record, _ in claimable):
            raise ValueError("Tool results span multiple model conversations")

        for record, result in claimable:
            record.result = result
            record.claim_fingerprint = fingerprint
        for record, result in claimable:
            if not record.future.done():
                record.future.set_result(result)
        return conversation

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
        agent_session: CodexAgentSession,
        invocation: CodexInvocationRequest,
        tools: list[CodexDynamicToolSpec],
        original_names: dict[str, str],
    ) -> None:
        async def handle_event(notification: Notification) -> None:
            payload = notification.payload
            if isinstance(payload, AgentMessageDeltaNotification) and payload.delta:
                await conversation.queue.put(_TextDelta(text=payload.delta))

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

    async def _streaming_response(
        self,
        request: web.Request,
        conversation: _Conversation,
        input_tokens: int,
    ) -> web.StreamResponse:
        async with conversation.response_lock:
            return await self._streaming_response_locked(
                request,
                conversation,
                input_tokens,
            )

    async def _streaming_response_locked(
        self,
        request: web.Request,
        conversation: _Conversation,
        input_tokens: int,
    ) -> web.StreamResponse:
        first = await conversation.queue.get()
        if isinstance(first, _Failed):
            return _anthropic_error(
                502,
                "api_error",
                "Codex model transport failed",
            )

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        message_id = f"msg_codex_{uuid4().hex}"
        try:
            await _write_sse(
                response,
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
            await self._write_boundary(response, conversation, first)
            await response.write_eof()
        except (ConnectionError, ConnectionResetError, asyncio.CancelledError):
            if conversation.task is not None:
                conversation.task.cancel()
            raise
        return response

    async def _write_boundary(
        self,
        response: web.StreamResponse,
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
                        response,
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    text_open = True
                await _write_sse(
                    response,
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": event.text},
                    },
                )
            elif isinstance(event, _ToolUse):
                if text_open:
                    await _write_sse(
                        response,
                        {"type": "content_block_stop", "index": index},
                    )
                    index += 1
                    text_open = False
                await _write_tool_use(response, index, event)
                await _write_message_end(response, "tool_use", 0)
                return
            elif isinstance(event, _Completed):
                if not emitted_text and event.result.final_response:
                    if not text_open:
                        await _write_sse(
                            response,
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                        text_open = True
                    await _write_sse(
                        response,
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
                        response,
                        {"type": "content_block_stop", "index": index},
                    )
                output_tokens = (
                    event.result.usage.output_tokens if event.result.usage else 0
                )
                await _write_message_end(response, "end_turn", output_tokens)
                return
            else:
                await _write_sse(
                    response,
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
    ) -> web.Response:
        async with conversation.response_lock:
            return await self._nonstreaming_response_locked(
                conversation,
                input_tokens,
            )

    async def _nonstreaming_response_locked(
        self,
        conversation: _Conversation,
        input_tokens: int,
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
        return web.json_response(
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
        candidate = f"{normalized[: 128 - len(suffix)]}{suffix}"
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
        if role not in {"user", "assistant"}:
            raise ValueError("Message roles must be user or assistant")
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


async def _write_sse(response: web.StreamResponse, event: dict[str, object]) -> None:
    event_type = str(event["type"])
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    await response.write(f"event: {event_type}\ndata: {payload}\n\n".encode())


async def _write_tool_use(
    response: web.StreamResponse,
    index: int,
    event: _ToolUse,
) -> None:
    await _write_sse(
        response,
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
        response,
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
        response,
        {"type": "content_block_stop", "index": index},
    )


async def _write_message_end(
    response: web.StreamResponse,
    stop_reason: str,
    output_tokens: int,
) -> None:
    await _write_sse(
        response,
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    await _write_sse(response, {"type": "message_stop"})


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
