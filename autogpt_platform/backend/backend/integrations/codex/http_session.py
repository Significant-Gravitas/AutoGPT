"""Agentic Codex turns over HTTP.

The ChatGPT backend speaks the Responses API, so a turn is an ordinary
streaming request plus a tool loop: stream a response, hand any function calls
to the caller's handler, feed the outputs back, repeat until the model stops
asking for tools.

Two details make the difference between this working and subtly degrading on
reasoning models:

- every output item is echoed back into the next request, reasoning items
  included, so the model keeps its chain across tool hops; and
- ``include=["reasoning.encrypted_content"]`` is requested so those items come
  back in a form that can be replayed at all.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.http_client import build_client, parse_rate_limits
from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexRateLimitsSnapshot,
    CodexStreamEvent,
    CodexTokenUsage,
)

logger = logging.getLogger(__name__)

EventHandler = Callable[[CodexStreamEvent], Awaitable[None]]
ToolHandler = Callable[[CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]]

# A turn that keeps asking for tools without converging is a bug somewhere;
# stop rather than spending someone's quota in a loop.
MAX_TOOL_ITERATIONS = 32


class CodexHttpSession:
    """One user's Codex turn, run over HTTPS."""

    def __init__(
        self,
        credentials: OAuth2Credentials,
        *,
        turn_timeout_seconds: float,
        tool_timeout_seconds: float,
        client: AsyncOpenAI | None = None,
    ) -> None:
        # Built per session and never shared: an AsyncOpenAI instance carries
        # the credential it was made with, so a reused one would bill another
        # user's subscription.
        self._client = client or build_client(credentials)
        self._turn_timeout_seconds = turn_timeout_seconds
        self._tool_timeout_seconds = tool_timeout_seconds
        self._rate_limits: CodexRateLimitsSnapshot | None = None

    @property
    def rate_limits(self) -> CodexRateLimitsSnapshot | None:
        """Quota as of the last call; the backend reports it on every response."""
        return self._rate_limits

    async def invoke(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: ToolHandler,
        event_handler: EventHandler | None = None,
    ) -> CodexInvocationResult:
        timeout = request.timeout_seconds or self._turn_timeout_seconds
        try:
            return await asyncio.wait_for(
                self._run(request, dynamic_tools, tool_handler, event_handler),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise CodexInvocationTimeoutError("codex_copilot_turn_timeout") from None

    async def _run(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: ToolHandler,
        event_handler: EventHandler | None,
    ) -> CodexInvocationResult:
        started = time.monotonic()
        conversation: list[dict[str, Any]] = [_user_message(request.prompt)]
        tools = [_tool_payload(spec) for spec in dynamic_tools]

        response_id = ""
        resolved_model: str | None = None
        final_text = ""
        reasoning_summary = ""
        usage = _ZERO_USAGE

        for _ in range(MAX_TOOL_ITERATIONS):
            turn = await self._stream_turn(request, conversation, tools, event_handler)

            response_id = turn.response_id or response_id
            resolved_model = turn.model or resolved_model
            usage = _add_usage(usage, turn.usage)
            if turn.text:
                final_text = turn.text
            if turn.reasoning_summary:
                reasoning_summary = turn.reasoning_summary

            # Replay the whole output, not just the tool calls: dropping the
            # reasoning items breaks the model's chain on reasoning models.
            conversation.extend(turn.output_items)

            if not turn.tool_calls:
                return CodexInvocationResult(
                    response_id=response_id,
                    final_response=final_text,
                    reasoning_summary=reasoning_summary or None,
                    status="completed",
                    resolved_model=resolved_model or request.model,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    usage=usage,
                )

            for call in turn.tool_calls:
                conversation.append(await self._dispatch_tool(call, tool_handler))

        raise CodexTurnLimitError(
            f"Codex turn did not converge within {MAX_TOOL_ITERATIONS} tool rounds"
        )

    async def _dispatch_tool(
        self, call: "_ToolCall", tool_handler: ToolHandler
    ) -> dict[str, Any]:
        request = CodexDynamicToolCall(
            thread_id=call.response_id,
            turn_id=call.response_id,
            call_id=call.call_id,
            namespace=None,
            tool=call.name,
            arguments=call.arguments,
        )
        try:
            result = await asyncio.wait_for(
                tool_handler(request), timeout=self._tool_timeout_seconds
            )
            output = result.content
        except asyncio.TimeoutError:
            # Report the timeout to the model rather than failing the turn: it
            # can usually recover by trying something else.
            output = f"Tool {call.name!r} timed out."
            logger.warning("Codex tool %s timed out", call.name)
        except Exception:
            # Tool failures are model-visible results, not transport failures:
            # the model can explain, retry, or choose another path. Do not
            # include exception text because tool errors may contain secrets.
            output = f"Tool {call.name!r} failed."
            logger.exception("Codex tool %s failed", call.name)

        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": output,
        }

    async def _stream_turn(
        self,
        request: CodexInvocationRequest,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        event_handler: EventHandler | None,
    ) -> "_TurnOutcome":
        payload: dict[str, Any] = {
            "model": request.model,
            "input": list(conversation),
            "store": False,
            "stream": True,
            # Without this the reasoning items replayed above are unusable.
            "include": ["reasoning.encrypted_content"],
        }
        if request.instructions:
            payload["instructions"] = request.instructions
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if request.effort:
            payload["reasoning"] = {"effort": request.effort}
        if request.output_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "schema": request.output_schema,
                    "strict": False,
                }
            }

        raw = await self._client.responses.with_raw_response.create(**payload)
        self._rate_limits = parse_rate_limits(raw.headers)

        outcome = _TurnOutcome()
        # with_raw_response returns LegacyAPIResponse for streaming requests;
        # its synchronous parse() returns the AsyncStream to iterate.
        stream = raw.parse()
        async for event in stream:
            await _consume_event(event, outcome, event_handler)
        return outcome


async def _consume_event(
    event: Any, outcome: "_TurnOutcome", event_handler: EventHandler | None
) -> None:
    kind = getattr(event, "type", "")

    if kind == "response.output_text.delta":
        delta = getattr(event, "delta", "") or ""
        outcome.text += delta
        if delta and event_handler:
            await event_handler(CodexStreamEvent(type="text_delta", delta=delta))
        return

    if kind == "response.reasoning_summary_text.delta":
        delta = getattr(event, "delta", "") or ""
        outcome.reasoning_summary += delta
        if delta and event_handler:
            await event_handler(CodexStreamEvent(type="reasoning_delta", delta=delta))
        return

    if kind == "response.output_item.done":
        item = getattr(event, "item", None)
        if item is None:
            return
        outcome.output_items.append(_as_dict(item))
        if getattr(item, "type", "") == "function_call":
            outcome.tool_calls.append(
                _ToolCall(
                    call_id=getattr(item, "call_id", "") or "",
                    name=getattr(item, "name", "") or "",
                    arguments=_parse_arguments(getattr(item, "arguments", "")),
                    response_id=outcome.response_id,
                )
            )
        return

    if kind in ("response.created", "response.completed", "response.in_progress"):
        response = getattr(event, "response", None)
        if response is None:
            return
        outcome.response_id = getattr(response, "id", "") or outcome.response_id
        outcome.model = getattr(response, "model", None) or outcome.model
        if (raw_usage := getattr(response, "usage", None)) is not None:
            outcome.usage = _usage_from(raw_usage)
        return

    if kind in ("response.failed", "error"):
        raise CodexTurnFailedError(_failure_message(event))


class CodexInvocationTimeoutError(RuntimeError):
    """The whole turn exceeded its deadline."""


class CodexTurnLimitError(RuntimeError):
    """The tool loop did not converge."""


class CodexTurnFailedError(RuntimeError):
    """ChatGPT reported the turn as failed."""


class _ToolCall:
    __slots__ = ("call_id", "name", "arguments", "response_id")

    def __init__(
        self, call_id: str, name: str, arguments: object, response_id: str
    ) -> None:
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.response_id = response_id


class _TurnOutcome:
    def __init__(self) -> None:
        self.response_id = ""
        self.model: str | None = None
        self.text = ""
        self.reasoning_summary = ""
        self.output_items: list[dict[str, Any]] = []
        self.tool_calls: list[_ToolCall] = []
        self.usage: CodexTokenUsage | None = None


_ZERO_USAGE = CodexTokenUsage(
    input_tokens=0,
    cached_input_tokens=0,
    output_tokens=0,
    reasoning_output_tokens=0,
    total_tokens=0,
)


def _user_message(prompt: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": prompt}],
    }


def _tool_payload(spec: CodexDynamicToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.input_schema,
        "strict": False,
    }


def _parse_arguments(raw: object) -> object:
    """Arguments arrive as a JSON string; hand the handler the parsed form."""
    if not isinstance(raw, str):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        # A malformed call is the model's mistake, not ours -- pass it through
        # so the handler can report it back rather than killing the turn.
        return raw


def _as_dict(item: Any) -> dict[str, Any]:
    """Normalise an SDK output item into the shape the next request replays."""
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        dumped = dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    return dict(item) if isinstance(item, dict) else {}


def _usage_from(raw: Any) -> CodexTokenUsage:
    input_details = getattr(raw, "input_tokens_details", None)
    output_details = getattr(raw, "output_tokens_details", None)
    return CodexTokenUsage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        reasoning_output_tokens=int(
            getattr(output_details, "reasoning_tokens", 0) or 0
        ),
        total_tokens=int(getattr(raw, "total_tokens", 0) or 0),
    )


def _add_usage(
    running: CodexTokenUsage, turn: CodexTokenUsage | None
) -> CodexTokenUsage:
    """Usage is per response, so a multi-hop turn has to accumulate it."""
    if turn is None:
        return running
    return CodexTokenUsage(
        input_tokens=running.input_tokens + turn.input_tokens,
        cached_input_tokens=running.cached_input_tokens + turn.cached_input_tokens,
        output_tokens=running.output_tokens + turn.output_tokens,
        reasoning_output_tokens=(
            running.reasoning_output_tokens + turn.reasoning_output_tokens
        ),
        total_tokens=running.total_tokens + turn.total_tokens,
    )


def _failure_message(event: Any) -> str:
    response = getattr(event, "response", None)
    error = getattr(response, "error", None) or getattr(event, "error", None)
    message = getattr(error, "message", None)
    return str(message or "ChatGPT reported the turn as failed")
