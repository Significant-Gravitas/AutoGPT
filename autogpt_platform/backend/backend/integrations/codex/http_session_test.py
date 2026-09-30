import asyncio
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.http_session import (
    MAX_TOOL_ITERATIONS,
    CodexHttpSession,
    CodexTurnFailedError,
    CodexTurnLimitError,
    _add_usage,
    _parse_arguments,
)
from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexStreamEvent,
    CodexTokenUsage,
)

# --------------------------------------------------------------------------- #
# Fakes for the streaming SDK surface
# --------------------------------------------------------------------------- #


class _Event:
    def __init__(self, type: str, **fields: Any) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class _Item:
    def __init__(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class _RawResponse:
    def __init__(self, events: list[_Event], headers: dict[str, str]) -> None:
        self._events = events
        self.headers = headers

    def parse(self):
        # Match LegacyAPIResponse.parse, which synchronously returns the
        # AsyncStream for an SSE response.
        async def stream():
            for event in self._events:
                yield event

        return stream()


class _FakeResponses:
    """Returns one scripted turn per call, recording each request."""

    def __init__(self, turns: list[list[_Event]], headers: dict[str, str]) -> None:
        self._turns = turns
        self._headers = headers
        self.requests: list[dict[str, Any]] = []

    @property
    def with_raw_response(self):
        return self

    async def create(self, **payload: Any) -> _RawResponse:
        self.requests.append(payload)
        events = self._turns.pop(0) if self._turns else []
        return _RawResponse(events, self._headers)


class _FakeClient:
    def __init__(self, turns: list[list[_Event]], headers: dict[str, str]) -> None:
        self.responses = _FakeResponses(turns, headers)


def _credentials() -> OAuth2Credentials:
    return OAuth2Credentials(
        provider="codex",
        title="t",
        access_token=SecretStr("at"),
        refresh_token=SecretStr("rt"),
        scopes=[],
    )


def _session(turns: list[list[_Event]], headers: dict[str, str] | None = None):
    client = _FakeClient(turns, headers or {})
    session = CodexHttpSession(
        _credentials(),
        turn_timeout_seconds=30,
        tool_timeout_seconds=5,
        client=client,  # type: ignore[arg-type]
    )
    return session, client


def _text_turn(text: str, response_id: str = "resp-1") -> list[_Event]:
    return [
        _Event("response.created", response=_Item(id=response_id, model="gpt-5.6-sol")),
        _Event("response.output_text.delta", delta=text),
        _Event(
            "response.completed",
            response=_Item(id=response_id, model="gpt-5.6-sol", usage=None),
        ),
    ]


def _tool_turn(name: str, arguments: str, call_id: str = "call-1") -> list[_Event]:
    return [
        _Event("response.created", response=_Item(id="resp-0", model="gpt-5.6-sol")),
        _Event(
            "response.output_item.done",
            item=_Item(
                type="function_call", name=name, arguments=arguments, call_id=call_id
            ),
        ),
    ]


async def _echo_tool(call: CodexDynamicToolCall) -> CodexDynamicToolResult:
    return CodexDynamicToolResult(content=f"ran {call.tool}")


def _request(**over: Any) -> CodexInvocationRequest:
    fields: dict[str, Any] = {"prompt": "hi", "model": "gpt-5.6-sol"}
    fields.update(over)
    return CodexInvocationRequest(**fields)


TOOL = CodexDynamicToolSpec(
    name="lookup", description="Look something up.", input_schema={"type": "object"}
)


# --------------------------------------------------------------------------- #
# The installed OpenAI SDK streaming contract
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_installed_sdk_raw_stream_is_consumed_without_awaiting_it() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test",
            http_client=http,
        )
        session = CodexHttpSession(
            _credentials(),
            turn_timeout_seconds=30,
            tool_timeout_seconds=5,
            client=client,
        )

        result = await session.invoke(_request(), [], _echo_tool)

    assert result.status == "completed"


# --------------------------------------------------------------------------- #
# The tool loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_turn_without_tool_calls_completes_immediately() -> None:
    session, client = _session([_text_turn("hello")])
    result = await session.invoke(_request(), [], _echo_tool)

    assert result.final_response == "hello"
    assert result.status == "completed"
    assert len(client.responses.requests) == 1


@pytest.mark.asyncio
async def test_a_tool_call_is_dispatched_and_its_output_fed_back() -> None:
    session, client = _session([_tool_turn("lookup", '{"q": "x"}'), _text_turn("done")])
    seen: list[CodexDynamicToolCall] = []

    async def handler(call: CodexDynamicToolCall) -> CodexDynamicToolResult:
        seen.append(call)
        return CodexDynamicToolResult(content="tool output")

    result = await session.invoke(_request(), [TOOL], handler)

    assert [c.tool for c in seen] == ["lookup"]
    # Arguments reach the handler parsed, not as a JSON blob.
    assert seen[0].arguments == {"q": "x"}
    assert result.final_response == "done"

    second_input = client.responses.requests[1]["input"]
    outputs = [i for i in second_input if i.get("type") == "function_call_output"]
    assert outputs == [
        {"type": "function_call_output", "call_id": "call-1", "output": "tool output"}
    ]


@pytest.mark.asyncio
async def test_every_output_item_is_replayed_so_reasoning_survives_a_tool_hop() -> None:
    """Replaying only the tool calls would break the chain on reasoning models."""
    first = _tool_turn("lookup", "{}")
    first.insert(
        1,
        _Event(
            "response.output_item.done",
            item=_Item(type="reasoning", id="rs-1", encrypted_content="opaque"),
        ),
    )
    session, client = _session([first, _text_turn("done")])

    await session.invoke(_request(), [TOOL], _echo_tool)

    replayed = client.responses.requests[1]["input"]
    assert any(item.get("type") == "reasoning" for item in replayed)
    assert client.responses.requests[0]["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_a_tool_timeout_is_reported_to_the_model_not_raised() -> None:
    session, client = _session([_tool_turn("lookup", "{}"), _text_turn("recovered")])

    async def slow(call: CodexDynamicToolCall) -> CodexDynamicToolResult:
        await asyncio.sleep(10)
        return CodexDynamicToolResult(content="never")

    session._tool_timeout_seconds = 0.01  # type: ignore[attr-defined]
    result = await session.invoke(_request(), [TOOL], slow)

    assert result.final_response == "recovered"
    output = [
        i
        for i in client.responses.requests[1]["input"]
        if i.get("type") == "function_call_output"
    ][0]
    assert "timed out" in output["output"]


@pytest.mark.asyncio
async def test_a_tool_failure_is_returned_to_the_model_without_leaking_details() -> (
    None
):
    session, client = _session([_tool_turn("lookup", "{}"), _text_turn("recovered")])

    async def broken(_call: CodexDynamicToolCall) -> CodexDynamicToolResult:
        raise RuntimeError("secret provider detail")

    result = await session.invoke(_request(), [TOOL], broken)

    assert result.final_response == "recovered"
    output = [
        item
        for item in client.responses.requests[1]["input"]
        if item.get("type") == "function_call_output"
    ][0]["output"]
    assert output == "Tool 'lookup' failed."
    assert "secret provider detail" not in output


@pytest.mark.asyncio
async def test_a_loop_that_never_converges_stops_instead_of_spending_quota() -> None:
    turns = [_tool_turn("lookup", "{}") for _ in range(MAX_TOOL_ITERATIONS + 2)]
    session, _ = _session(turns)

    with pytest.raises(CodexTurnLimitError):
        await session.invoke(_request(), [TOOL], _echo_tool)


@pytest.mark.asyncio
async def test_a_failed_response_raises_rather_than_returning_empty_text() -> None:
    events = [_Event("response.failed", response=_Item(error=_Item(message="boom")))]
    session, _ = _session([events])

    with pytest.raises(CodexTurnFailedError):
        await session.invoke(_request(), [], _echo_tool)


# --------------------------------------------------------------------------- #
# Streaming, payload shape and accounting
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_text_and_reasoning_deltas_are_reported_separately() -> None:
    events = _text_turn("hi")
    events.insert(1, _Event("response.reasoning_summary_text.delta", delta="thinking"))
    session, _ = _session([events])

    seen: list[CodexStreamEvent] = []

    async def on_event(event: CodexStreamEvent) -> None:
        seen.append(event)

    result = await session.invoke(_request(), [], _echo_tool, on_event)

    assert [(e.type, e.delta) for e in seen] == [
        ("reasoning_delta", "thinking"),
        ("text_delta", "hi"),
    ]
    assert result.reasoning_summary == "thinking"


@pytest.mark.asyncio
async def test_tools_are_omitted_entirely_when_none_are_offered() -> None:
    """Sending an empty tools array is not the same as sending none."""
    session, client = _session([_text_turn("hi")])
    await session.invoke(_request(), [], _echo_tool)

    payload = client.responses.requests[0]
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert payload["store"] is False


@pytest.mark.asyncio
async def test_effort_and_output_schema_are_passed_through() -> None:
    session, client = _session([_text_turn("hi")])
    await session.invoke(
        _request(effort="high", output_schema={"type": "object"}), [], _echo_tool
    )

    payload = client.responses.requests[0]
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["text"]["format"]["schema"] == {"type": "object"}


@pytest.mark.asyncio
async def test_rate_limits_are_captured_from_the_response_headers() -> None:
    session, _ = _session(
        [_text_turn("hi")],
        headers={
            "x-codex-plan-type": "pro",
            "x-codex-primary-used-percent": "7",
            "x-codex-primary-window-minutes": "10080",
        },
    )
    await session.invoke(_request(), [], _echo_tool)

    assert session.rate_limits is not None
    assert session.rate_limits.plan_type == "pro"
    assert session.rate_limits.primary is not None
    assert session.rate_limits.primary.used_percent == 7


def test_usage_accumulates_across_tool_hops() -> None:
    """Usage is reported per response, so a multi-hop turn must sum it."""
    one = CodexTokenUsage(
        input_tokens=10,
        cached_input_tokens=1,
        output_tokens=2,
        reasoning_output_tokens=3,
        total_tokens=12,
    )
    total = _add_usage(_add_usage(one, one), None)
    assert total.input_tokens == 20
    assert total.total_tokens == 24
    assert total.reasoning_output_tokens == 6


@pytest.mark.parametrize(
    "raw,expected",
    [('{"a": 1}', {"a": 1}), ("", {}), ("not json", "not json"), ({"a": 1}, {"a": 1})],
)
def test_tool_arguments_are_parsed_without_killing_the_turn(raw, expected) -> None:
    assert _parse_arguments(raw) == expected


# --------------------------------------------------------------------------- #
# Rate-limit reporting through the transport
#
# ChatGPT reports quota only on inference response headers, so a connection
# that has not run a turn genuinely has nothing to report.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rate_limits_are_unknown_before_any_turn_rather_than_an_error() -> None:
    from backend.integrations.codex.transport import CodexTransport

    class _Lease:
        credentials = _credentials_with_state()

    limits = await CodexTransport().rate_limits(_Lease())  # type: ignore[arg-type]
    assert limits.primary is None
    assert limits.plan_type is None


def _credentials_with_state() -> OAuth2Credentials:
    from backend.integrations.codex.auth_bundle import (
        CodexAuthBundleV1,
        CodexAuthTokensV1,
    )
    from backend.integrations.codex.credential_codec import credentials_from_bundle

    token = (
        "eyJhbGciOiJub25lIn0." "eyJleHAiOjk5OTk5OTk5OTksImVtYWlsIjoiYUBiLmMifQ." "sig"
    )
    return credentials_from_bundle(
        CodexAuthBundleV1(
            tokens=CodexAuthTokensV1(
                id_token=SecretStr(token),
                access_token=SecretStr(token),
                refresh_token=SecretStr("rt"),
            ),
            codex_runtime_version="http",
        )
    )
