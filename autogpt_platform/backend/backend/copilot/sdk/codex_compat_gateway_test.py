import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import pytest
from aiohttp import ClientSession
from openai_codex.generated.v2_all import AgentMessageDeltaNotification
from openai_codex.models import Notification

from backend.copilot.sdk.codex_compat_gateway import (
    CodexAnthropicGateway,
    _safe_tool_name,
)
from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexTokenUsage,
)
from backend.integrations.credential_lease import CredentialLease


class _FakeAgentSession:
    def __init__(self, *, use_tool: bool = False) -> None:
        self.use_tool = use_tool
        self.requests: list[CodexInvocationRequest] = []
        self.tools: list[list[CodexDynamicToolSpec]] = []
        self.tool_result: CodexDynamicToolResult | None = None
        self.cancelled = asyncio.Event()

    async def invoke(
        self,
        request,
        dynamic_tools,
        tool_handler,
        event_handler=None,
    ) -> CodexInvocationResult:
        self.requests.append(request)
        self.tools.append(dynamic_tools)
        try:
            if self.use_tool:
                self.tool_result = await tool_handler(
                    CodexDynamicToolCall(
                        thread_id="thread-1",
                        turn_id="turn-1",
                        call_id="call-1",
                        tool=dynamic_tools[0].name,
                        arguments={"query": "status"},
                    )
                )
            if event_handler is not None:
                await event_handler(
                    Notification(
                        method="item/agentMessage/delta",
                        payload=AgentMessageDeltaNotification(
                            delta="done" if self.use_tool else "hello",
                            itemId="item-1",
                            threadId="thread-1",
                            turnId="turn-1",
                        ),
                    )
                )
            return _result("done" if self.use_tool else "hello")
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _FailingAgentSession(_FakeAgentSession):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def invoke(
        self,
        request,
        dynamic_tools,
        tool_handler,
        event_handler=None,
    ) -> CodexInvocationResult:
        raise self.error


class _FinalWithoutDeltaSession:
    async def invoke(
        self,
        request,
        dynamic_tools,
        tool_handler,
        event_handler=None,
    ) -> CodexInvocationResult:
        assert event_handler is not None
        await event_handler(
            Notification(
                method="item/agentMessage/delta",
                payload=AgentMessageDeltaNotification(
                    delta="before tool",
                    itemId="item-before",
                    threadId="thread-before",
                    turnId="turn-before",
                ),
            )
        )
        await tool_handler(
            CodexDynamicToolCall(
                thread_id="thread-before",
                turn_id="turn-before",
                call_id="raw-before",
                tool=dynamic_tools[0].name,
                arguments={},
            )
        )
        return _result("after tool")


class _CollidingCallSession:
    def __init__(self) -> None:
        self.results: dict[int, CodexDynamicToolResult] = {}
        self.raw_call_ids: list[str] = []

    async def invoke(
        self,
        request,
        dynamic_tools,
        tool_handler,
        event_handler=None,
    ) -> CodexInvocationResult:
        invocation_index = len(self.raw_call_ids)
        self.raw_call_ids.append("raw-collision")
        result = await tool_handler(
            CodexDynamicToolCall(
                thread_id=f"thread-{invocation_index}",
                turn_id=f"turn-{invocation_index}",
                call_id="raw-collision",
                tool=dynamic_tools[0].name,
                arguments={"invocation": invocation_index},
            )
        )
        self.results[invocation_index] = result
        assert event_handler is not None
        await event_handler(
            Notification(
                method="item/agentMessage/delta",
                payload=AgentMessageDeltaNotification(
                    delta=f"completed:{result.content}",
                    itemId=f"item-{invocation_index}",
                    threadId=f"thread-{invocation_index}",
                    turnId=f"turn-{invocation_index}",
                ),
            )
        )
        return _result(f"completed:{result.content}")


class _FakeTransport:
    def __init__(self, session: _FakeAgentSession) -> None:
        self.session = session
        self.entered = False
        self.exited = False

    @asynccontextmanager
    async def agent_session(
        self,
        _lease: CredentialLease,
    ) -> AsyncIterator[_FakeAgentSession]:
        self.entered = True
        try:
            yield self.session
        finally:
            self.exited = True


def _result(text: str) -> CodexInvocationResult:
    return CodexInvocationResult(
        response_id="response-1",
        final_response=text,
        status="completed",
        usage=CodexTokenUsage(
            input_tokens=10,
            cached_input_tokens=2,
            output_tokens=3,
            reasoning_output_tokens=1,
            total_tokens=13,
        ),
    )


def _lease() -> CredentialLease:
    return cast(CredentialLease, object())


def _headers(gateway: CodexAnthropicGateway) -> dict[str, str]:
    return {"Authorization": f"Bearer {gateway.auth_token}"}


async def _post_simple_message(gateway: CodexAnthropicGateway) -> int:
    async with ClientSession() as client:
        response = await client.post(
            f"{gateway.base_url}/v1/messages",
            headers=_headers(gateway),
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        await response.read()
        return response.status


def _events(body: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _tool_request(message: str, *, stream: bool | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": message}],
        "tools": [
            {
                "name": "mcp.tool/status",
                "description": "Read status",
                "input_schema": {"type": "object"},
            }
        ],
    }
    if stream is not None:
        payload["stream"] = stream
    return payload


def _tool_result_request(
    gateway_call_id: str,
    content: str,
    *,
    stream: bool | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": gateway_call_id,
                        "content": content,
                    }
                ],
            }
        ]
    }
    if stream is not None:
        payload["stream"] = stream
    return payload


def _tool_use_id(payload: dict[str, object]) -> str:
    content = payload["content"]
    assert isinstance(content, list)
    block = next(
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_use"
    )
    call_id = block["id"]
    assert isinstance(call_id, str)
    return call_id


def test_reserved_mcp_tool_name_is_deterministically_remapped() -> None:
    original = "mcp__copilot__add_understanding"

    first = _safe_tool_name(original, set())
    second = _safe_tool_name(original, set())

    assert first == second
    assert first != original
    assert not first.startswith("mcp__")
    assert len(first) <= 128


def test_reserved_mcp_tool_name_collision_is_remapped_uniquely() -> None:
    original = "mcp__copilot__add_understanding"
    first = _safe_tool_name(original, set())

    collided = _safe_tool_name(original, {first})

    assert collided != first
    assert collided == _safe_tool_name(original, {first})
    assert not collided.startswith("mcp__")
    assert len(collided) <= 128


def test_reserved_mcp_tool_name_remap_respects_max_length() -> None:
    remapped = _safe_tool_name("mcp__" + "x" * 300, set())

    assert len(remapped) == 128
    assert not remapped.startswith("mcp__")


def test_non_reserved_valid_tool_name_is_unchanged() -> None:
    assert _safe_tool_name("regular_tool", set()) == "regular_tool"


@pytest.mark.asyncio
async def test_streams_codex_text_as_anthropic_events() -> None:
    session = _FakeAgentSession()
    transport = _FakeTransport(session)
    async with CodexAnthropicGateway(
        credential_lease=_lease(),
        model="gpt-5.6-terra",
        effort="high",
        transport=transport,
    ) as gateway:
        async with ClientSession() as client:
            response = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json={
                    "model": "ignored-by-transport",
                    "stream": True,
                    "system": "follow the system",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert response.status == 200
            events = _events(await response.text())

        assert [event["type"] for event in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[2]["delta"] == {"type": "text_delta", "text": "hello"}
        assert events[0]["message"]["usage"]["input_tokens"] > 0
        assert session.requests[0].model == "gpt-5.6-terra"
        assert session.requests[0].effort == "high"
        assert session.requests[0].instructions == "follow the system"
        assert '"content":"hello"' in session.requests[0].prompt
        assert gateway.result is not None
        assert gateway.result.usage is not None
        assert gateway.result.usage.input_tokens == 10

    assert transport.entered
    assert transport.exited


@pytest.mark.asyncio
async def test_transport_failure_logs_useful_bounded_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _FailingAgentSession(
        RuntimeError("app-server overloaded while starting turn " + "x" * 500)
    )
    transport = _FakeTransport(session)
    with caplog.at_level(
        logging.ERROR,
        logger="backend.copilot.sdk.codex_compat_gateway",
    ):
        async with CodexAnthropicGateway(
            credential_lease=_lease(),
            model="gpt-5.6-terra",
            transport=transport,
        ) as gateway:
            status = await _post_simple_message(gateway)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.copilot.sdk.codex_compat_gateway"
    ]
    assert status == 502
    assert len(messages) == 1
    assert "exception_type=RuntimeError" in messages[0]
    assert "app-server overloaded while starting turn" in messages[0]
    assert len(messages[0]) < 340
    assert messages[0].endswith("...")


@pytest.mark.asyncio
async def test_transport_failure_logs_redact_all_auth_shapes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signaturevalue"
    bearer = "bearer-top-secret-123"
    access_token = "access-top-secret-456"
    refresh_token = "refresh-top-secret-789"
    provider_secret = "provider-top-secret-012"
    device_code = "ABCD-EFGH"
    session = _FailingAgentSession(RuntimeError("not configured"))
    transport = _FakeTransport(session)
    with caplog.at_level(
        logging.ERROR,
        logger="backend.copilot.sdk.codex_compat_gateway",
    ):
        async with CodexAnthropicGateway(
            credential_lease=_lease(),
            model="gpt-5.6-terra",
            transport=transport,
        ) as gateway:
            capability = gateway.auth_token
            session.error = RuntimeError(
                f"upstream auth failed jwt={jwt} Authorization: Bearer {bearer} "
                f"access_token={access_token} "
                f'"refresh_token":"{refresh_token}" '
                f"device verification {device_code} capability={capability} "
                f'provider_state={{"tokens":{{"access_token":"{provider_secret}"}}}}'
            )
            status = await _post_simple_message(gateway)

    message = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.copilot.sdk.codex_compat_gateway"
    )
    assert status == 502
    assert "exception_type=RuntimeError" in message
    assert "upstream auth failed" in message
    assert "[REDACTED]" in message
    for secret in (
        jwt,
        bearer,
        access_token,
        refresh_token,
        provider_secret,
        device_code,
        capability,
    ):
        assert secret not in message


@pytest.mark.asyncio
async def test_round_trips_tool_use_through_claude_harness() -> None:
    original_tool_name = "mcp__copilot__add_understanding"
    session = _FakeAgentSession(use_tool=True)
    transport = _FakeTransport(session)
    async with CodexAnthropicGateway(
        credential_lease=_lease(),
        model="gpt-5.6-sol",
        transport=transport,
    ) as gateway:
        async with ClientSession() as client:
            first = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json={
                    "stream": True,
                    "messages": [{"role": "user", "content": "check"}],
                    "tools": [
                        {
                            "name": original_tool_name,
                            "description": "Read status",
                            "input_schema": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                        }
                    ],
                },
            )
            first_events = _events(await first.text())
            tool_start = next(
                event
                for event in first_events
                if event["type"] == "content_block_start"
            )
            content_block = tool_start["content_block"]
            assert content_block["type"] == "tool_use"
            assert content_block["name"] == original_tool_name
            assert content_block["input"] == {}
            gateway_call_id = content_block["id"]
            assert isinstance(gateway_call_id, str)
            assert gateway_call_id.startswith("toolu_codex_")
            assert gateway_call_id != "call-1"
            assert first_events[-2]["delta"] == {
                "stop_reason": "tool_use",
                "stop_sequence": None,
            }

            second = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json={
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "check"},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": gateway_call_id,
                                    "name": original_tool_name,
                                    "input": {"query": "status"},
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": gateway_call_id,
                                    "content": "all green",
                                }
                            ],
                        },
                    ],
                },
            )
            second_events = _events(await second.text())

        assert any(
            event.get("delta") == {"type": "text_delta", "text": "done"}
            for event in second_events
        )
        assert second_events[-2]["delta"] == {
            "stop_reason": "end_turn",
            "stop_sequence": None,
        }
        assert session.tool_result == CodexDynamicToolResult(
            content="all green",
            success=True,
        )
        assert session.tools[0][0].name != original_tool_name
        assert not session.tools[0][0].name.startswith("mcp__")


@pytest.mark.asyncio
async def test_messages_default_to_nonstreaming_and_report_estimated_input() -> None:
    transport = _FakeTransport(_FakeAgentSession())
    async with CodexAnthropicGateway(
        credential_lease=_lease(),
        model="gpt-5.6-luna",
        transport=transport,
    ) as gateway:
        async with ClientSession() as client:
            response = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            payload = await response.json()
            count_response = await client.post(
                f"{gateway.base_url}/v1/messages/count_tokens",
                headers=_headers(gateway),
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            count_payload = await count_response.json()

    assert response.status == 200
    assert payload["type"] == "message"
    assert payload["content"] == [{"type": "text", "text": "hello"}]
    assert payload["usage"]["input_tokens"] > 0
    assert payload["usage"]["input_tokens"] == count_payload["input_tokens"]


@pytest.mark.asyncio
async def test_final_response_fallback_is_scoped_to_each_tool_boundary() -> None:
    transport = _FakeTransport(_FinalWithoutDeltaSession())
    async with CodexAnthropicGateway(
        credential_lease=_lease(),
        model="gpt-5.6-terra",
        transport=transport,
    ) as gateway:
        async with ClientSession() as client:
            first = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json=_tool_request("run", stream=True),
            )
            first_events = _events(await first.text())
            tool_start = next(
                event
                for event in first_events
                if event["type"] == "content_block_start"
                and event["content_block"]["type"] == "tool_use"
            )
            gateway_call_id = tool_start["content_block"]["id"]
            assert isinstance(gateway_call_id, str)

            second = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json=_tool_result_request(
                    gateway_call_id,
                    "ok",
                    stream=True,
                ),
            )
            second_events = _events(await second.text())

    assert any(
        event.get("delta") == {"type": "text_delta", "text": "before tool"}
        for event in first_events
    )
    assert any(
        event.get("delta") == {"type": "text_delta", "text": "after tool"}
        for event in second_events
    )


@pytest.mark.asyncio
async def test_raw_codex_call_id_collisions_are_isolated_by_gateway_ids() -> None:
    agent_session = _CollidingCallSession()
    transport = _FakeTransport(agent_session)
    async with CodexAnthropicGateway(
        credential_lease=_lease(),
        model="gpt-5.6-sol",
        transport=transport,
    ) as gateway:
        async with ClientSession() as client:
            first_response, second_response = await asyncio.gather(
                client.post(
                    f"{gateway.base_url}/v1/messages",
                    headers=_headers(gateway),
                    json=_tool_request("first"),
                ),
                client.post(
                    f"{gateway.base_url}/v1/messages",
                    headers=_headers(gateway),
                    json=_tool_request("second"),
                ),
            )
            first_payload, second_payload = await asyncio.gather(
                first_response.json(),
                second_response.json(),
            )
            first_id = _tool_use_id(first_payload)
            second_id = _tool_use_id(second_payload)

            cross_conversation = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": first_id,
                                    "content": "wrong-first",
                                },
                                {
                                    "type": "tool_result",
                                    "tool_use_id": second_id,
                                    "content": "wrong-second",
                                },
                            ],
                        }
                    ]
                },
            )

            first_result, second_result = await asyncio.gather(
                client.post(
                    f"{gateway.base_url}/v1/messages",
                    headers=_headers(gateway),
                    json=_tool_result_request(first_id, "first-result"),
                ),
                client.post(
                    f"{gateway.base_url}/v1/messages",
                    headers=_headers(gateway),
                    json=_tool_result_request(second_id, "second-result"),
                ),
            )
            first_completed, second_completed = await asyncio.gather(
                first_result.json(),
                second_result.json(),
            )

    assert first_id != second_id
    assert cross_conversation.status == 400
    assert first_id.startswith("toolu_codex_")
    assert second_id.startswith("toolu_codex_")
    assert agent_session.raw_call_ids == ["raw-collision", "raw-collision"]
    assert {result.content for result in agent_session.results.values()} == {
        "first-result",
        "second-result",
    }
    completed_text = {
        block["text"]
        for payload in (first_completed, second_completed)
        for block in payload["content"]
        if block["type"] == "text"
    }
    assert completed_text == {
        "completed:first-result",
        "completed:second-result",
    }


@pytest.mark.asyncio
async def test_duplicate_tool_result_request_is_claimed_once_without_new_turn() -> None:
    agent_session = _FakeAgentSession(use_tool=True)
    transport = _FakeTransport(agent_session)
    async with CodexAnthropicGateway(
        credential_lease=_lease(),
        model="gpt-5.6-terra",
        transport=transport,
    ) as gateway:
        async with ClientSession() as client:
            first = await client.post(
                f"{gateway.base_url}/v1/messages",
                headers=_headers(gateway),
                json=_tool_request("run"),
            )
            gateway_call_id = _tool_use_id(await first.json())
            continuation = _tool_result_request(gateway_call_id, "one result")

            responses = await asyncio.gather(
                client.post(
                    f"{gateway.base_url}/v1/messages",
                    headers=_headers(gateway),
                    json=continuation,
                ),
                client.post(
                    f"{gateway.base_url}/v1/messages",
                    headers=_headers(gateway),
                    json=continuation,
                ),
            )
            payloads = await asyncio.gather(
                *(response.json() for response in responses)
            )

    assert sorted(response.status for response in responses) == [200, 409]
    duplicate_payload = next(
        payload
        for response, payload in zip(responses, payloads, strict=True)
        if response.status == 409
    )
    assert duplicate_payload["error"]["message"] == (
        "This tool-result request was already accepted"
    )
    assert len(agent_session.requests) == 1
    assert agent_session.tool_result == CodexDynamicToolResult(
        content="one result",
        success=True,
    )


@pytest.mark.asyncio
async def test_rejects_missing_gateway_capability() -> None:
    transport = _FakeTransport(_FakeAgentSession())
    async with CodexAnthropicGateway(
        credential_lease=_lease(),
        model="gpt-5.6-luna",
        transport=transport,
    ) as gateway:
        async with ClientSession() as client:
            response = await client.post(
                f"{gateway.base_url}/v1/messages",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            payload = await response.json()
        assert response.status == 401
        assert payload["error"]["type"] == "authentication_error"
